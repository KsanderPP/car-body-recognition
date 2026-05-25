from pathlib import Path
import copy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "collected_crops" / "refined_split"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS_STAGE3 = 10
LR_STAGE3 = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

INPUT_CHECKPOINT = CHECKPOINT_DIR / "best_stage2_adapted.pth"
OUTPUT_CHECKPOINT = CHECKPOINT_DIR / "best_stage3.pth"
OUTPUT_HISTORY = LOGS_DIR / "history_stage3.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    epoch_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def main():
    if not INPUT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {INPUT_CHECKPOINT}")

    train_ds = ImageFolder(DATA_DIR / "train", transform=train_tf)
    valid_ds = ImageFolder(DATA_DIR / "valid", transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    checkpoint = torch.load(INPUT_CHECKPOINT, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    if train_ds.classes != class_names:
        raise ValueError(
            f"Class mismatch.\n"
            f"Train dataset classes: {train_ds.classes}\n"
            f"Checkpoint classes:    {class_names}"
        )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_STAGE3,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_valid_f1 = checkpoint.get("best_valid_f1", 0.0)
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    print("Classes:", class_names)
    print("Train samples:", len(train_ds))
    print("Valid samples:", len(valid_ds))
    print(f"Starting from checkpoint: {INPUT_CHECKPOINT.name}")
    print(f"Initial best_valid_f1 from checkpoint: {best_valid_f1:.4f}")

    for epoch in range(EPOCHS_STAGE3):
        print(f"\n[Stage 3] Epoch {epoch + 1}/{EPOCHS_STAGE3}")

        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc, valid_f1 = run_epoch(model, valid_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} valid_f1={valid_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_f1": valid_f1,
        })

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_valid_f1": best_valid_f1,
                    "source_checkpoint": str(INPUT_CHECKPOINT),
                },
                OUTPUT_CHECKPOINT
            )
            print("Saved new best stage 3 model.")

        scheduler.step(valid_f1)

    model.load_state_dict(best_model_wts)

    with open(OUTPUT_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation macro F1 after stage 3: {best_valid_f1:.4f}")
    print(f"Saved checkpoint: {OUTPUT_CHECKPOINT}")
    print(f"Saved history:    {OUTPUT_HISTORY}")


if __name__ == "__main__":
    main()
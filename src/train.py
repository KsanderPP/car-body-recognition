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
DATA_DIR = PROJECT_ROOT / "data" / "cars_body_type"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS_STAGE1 = 5
LR_STAGE1 = 1e-3
EPOCHS_STAGE2 = 8
LR_STAGE2 = 1e-4
NUM_WORKERS = 0
MODEL_NAME = "efficientnet_b0"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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

class_names = train_ds.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Train samples:", len(train_ds))
print("Valid samples:", len(valid_ds))


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes
)

for param in model.parameters():
    param.requires_grad = False

if hasattr(model, "classifier"):
    for param in model.classifier.parameters():
        param.requires_grad = True
elif hasattr(model, "fc"):
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE1,
    weight_decay=1e-4
)


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
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


best_valid_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
history = []

for epoch in range(EPOCHS_STAGE1):
    print(f"\nEpoch {epoch + 1}/{EPOCHS_STAGE1}")

    train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc, valid_f1 = run_epoch(model, valid_loader, criterion)

    print(
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
        f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} valid_f1={valid_f1:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
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
                "model_name": MODEL_NAME,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_valid_f1": best_valid_f1,
            },
            CHECKPOINT_DIR / "best_stage1.pth"
        )
        print("Saved new best model.")

model.load_state_dict(best_model_wts)

with open(LOGS_DIR / "history_stage1.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print(f"\nBest validation macro F1: {best_valid_f1:.4f}")
print("Training stage 1 finished.")

print("\nStarting stage 2 fine-tuning...")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_STAGE2,
    weight_decay=1e-4
)

best_valid_f1_stage2 = best_valid_f1
best_model_wts_stage2 = copy.deepcopy(model.state_dict())

history_stage2 = []

for epoch in range(EPOCHS_STAGE2):
    print(f"\n[Stage 2] Epoch {epoch + 1}/{EPOCHS_STAGE2}")

    train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc, valid_f1 = run_epoch(model, valid_loader, criterion)

    print(
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
        f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} valid_f1={valid_f1:.4f}"
    )

    history_stage2.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
    })

    if valid_f1 > best_valid_f1_stage2:
        best_valid_f1_stage2 = valid_f1
        best_model_wts_stage2 = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_name": MODEL_NAME,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_valid_f1": best_valid_f1_stage2,
            },
            CHECKPOINT_DIR / "best_stage2.pth"
        )
        print("Saved new best stage 2 model.")

model.load_state_dict(best_model_wts_stage2)

with open(LOGS_DIR / "history_stage2.json", "w", encoding="utf-8") as f:
    json.dump(history_stage2, f, indent=2)

print(f"\nBest validation macro F1 after stage 2: {best_valid_f1_stage2:.4f}")
print("Training stage 2 finished.")
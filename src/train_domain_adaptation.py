from pathlib import Path
import copy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DATA_DIR = PROJECT_ROOT / "data" / "cars_body_type"
ADAPT_DATA_DIR = PROJECT_ROOT / "collected_crops" / "split"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BASE_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_stage2.pth"

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 5
LR = 1e-5
NUM_WORKERS = 0

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


base_train_ds = ImageFolder(BASE_DATA_DIR / "train", transform=train_tf)
base_valid_ds = ImageFolder(BASE_DATA_DIR / "valid", transform=eval_tf)

adapt_train_ds = ImageFolder(ADAPT_DATA_DIR / "train", transform=train_tf)
adapt_valid_ds = ImageFolder(ADAPT_DATA_DIR / "valid", transform=eval_tf)

print("Base classes:", base_train_ds.classes)
print("Adapt classes:", adapt_train_ds.classes)

if base_train_ds.classes != adapt_train_ds.classes:
    raise ValueError("Class names/order in adaptation dataset must match base dataset.")

train_ds = ConcatDataset([base_train_ds, adapt_train_ds])
valid_ds = ConcatDataset([base_valid_ds, adapt_valid_ds])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]
model_name = checkpoint["model_name"]
num_classes = len(class_names)

model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


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

for epoch in range(EPOCHS):
    print(f"\nAdaptation epoch {epoch + 1}/{EPOCHS}")

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
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_valid_f1": best_valid_f1,
            },
            CHECKPOINT_DIR / "best_stage2_adapted.pth"
        )
        print("Saved new adapted model.")

model.load_state_dict(best_model_wts)

with open(LOGS_DIR / "history_adaptation.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print(f"\nBest adaptation macro F1: {best_valid_f1:.4f}")
print("Domain adaptation finished.")
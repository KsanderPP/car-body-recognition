from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPT_DATA_DIR = PROJECT_ROOT / "collected_crops" / "split"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 0

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_stage2_adapted.pth"
VALID_DIR = ADAPT_DATA_DIR / "valid"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

valid_ds = ImageFolder(VALID_DIR, transform=eval_tf)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]
num_classes = len(class_names)
model_name = checkpoint["model_name"]

print("Classes from checkpoint:", class_names)
print("Classes from valid set:", valid_ds.classes)
print("Valid samples:", len(valid_ds))

model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"valid accuracy: {acc:.4f}")
print(f"valid macro F1: {f1:.4f}")

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

with open(LOGS_DIR / "adapted_valid_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "valid_accuracy": acc,
            "valid_macro_f1": f1,
            "classification_report": report_dict
        },
        f,
        indent=2
    )

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Valid Set (Adapted Model)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix_adapted_valid.png", dpi=200)
plt.close()

print("Saved metrics and confusion matrix.")

cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("Normalized Confusion Matrix - Budapest Valid Set (Adapted Model)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix_adapted_valid_normalized.png", dpi=200)
plt.close()
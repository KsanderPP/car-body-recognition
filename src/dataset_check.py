from pathlib import Path
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cars_body_type"

train_dir = DATA_DIR / "train"
valid_dir = DATA_DIR / "valid"
test_dir = DATA_DIR / "test"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("train exists:", train_dir.exists())
print("valid exists:", valid_dir.exists())
print("test exists:", test_dir.exists())

train_ds = ImageFolder(train_dir)
valid_ds = ImageFolder(valid_dir)
test_ds = ImageFolder(test_dir)

print("Train classes:", train_ds.classes)
print("Valid classes:", valid_ds.classes)
print("Test classes:", test_ds.classes)

print("Train size:", len(train_ds))
print("Valid size:", len(valid_ds))
print("Test size:", len(test_ds))

print("Class to idx:", train_ds.class_to_idx)
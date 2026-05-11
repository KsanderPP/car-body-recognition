from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_ROOT / "collected_crops" / "labeled"
SPLIT_DIR = PROJECT_ROOT / "collected_crops" / "split"

TRAIN_RATIO = 0.8
SEED = 42


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(SEED)

    train_dir = SPLIT_DIR / "train"
    valid_dir = SPLIT_DIR / "valid"

    clear_dir(train_dir)
    clear_dir(valid_dir)

    class_dirs = [d for d in LABELED_DIR.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.*"))
        images = [p for p in images if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        if len(images) == 0:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        train_images = images[:split_idx]
        valid_images = images[split_idx:]

        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (valid_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)

        for img in valid_images:
            shutil.copy2(img, valid_dir / class_name / img.name)

        print(f"{class_name}: train={len(train_images)}, valid={len(valid_images)}")

    print("Split prepared.")


if __name__ == "__main__":
    main()
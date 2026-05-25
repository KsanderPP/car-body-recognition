from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "collected_crops" / "split" / "test"
TARGET_DIR = PROJECT_ROOT / "collected_crops" / "refined_split"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_image_files(class_dir: Path):
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def ensure_empty_structure(class_names):
    if TARGET_DIR.exists():
        raise FileExistsError(
            f"Target directory already exists: {TARGET_DIR}\n"
            f"Usuń go ręcznie albo zmień TARGET_DIR."
        )

    for split_name in ["train", "valid", "test"]:
        for class_name in class_names:
            (TARGET_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)


def copy_files(files, destination_dir: Path):
    for src_path in files:
        dst_path = destination_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    random.seed(RANDOM_SEED)

    class_dirs = sorted([p for p in SOURCE_DIR.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in: {SOURCE_DIR}")

    class_names = [p.name for p in class_dirs]
    ensure_empty_structure(class_names)

    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Classes: {class_names}\n")

    total_counts = {"train": 0, "valid": 0, "test": 0}

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = get_image_files(class_dir)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        n_test = n - n_train - n_valid

        train_files = files[:n_train]
        valid_files = files[n_train:n_train + n_valid]
        test_files = files[n_train + n_valid:]

        copy_files(train_files, TARGET_DIR / "train" / class_name)
        copy_files(valid_files, TARGET_DIR / "valid" / class_name)
        copy_files(test_files, TARGET_DIR / "test" / class_name)

        total_counts["train"] += len(train_files)
        total_counts["valid"] += len(valid_files)
        total_counts["test"] += len(test_files)

        print(
            f"{class_name:<15} total={n:<4} "
            f"train={len(train_files):<4} "
            f"valid={len(valid_files):<4} "
            f"test={len(test_files):<4}"
        )

    print("\nDone.")
    print(
        f"Total -> train={total_counts['train']}, "
        f"valid={total_counts['valid']}, "
        f"test={total_counts['test']}"
    )


if __name__ == "__main__":
    main()
"""Split labeled images into the YOLO dataset/images|labels/train|val layout."""

import argparse
import hashlib
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_dir", type=Path, help="Folder with the source images, e.g. images/")
    parser.add_argument("labels_dir", type=Path, help="Folder with the YOLO .txt labels exported from makesense.ai")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Output dataset root.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Fraction of images held out for validation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def assign_split(stem: str, seed: int, val_ratio: float) -> str:
    # Deterministic per-image assignment (hash of the filename, not a global
    # shuffle) so re-running after adding new images doesn't move images
    # that were already split into dataset/.
    digest = hashlib.md5(f"{seed}:{stem}".encode()).hexdigest()
    bucket = int(digest, 16) % 1000
    return "val" if bucket < val_ratio * 1000 else "train"


def main() -> None:
    args = parse_args()

    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    missing = [p.name for p in images if not (args.labels_dir / f"{p.stem}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing YOLO .txt label for: {', '.join(missing)}")

    splits: dict[str, list[Path]] = {"train": [], "val": []}
    for image_path in images:
        splits[assign_split(image_path.stem, args.seed, args.val_ratio)].append(image_path)

    for split, split_images in splits.items():
        image_dir = args.dataset_dir / "images" / split
        label_dir = args.dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            shutil.copy2(image_path, image_dir / image_path.name)
            label_path = args.labels_dir / f"{image_path.stem}.txt"
            shutil.copy2(label_path, label_dir / label_path.name)

    print(f"train: {len(splits['train'])} images, val: {len(splits['val'])} images -> {args.dataset_dir}")


if __name__ == "__main__":
    main()

"""Fine-tune a YOLOv8 model to detect wood-piece end-faces."""

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA_CONFIG = "dataset/data.yaml"
DEFAULT_BASE_MODEL = "yolov8s.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 4
DEFAULT_RUN_NAME = "wood_counter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA_CONFIG), help="Path to the YOLO dataset config.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Pretrained weights to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="Training device, e.g. 'mps', 'cuda:0', or 'cpu' (auto-detected if omitted).")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Name of this training run under runs/detect/.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience in epochs (Ultralytics default: 100). Pass 0 to disable and run the full --epochs count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}\nRun split_dataset.py first.")

    model = YOLO(args.base_model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"training complete, best weights at: {best_weights}")
    print(f"copy it to the project root as a candidate to review: cp {best_weights} model_new.pt")
    print("once validated, replace the in-use model: mv model_new.pt model.pt")


if __name__ == "__main__":
    main()

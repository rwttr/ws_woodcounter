"""Count wood pieces in an image using a YOLOv8 model with SAHI sliced inference."""

import argparse
from pathlib import Path

import yaml

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_IMAGE_PATH = "images/sample1.jpg"
DEFAULT_EXPORT_DIR = "results"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_DEVICE = "mps"

# Sliding-window size and overlap used by SAHI to tile the image before inference.
SLICE_HEIGHT = 256
SLICE_WIDTH = 256
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# Wood end-faces are always noticeably wider than tall (observed width/height
# ratio 2.06-5.40 across 283 labeled boxes in sample1.jpg); reject detections
# outside a looser margin around that range as implausible false positives.
DEFAULT_MIN_ASPECT_RATIO = 1.5
DEFAULT_MAX_ASPECT_RATIO = 6.0

# Known physical size of a piece, e.g. "4 x 2 in" - not measured from the
# image (no scale reference exists in these photos), just displayed as a fact.
DEFAULT_PIECE_SIZE = "4 x 2 in"

HULL_COLOR_BGR = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count wood pieces in an image via SAHI + YOLOv8.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file (config.yaml). CLI flags override config values.")
    parser.add_argument("--image", type=Path, default=None, help="Path to the input image.")
    parser.add_argument("--model", type=Path, default=None, help="Path to the YOLOv8 weights file.")
    parser.add_argument("--output-txt", type=Path, default=None, help="Text file to write the detected piece count to.")
    parser.add_argument("--export-dir", type=Path, default=Path(DEFAULT_EXPORT_DIR), help="Directory to save the annotated result image.")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 'cuda:0', 'cpu', or 'mps'.")
    parser.add_argument("--min-aspect-ratio", type=float, default=DEFAULT_MIN_ASPECT_RATIO, help="Reject detections narrower (width/height) than this ratio.")
    parser.add_argument("--max-aspect-ratio", type=float, default=DEFAULT_MAX_ASPECT_RATIO, help="Reject detections wider (width/height) than this ratio.")
    parser.add_argument("--piece-size", default=DEFAULT_PIECE_SIZE, help="Known physical size of a piece to display, e.g. '4 x 2 in'.")
    args = parser.parse_args()

    # Load YAML config and apply as defaults for values not set via CLI.
    cfg: dict = {}
    if args.config is not None:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}

    if args.image is None:
        args.image = Path(cfg.get("image_path", DEFAULT_IMAGE_PATH))
    if args.model is None:
        args.model = Path(cfg.get("model", DEFAULT_MODEL_PATH))
    if args.output_txt is None and "output_txt" in cfg:
        args.output_txt = Path(cfg["output_txt"])
    if args.confidence is None:
        args.confidence = float(cfg.get("confidence", DEFAULT_CONFIDENCE_THRESHOLD))
    if args.device is None:
        args.device = cfg.get("device", DEFAULT_DEVICE)

    return args


def draw_result_overlay(image_path: Path, lines: list[str], hull_box: tuple[float, float, float, float] | None) -> None:
    image = cv2.imread(str(image_path))

    if hull_box is not None:
        x1, y1, x2, y2 = (int(v) for v in hull_box)
        hull_thickness = max(int(image.shape[1] / 300), 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), HULL_COLOR_BGR, hull_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(image.shape[1] / 900, 1.0)
    thickness = max(int(font_scale * 2), 2)
    margin = int(10 * font_scale)

    sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
    line_height = max(text_height + baseline for (_, text_height), baseline in sizes)
    banner_width = max(text_width for (text_width, _), _ in sizes) + 2 * margin
    banner_height = line_height * len(lines) + 2 * margin
    cv2.rectangle(image, (0, 0), (banner_width, banner_height), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        baseline_y = margin + line_height * i + line_height - int(margin / 2)
        cv2.putText(image, line, (margin, baseline_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(str(image_path), image)


def count_wood_pieces(
    image_path: Path,
    model_path: Path,
    export_dir: Path,
    confidence: float,
    device: str,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    piece_size: str,
) -> int:
    """Run sliced detection on image_path and return the number of wood pieces found."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Pass a YOLOv8 model trained on wood end-faces with --model, "
            "e.g. --model best.pt"
        )

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=confidence,
        device=device,
    )

    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
    )

    def aspect_ratio(prediction) -> float:
        box = prediction.bbox
        return (box.maxx - box.minx) / (box.maxy - box.miny)

    result.object_prediction_list = [
        prediction
        for prediction in result.object_prediction_list
        if min_aspect_ratio <= aspect_ratio(prediction) <= max_aspect_ratio
    ]
    count = len(result.object_prediction_list)

    info_lines = [f"wood count: {count}", f"piece size: {piece_size}"]
    hull_box = None
    if count > 0:
        avg_ratio = sum(aspect_ratio(p) for p in result.object_prediction_list) / count
        info_lines.append(f"avg w:h = {avg_ratio:.1f}:1")
        boxes = [p.bbox for p in result.object_prediction_list]
        hull_box = (
            min(box.minx for box in boxes),
            min(box.miny for box in boxes),
            max(box.maxx for box in boxes),
            max(box.maxy for box in boxes),
        )

    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / "sahi_result_image.png"
    result.export_visuals(export_dir=str(export_dir), file_name="sahi_result_image", hide_labels=True, hide_conf=True)
    draw_result_overlay(output_path, info_lines, hull_box)

    return count


def main() -> None:
    args = parse_args()
    count = count_wood_pieces(
        image_path=args.image,
        model_path=args.model,
        export_dir=args.export_dir,
        confidence=args.confidence,
        device=args.device,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        piece_size=args.piece_size,
    )
    print(f"wood count: {count} pieces")
    print(f"annotated result saved to {args.export_dir / 'sahi_result_image.png'}")

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_txt, "w") as fh:
            fh.write(f"{count}\n")
        print(f"count written to {args.output_txt}")


if __name__ == "__main__":
    main()

# Wood Counter

Counts square wood pieces in warehouse stack photos by detecting the visible
end-grain face of each piece with a YOLOv8 model.
[SAHI](https://github.com/obss/sahi) tiles large images into overlapping
slices so small, densely packed pieces are still reliably detected.

## Project layout

```
predict.py          Runs SAHI-sliced detection on an image and counts pieces
train.py            Fine-tunes YOLOv8 on the labeled dataset
split_dataset.py    Splits labeled images into dataset/images|labels/train|val
config.yaml         Runtime configuration (image path, model, output file)
requirements.txt    Python dependencies, incl. CPU-only PyTorch
classes.txt         Label list for the annotation tool (single class: wood)
model.pt            Trained weights, committed — ready for inference out of the box
yolov8s.pt          Pretrained COCO weights used as the training base model (not committed)
dataset/            YOLO-format training data
  data.yaml         Dataset config consumed by train.py
  images/{train,val}
  labels/{train,val}
images/             Source photos of wood stacks (sample1.jpg committed as a test image)
results/            Annotated output images and count .txt files from predict.py (empty in the repo, tracked via .gitkeep)
runs/               Ultralytics training artefacts
```

## Setup

> The `_venv/` directory is local only and is not committed. Create a fresh
> virtual environment on every machine.

**Requires Python 3.14** (developed and tested against 3.14.6).

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins CPU-only PyTorch wheels (`torch==2.13.0`,
`torchvision==0.28.0`) since the target deployment is a CPU-only remote
server. If you're setting up on a machine with an NVIDIA GPU or Apple
Silicon and want GPU/MPS acceleration instead, install the matching
PyTorch build first, then install the rest:

| Hardware | Install command |
|---|---|
| NVIDIA GPU (CUDA 12.x) | `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu121` |
| Apple Silicon (MPS) | `pip install torch==2.13.0 torchvision==0.28.0` |

Then install the rest without re-resolving torch:

```bash
pip install -r requirements.txt --no-deps ultralytics sahi opencv-python pyyaml
```

For other CUDA versions visit <https://pytorch.org/get-started/locally/>.

## Configuration

Edit `config.yaml` before running predictions:

```yaml
model: model.pt       # path to trained YOLOv8 weights
image_dir: images     # directory containing the input image (exactly one file)
output_dir: results   # directory where <image>_count.txt and the annotated image are written
device: cpu           # 'cpu', 'cuda:0', or 'mps'
confidence: 0.5       # detection confidence threshold
```

`image_dir` must contain exactly one image — drop the photo to process there
and remove any others. All fields can also be overridden directly on the
command line — CLI flags take precedence over the config file, and `--image`
can point at a specific file to bypass directory auto-discovery entirely.

## Running predictions

```bash
# Using the config file (processes the single image in images/)
python predict.py --config config.yaml

# Override individual values without editing config.yaml
python predict.py --config config.yaml --image-dir /path/to/incoming --device cuda:0

# Point directly at a specific image file, skipping directory discovery
python predict.py --image images/sample1.jpg --model model.pt
```

The script:
- Prints the wood piece count to stdout.
- Saves an annotated PNG to `<output_dir>/sahi_result_image.png`.
- Writes the integer count to `<output_dir>/<image_stem>_count.txt`, e.g.
  `results/sample1_count.txt` (overwritten each run for that filename).

## Training workflow

### 1. Label images

Annotate each wood end-face bounding box with
[makesense.ai](https://www.makesense.ai) (runs fully in-browser):

1. Upload images from `images/`.
2. Load `classes.txt` as the label list.
3. Draw one box per visible wood piece end-face.
4. Export → **YOLO format**, unzip into `labels_export/`.

### 2. Split into train/val

```bash
python split_dataset.py images/ labels_export/
```

Copies matched image/label pairs into `dataset/images/{train,val}` and
`dataset/labels/{train,val}` (default 80/20 split, deterministic per filename).

### 3. Train

```bash
python train.py
# Key flags: --epochs, --batch, --imgsz, --device, --base-model
```

After training copy the best weights to the project root:

```bash
cp runs/detect/wood_counter/weights/best.pt model.pt
```

### 4. Predict

Update `config.yaml` with the desired image and run:

```bash
python predict.py --config config.yaml
```

## Notes

- `model.pt` (the current trained weights) is committed to the repo so the
  model is ready to use for inference right after cloning — no training
  required. `images/sample1.jpg` is committed as a ready-to-run test input;
  `results/` is tracked as an empty folder (`.gitkeep`) since its contents
  are generated output and gitignored.
- The default `device` in `config.yaml` is `cpu` for remote-server
  compatibility. Change to `cuda:0` to use a GPU.
- This is a POC trained on a small set of photos — expect it to overfit to
  these specific lighting conditions, stack types, and wood dimensions.

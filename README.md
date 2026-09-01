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
requirements.txt    Python dependencies (PyTorch installed separately)
classes.txt         Label list for the annotation tool (single class: wood)
best.pt             Trained weights, committed — ready for inference out of the box
yolov8s.pt          Pretrained COCO weights used as the training base model (not committed)
dataset/            YOLO-format training data
  data.yaml         Dataset config consumed by train.py
  images/{train,val}
  labels/{train,val}
images/             Source photos of wood stacks (empty in the repo, tracked via .gitkeep)
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

### 2. Install PyTorch

PyTorch must be installed before the other dependencies so the correct
GPU/CPU build is selected for your hardware. Tested versions:
`torch==2.13.0`, `torchvision==0.28.0`.

| Hardware | Install command |
|---|---|
| NVIDIA GPU (CUDA 12.x) | `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu121` |
| CPU only | `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu` |
| Apple Silicon (MPS) | `pip install torch==2.13.0 torchvision==0.28.0` |

For other CUDA versions visit <https://pytorch.org/get-started/locally/>.

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` before running predictions:

```yaml
model: best.pt          # path to trained YOLOv8 weights
image_path: images/sample1.jpg  # input image
output_txt: results/count.txt   # file where the piece count is written
device: cpu             # 'cpu', 'cuda:0', or 'mps'
confidence: 0.5         # detection confidence threshold
```

All fields can also be overridden directly on the command line — CLI flags
take precedence over the config file.

## Running predictions

```bash
# Using the config file
python predict.py --config config.yaml

# Override individual values without editing config.yaml
python predict.py --config config.yaml --image images/sample2.jpg --device cuda:0

# Without a config file (all defaults from predict.py constants)
python predict.py --image images/sample1.jpg --model best.pt
```

The script:
- Prints the wood piece count to stdout.
- Saves an annotated PNG to `results/sahi_result_image.png`.
- Writes the integer count to the file specified by `output_txt` (one number
  per line, so repeated runs append cleanly if you redirect stdout, but the
  file is overwritten each run).

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
cp runs/detect/wood_counter/weights/best.pt best.pt
```

### 4. Predict

Update `config.yaml` with the desired image and run:

```bash
python predict.py --config config.yaml
```

## Notes

- `best.pt` (the current trained weights) is committed to the repo so the
  model is ready to use for inference right after cloning — no training
  required. `images/` and `results/` are tracked as empty folders
  (`.gitkeep`) for clarity; their contents (source photos, prediction
  outputs) are gitignored.
- The default `device` in `config.yaml` is `cpu` for remote-server
  compatibility. Change to `cuda:0` to use a GPU.
- This is a POC trained on a small set of photos — expect it to overfit to
  these specific lighting conditions, stack types, and wood dimensions.

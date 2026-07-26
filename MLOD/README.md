# MLOD

**MLOD** is a modular implementation of a multi-label Cascade R-CNN for
vertebral lesion detection on sagittal spine MRI.

The original monolithic training and inference programs were separated into
configuration, data loading, model, loss, metric, training, inference, logging,
and utility modules.

## Repository structure

```text
MLOD/
├── config/
│   ├── default.yaml
│   └── thresholds.example.json
├── data/
│   └── README.md
├── log/
│   └── .gitkeep
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── profile_dataset.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── layers.py
│   │   └── mlod.py
│   ├── inference.py
│   ├── losses.py
│   ├── metrics.py
│   └── trainer.py
├── utils/
│   ├── boxes.py
│   ├── checkpoint.py
│   ├── dataset_profile.py
│   ├── logging.py
│   └── reproducibility.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd MLOD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Set `data.root` in `config/default.yaml`, or override it with `--data-root`.

```text
dataset_root/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Annotation format:

```text
<class_ids> <center_x> <center_y> <width> <height>
```

Example:

```text
0,2 0.512 0.486 0.238 0.164
```

The clinical dataset is not distributed through this repository.

## Training

```bash
python scripts/train.py \
  --config config/default.yaml \
  --data-root /path/to/datasets_yolo_8_1_1 \
  --device cuda
```

The best checkpoint is selected by validation `mAP@0.50`.

## Evaluation

```bash
python scripts/evaluate.py \
  --config config/default.yaml \
  --data-root /path/to/datasets_yolo_8_1_1 \
  --checkpoint outputs/checkpoints/best_map50.pth \
  --split test
```

## Inference

```bash
python scripts/infer.py \
  --config config/default.yaml \
  --image /path/to/image.jpg \
  --checkpoint outputs/checkpoints/best_map50.pth \
  --thresholds config/thresholds.example.json \
  --output outputs/inference/result.jpg
```

## Dataset profiling

```bash
python scripts/profile_dataset.py \
  --root /path/to/datasets_yolo_8_1_1 \
  --output-dir outputs/dataset_profile
```

## Architecture

- ConvNeXt-Large backbone
- Optional two-layer convolutional neck
- Configurable BatchNorm2d or LayerNorm2d
- Configurable ReLU or GELU
- Region Proposal Network
- Variable-stage Cascade R-CNN head
- Multi-label sigmoid classification
- Label-relation attention
- Asymmetric Loss with class-balanced weighting
- Smooth L1 box regression

## Important compatibility note

The following settings must match the checkpoint:

- Number of classes
- Number of cascade stages
- Neck normalization
- Neck activation
- Neck channel count
- Label-attention configuration

## Privacy

Do not commit clinical images, patient identifiers, DICOM metadata, patient
mapping files, model checkpoints, or raw evaluation files containing server
paths.

# YOLOv8 Mask Detector

Real-time 3-class face mask detection system based on YOLOv8. Detects:

- **with_mask** (0)
- **without_mask** (1)
- **mask_weared_incorrect** (2)

## Dataset

The dataset is sourced from Kaggle ([andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)) with **853 annotated images** in PASCAL VOC format.

### Preprocessing (already completed)

The raw dataset has been converted and split:

| Step | Script | Status |
|------|--------|--------|
| Download raw VOC data | `scripts/download_dataset.py` | Completed |
| Convert VOC → YOLO + stratified 80/10/10 split | `scripts/voc2yolo.py` | Completed |

**Processed data** is in `data/processed/`:

| Split | Images | Labels |
|-------|--------|--------|
| Train | 682 | 682 |
| Val   | 86  | 86  |
| Test  | 85  | 85  |

Class distribution (train+val+test):

| Class | Instances |
|-------|-----------|
| with_mask (0) | 3232 |
| without_mask (1) | 717 |
| mask_weared_incorrect (2) | 123 |

The dataset exhibits a long-tail distribution (imbalance ratio ~26:1). See `reports/figures/longtail_distribution.png` for a visualization.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (nbstripout for notebook output cleaning)
pre-commit install

# (Optional) Enable nbdime for git diff on notebooks
nbdime config-git --enable --global
```

## Quick Start

### 1. Download and prepare dataset (if starting from scratch)

```bash
python scripts/download_dataset.py
python scripts/voc2yolo.py
```

### 2. Analyze class distribution

```bash
python scripts/analyze_data.py
# Output: reports/figures/longtail_distribution.png
```

### 3. (Optional) Augment minority class

```bash
python data/augment.py
# Output: data/augmented/images/ and data/augmented/labels/
```

### 4. Train

```bash
yolo detect train data=configs/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 5. Evaluate

```bash
yolo detect val data=configs/dataset.yaml model=runs/detect/train/weights/best.pt split=test
```

### 6. Inference

```bash
# Single image
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# Webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

## Project Structure

```
├── configs/             # YOLOv8 dataset/model YAML configs
├── scripts/             # Data pipeline scripts (download, convert, analyze)
├── data/
│   ├── raw/             # Original Kaggle dataset (gitignored)
│   ├── processed/       # Converted YOLO-format dataset
│   └── augmented/       # Oversampled dataset for minority class
├── logs/                # Training logs and TensorBoard events (gitignored)
└── reports/figures/     # Evaluation plots and analysis outputs
```

## License

MIT. See [LICENSE](LICENSE).

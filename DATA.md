# D4RT Dataset Guide

This guide covers downloading and preparing datasets for training and evaluating D4RT.

## Quick Start

For initial training, we recommend **PointOdyssey** - it has complete ground truth (RGB, depth, 3D tracks, camera poses) and is readily available.

```bash
# Download sample set (3.1GB) for quick testing
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0 -O ./data/pointodyssey --remaining-ok

# Or download just the sample
gdown 1SAMPLE_FILE_ID -O ./data/pointodyssey/sample.tar.gz
```

---

## Datasets Overview

| Dataset | Type | Size | Ground Truth | Best For |
|---------|------|------|--------------|----------|
| **PointOdyssey** | Synthetic | ~170GB | RGB, Depth, 3D tracks, Normals, Cameras | Training (recommended) |
| **TartanAir** | Synthetic | ~3TB (full) | RGB, Depth, Flow, Poses, Segmentation | Large-scale training |
| **Sintel** | Synthetic | ~7GB | RGB, Depth, Flow, Cameras | Evaluation, fine-tuning |
| **ScanNet** | Real indoor | ~1.5TB | RGB-D, Poses | Real-world fine-tuning |

---

## 1. PointOdyssey (Recommended)

Large-scale synthetic dataset for long-term point tracking with complete 3D annotations.

**Paper**: [PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking](https://arxiv.org/abs/2307.15055) (ICCV 2023)

**Contents**:
- 159 videos (~200,000 frames)
- ~20,000 trajectories per video
- RGB, depth, instance segmentation, surface normals
- Camera intrinsics and extrinsics
- 2D and 3D point trajectories with visibility

### Download Options

#### Option A: Google Drive (Recommended)

```bash
# Install gdown
pip install gdown

# Download from Google Drive folder
# Full folder: https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0

# Sample (3.1GB) - for quick testing
gdown "https://drive.google.com/uc?id=<SAMPLE_FILE_ID>" -O sample.tar.gz

# Validation (19GB)
gdown "https://drive.google.com/uc?id=<VAL_FILE_ID>" -O val.tar.gz

# Training (125GB) - single file or split parts
gdown "https://drive.google.com/uc?id=<TRAIN_FILE_ID>" -O train.tar.gz

# Or download the entire folder
gdown --folder https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0
```

#### Option B: HuggingFace

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download entire dataset
huggingface-cli download aharley/pointodyssey --repo-type dataset --local-dir ./data/pointodyssey

# Or download specific files
huggingface-cli download aharley/pointodyssey sample.tar.gz --repo-type dataset --local-dir ./data/pointodyssey
huggingface-cli download aharley/pointodyssey val.tar.gz --repo-type dataset --local-dir ./data/pointodyssey
```

**File sizes on HuggingFace**:
- `sample.tar.gz`: 3.32 GB
- `val.tar.gz`: 20.4 GB
- `test.tar.gz`: 26.5 GB
- `train.tar.gz.partaa`: 34.4 GB
- `train.tar.gz.partab`: 34.4 GB
- `train.tar.gz.partac`: 34.4 GB
- `train.tar.gz.partad`: 31.3 GB

### Extract and Setup

```bash
cd data/pointodyssey

# Extract sample/val/test
tar -xzf sample.tar.gz
tar -xzf val.tar.gz
tar -xzf test.tar.gz

# For split training files, combine first then extract
cat train.tar.gz.parta* > train.tar.gz
tar -xzf train.tar.gz
```

### Directory Structure

```
pointodyssey/
├── train/
│   ├── sequence_001/
│   │   ├── rgbs/           # RGB frames (*.jpg)
│   │   ├── depths/         # Depth maps (*.npy)
│   │   ├── normals/        # Surface normals
│   │   ├── masks/          # Instance segmentation
│   │   ├── anno.npz        # Annotations (trajs_2d, trajs_3d, visibility)
│   │   └── intrinsics.npy  # Camera intrinsics
│   └── ...
├── val/
└── test/
```

### Usage

```python
from data.video_dataset import PointOdysseyDataset

dataset = PointOdysseyDataset(
    data_root="./data/pointodyssey",
    split="train",  # or "val", "test", "sample"
    num_frames=24,
    img_size=256,
    num_queries=2048
)

sample = dataset[0]
# sample contains: video, depth, tracks (3D), visibility, intrinsics
```

### Links
- **Project Page**: https://pointodyssey.com/
- **HuggingFace**: https://huggingface.co/datasets/aharley/pointodyssey
- **Google Drive**: https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0
- **GitHub**: https://github.com/y-zheng18/point_odyssey

---

## 2. TartanAir

Large-scale synthetic dataset with diverse environments and complete sensor data.

**Paper**: [TartanAir: A Dataset to Push the Limits of Visual SLAM](https://arxiv.org/abs/2003.14338)

**Contents**:
- Multiple environments (indoor, outdoor, fantasy)
- RGB, depth, optical flow, semantic segmentation
- Camera poses, IMU data
- LiDAR point clouds (V2)

### Download

```bash
# Install TartanAir toolkit
pip install tartanair

# Initialize and download
python -c "
import tartanair as ta

# Set your data directory
ta.init('/path/to/tartanair')

# List available environments
ta.list_envs()

# Download specific environment (recommended to start)
ta.download(
    env='AbandonedCableExposure',
    difficulty=['easy', 'hard'],
    modality=['image', 'depth', 'seg', 'flow'],
    camera=['lcam_front']
)
"
```

### Download Specific Environments

```python
import tartanair as ta

ta.init('./data/tartanair')

# Recommended environments for training
environments = [
    'AbandonedCableExposure',
    'AbandonedFactory',
    'AbandonedSchool',
    'AmericanDiner',
    'AmusementPark'
]

for env in environments:
    ta.download(
        env=env,
        difficulty=['easy'],
        modality=['image', 'depth'],
        camera=['lcam_front']
    )
```

### Directory Structure

```
tartanair/
├── AbandonedCableExposure/
│   ├── Easy/
│   │   ├── P000/
│   │   │   ├── image_lcam_front/    # RGB images
│   │   │   ├── depth_lcam_front/    # Depth maps
│   │   │   ├── seg_lcam_front/      # Segmentation
│   │   │   ├── flow_lcam_front/     # Optical flow
│   │   │   └── pose_lcam_front.txt  # Camera poses
│   │   └── ...
│   └── Hard/
└── ...
```

### Links
- **Project Page**: https://tartanair.org/
- **GitHub**: https://github.com/castacks/tartanair_tools
- **License**: CC BY 4.0

---

## 3. MPI Sintel

High-quality synthetic dataset derived from the open-source animated film "Sintel".

**Paper**: [A Naturalistic Open Source Movie for Optical Flow Evaluation](https://files.is.tue.mpg.de/black/papers/ButlerECCV2012.pdf) (ECCV 2012)

**Contents**:
- 23 training sequences, 12 test sequences
- Clean and Final rendering passes
- Depth maps, optical flow, camera matrices
- Motion blur, atmospheric effects, specular reflections

### Download

```bash
mkdir -p data/sintel && cd data/sintel

# Main dataset with flow (~5.3GB)
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip MPI-Sintel-complete.zip

# US Mirror (faster for US users)
# wget http://sintel.cs.washington.edu/MPI-Sintel-complete.zip

# Depth training data
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-depth-training-20150305.zip
unzip MPI-Sintel-depth-training-20150305.zip

# Camera data (intrinsics + extrinsics)
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
unzip MPI-Sintel-training_extras.zip
```

### Directory Structure

```
sintel/
├── training/
│   ├── clean/              # Clean pass images
│   │   ├── alley_1/
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   └── ...
│   ├── final/              # Final pass (with effects)
│   ├── flow/               # Optical flow (.flo)
│   ├── depth/              # Depth maps (.dpt)
│   ├── camdata_left/       # Camera matrices
│   └── invalid/            # Invalid pixel masks
└── test/
    ├── clean/
    └── final/
```

### Usage

```python
from data.video_dataset import SintelDataset

dataset = SintelDataset(
    data_root="./data/sintel",
    split="training",
    pass_name="final",  # or "clean"
    num_frames=24,
    img_size=256
)
```

### Links
- **Project Page**: http://sintel.is.tue.mpg.de/
- **Downloads**: http://sintel.is.tue.mpg.de/downloads
- **Depth Data**: http://sintel.is.tue.mpg.de/depth

---

## 4. ScanNet (Real Indoor)

Large-scale real-world RGB-D dataset of indoor scenes.

**Paper**: [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405) (CVPR 2017)

**Contents**:
- 1513 training scenes, 100 validation scenes
- RGB-D video sequences
- Camera poses (from BundleFusion)
- 3D semantic/instance segmentation

### Download

**Requires signing Terms of Use**: http://www.scan-net.org/

```bash
# After approval, download the script
# Download from: https://github.com/ScanNet/ScanNet

# Clone repo
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet

# Download specific scenes (after getting download script)
python download-scannet.py -o ./data/scannet --type .sens
python download-scannet.py -o ./data/scannet --type _2d-instance-filt.zip
```

### Directory Structure

```
scannet/
├── scans/
│   ├── scene0000_00/
│   │   ├── color/          # RGB frames (*.jpg)
│   │   ├── depth/          # Depth maps (*.png, 16-bit, mm)
│   │   ├── pose/           # Camera poses (*.txt, 4x4 matrix)
│   │   ├── intrinsic/      # Camera intrinsics
│   │   └── ...
│   └── ...
├── scannetv2_train.txt
└── scannetv2_val.txt
```

### Usage

```python
from data.video_dataset import ScanNetDataset

dataset = ScanNetDataset(
    data_root="./data/scannet/scans",
    split="train",
    num_frames=24,
    frame_skip=10,  # ScanNet has many frames, skip for efficiency
    img_size=256
)
```

### Links
- **Project Page**: http://www.scan-net.org/
- **GitHub**: https://github.com/ScanNet/ScanNet

---

## Training Recommendations

### Stage 1: Initial Training
Start with **PointOdyssey sample** (3.1GB) to verify your setup:

```bash
python train.py \
    --config configs/d4rt_base.yaml \
    --data.train_dataset PointOdysseyDataset \
    --data.train_root ./data/pointodyssey \
    --data.train_split sample
```

### Stage 2: Full Training
Use full **PointOdyssey** training set:

```bash
python train.py \
    --config configs/d4rt_base.yaml \
    --data.train_dataset PointOdysseyDataset \
    --data.train_root ./data/pointodyssey \
    --data.train_split train
```

### Stage 3: Multi-Dataset Training
Combine datasets for better generalization:

```python
from torch.utils.data import ConcatDataset
from data.video_dataset import PointOdysseyDataset, SintelDataset

datasets = [
    PointOdysseyDataset("./data/pointodyssey", split="train"),
    SintelDataset("./data/sintel", split="training"),
]
combined = ConcatDataset(datasets)
```

### Stage 4: Real-World Fine-tuning
Fine-tune on **ScanNet** for real-world deployment:

```bash
python train.py \
    --config configs/d4rt_base.yaml \
    --data.train_dataset ScanNetDataset \
    --data.train_root ./data/scannet/scans \
    --model.pretrained checkpoints/d4rt_pointodyssey.pth \
    --training.lr 1e-5
```

---

## Storage Requirements

| Dataset | Download | Extracted | Recommended |
|---------|----------|-----------|-------------|
| PointOdyssey (sample) | 3.1 GB | ~5 GB | Yes - start here |
| PointOdyssey (full) | 170 GB | ~250 GB | Yes - full training |
| TartanAir (subset) | ~50 GB | ~80 GB | Optional |
| Sintel | ~7 GB | ~15 GB | Yes - evaluation |
| ScanNet | ~1.5 TB | ~2 TB | Optional - real data |

---

## Troubleshooting

### Download Issues

**Google Drive quota exceeded**:
```bash
# Use gdown with cookies or try HuggingFace alternative
gdown --fuzzy "URL"
```

**HuggingFace slow download**:
```bash
# Use hf_transfer for faster downloads
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ...
```

**Sintel certificate error**:
```bash
# Use wget with no-check-certificate
wget --no-check-certificate http://files.is.tue.mpg.de/sintel/...
```

### Data Loading Issues

**Out of memory**:
- Reduce `num_frames` (e.g., 16 instead of 48)
- Reduce `img_size` (e.g., 128 instead of 256)
- Reduce `num_queries` (e.g., 1024 instead of 2048)

**Slow loading**:
- Use SSD storage
- Enable `num_workers > 0` in DataLoader
- Pre-extract and cache processed data

---

## License Information

| Dataset | License |
|---------|---------|
| PointOdyssey | CC BY-NC-SA 4.0 |
| TartanAir | CC BY 4.0 |
| Sintel | Custom (research use) |
| ScanNet | Terms of Use (research only) |

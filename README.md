# D4RT: Dynamic 4D Reconstruction Transformer

PyTorch implementation of D4RT, a feedforward transformer for 4D scene reconstruction from video.

## Features

- **Unified query interface**: Single model handles depth estimation, point tracking, and 3D reconstruction
- **Query format**: `(u, v, t_src, t_tgt, t_cam)` - pixel coordinates and temporal indices
- **Efficient attention**: FlashAttention support via PyTorch 2.0+
- **Pretrained backbones**: Optional timm ViT initialization

## Installation

```bash
# Clone repository
git clone https://github.com/jiangyurong609/d4rt-pytorch.git
cd d4rt

# Install dependencies
pip install -r requirements.txt

# Optional: Install pytorch3d for optimized Umeyama alignment
pip install pytorch3d
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- timm >= 0.9.0

## Quick Start

```python
import torch
from models import D4RT

# Initialize model
model = D4RT(
    encoder_variant='base',
    img_size=256,
    temporal_size=24,
    decoder_depth=8
)

# Input video: (B, T, H, W, C)
video = torch.randn(1, 24, 256, 256, 3)

# Query points
coords = torch.rand(1, 100, 2)  # (u, v) in [0, 1]
t_src = torch.zeros(1, 100, dtype=torch.long)
t_tgt = torch.ones(1, 100, dtype=torch.long) * 10
t_cam = torch.zeros(1, 100, dtype=torch.long)

# Forward pass
outputs = model(video, coords, t_src, t_tgt, t_cam)

# Outputs: pos_3d, pos_2d, visibility, displacement, normal, confidence
print(outputs['pos_3d'].shape)  # (1, 100, 3)
```

## Data

See [DATA.md](DATA.md) for dataset download instructions.

**Recommended for training**: PointOdyssey (~170GB with full ground truth)

```bash
# Quick start with sample set (3.1GB)
pip install huggingface_hub
huggingface-cli download aharley/pointodyssey sample.tar.gz --repo-type dataset --local-dir ./data/pointodyssey
```

## Training

```bash
# Train on PointOdyssey
python train.py \
    --config configs/d4rt_base.yaml \
    --data.train_root ./data/pointodyssey \
    --data.train_split train

# Distributed training
torchrun --nproc_per_node=4 train.py \
    --config configs/d4rt_base.yaml
```

## Evaluation

```bash
# Evaluate depth estimation
python evaluate.py \
    --config configs/d4rt_base.yaml \
    --checkpoint checkpoints/d4rt_base.pth \
    --task depth \
    --data_root ./data/sintel

# Evaluate point tracking
python evaluate.py \
    --task tracking \
    --data_root ./data/pointodyssey
```

## Model Architecture

```
D4RT
├── Encoder (ViT-based)
│   ├── 3D Patch Embedding (t=2, h=8, w=8)
│   ├── Positional Embedding (spatial + temporal)
│   └── Transformer Blocks (interleaved local/global attention)
│
└── Decoder (Cross-attention)
    ├── Query Embedding
    │   ├── Fourier (u, v coordinates)
    │   ├── Timestep (t_src, t_tgt, t_cam)
    │   └── Patch (local RGB context)
    ├── Cross-Attention Blocks
    └── Output Heads (3D pos, 2D pos, visibility, etc.)
```

## Query Types

| Task | Query | Output |
|------|-------|--------|
| Depth | `(u, v, t, t, t)` | 3D position in camera frame |
| Tracking | `(u, v, t_src, t_tgt, t_src)` | 2D position at t_tgt |
| 3D Tracking | `(u, v, t_src, t_tgt, t_cam)` | 3D position at t_tgt in t_cam frame |
| Point Cloud | Grid queries at all frames | Dense 3D reconstruction |

## Project Structure

```
d4rt/
├── models/
│   ├── d4rt.py          # Main model
│   ├── encoder.py       # ViT encoder with timm support
│   ├── decoder.py       # Cross-attention decoder
│   └── embeddings.py    # Fourier, timestep, patch embeddings
├── losses/
│   └── losses.py        # Multi-task loss functions
├── data/
│   ├── dataset.py       # Base dataset and query sampler
│   └── video_dataset.py # Dataset implementations
├── utils/
│   ├── camera.py        # Umeyama alignment, pose estimation
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Visualization utilities
├── configs/
│   ├── d4rt_base.yaml
│   └── d4rt_large.yaml
├── train.py
├── evaluate.py
├── inference.py
├── DATA.md              # Dataset guide
└── requirements.txt
```

## Citation

```bibtex
@article{d4rt2024,
  title={D4RT: Dynamic 4D Reconstruction Transformer},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License

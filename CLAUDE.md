# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D4RT (Dynamic 4D Reconstruction and Tracking) is a feedforward transformer model for efficient 4D scene reconstruction from video. It jointly infers depth, spatiotemporal correspondence, and camera parameters through a unified query-based interface.

## Architecture

The model follows an encoder-decoder architecture:

1. **Encoder**: ViT-based encoder with interleaved local (frame-wise) and global self-attention layers
   - Processes video V ∈ R^(T×H×W×3) into Global Scene Representation F ∈ R^(N×C)
   - Supports ViT-B, ViT-L, ViT-H, ViT-g variants
   - Uses 2×16×16 spatio-temporal patch size

2. **Decoder**: Lightweight cross-attention transformer
   - Takes query q = (u, v, t_src, t_tgt, t_cam) and outputs 3D position P
   - Query components: Fourier embedding for (u,v) + discrete timestep embeddings + local 9×9 RGB patch embedding
   - Each query decoded independently (no self-attention between queries)

## Key Concepts

- **Query Interface**: A query (u, v, t_src, t_tgt, t_cam) asks "What is the 3D position of point (u,v) from frame t_src, at timestep t_tgt, in camera coordinate system t_cam?"
- **Unified Tasks**: Point tracking, point clouds, depth maps, camera extrinsics/intrinsics all derived from same query interface
- **Independent Decoding**: Queries don't interact, enabling efficient parallel inference

## Common Commands

```bash
# Training
python train.py --config configs/d4rt_base.yaml

# Training with specific encoder
python train.py --encoder vit_large --batch_size 1 --num_gpus 8

# Evaluation
python evaluate.py --checkpoint path/to/checkpoint.pth --dataset sintel
python evaluate.py --checkpoint path/to/checkpoint.pth --dataset scannet --task depth

# Single video inference
python inference.py --video path/to/video.mp4 --output_dir results/

# Run tests
pytest tests/
pytest tests/test_model.py -v
pytest tests/test_model.py::test_encoder -v
```

## Project Structure

```
d4rt/
├── models/
│   ├── d4rt.py          # Main D4RT model
│   ├── encoder.py       # ViT encoder with local/global attention
│   ├── decoder.py       # Cross-attention decoder
│   └── embeddings.py    # Fourier, patch, timestep embeddings
├── data/
│   ├── dataset.py       # Base dataset class
│   ├── video_dataset.py # Video loading and preprocessing
│   └── augmentations.py # Data augmentation
├── losses/
│   └── losses.py        # All loss functions
├── utils/
│   ├── camera.py        # Camera pose estimation (Umeyama algorithm)
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Visualization utilities
├── configs/             # Training configurations
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── inference.py         # Inference script
```

## Loss Functions

The model uses weighted sum of losses:
- **L_3D**: L1 loss on normalized 3D positions (primary)
- **L_2D**: L1 loss on 2D image coordinates
- **L_vis**: BCE for visibility prediction
- **L_disp**: L1 on motion displacement
- **L_normal**: Cosine similarity for surface normals
- **L_conf**: Confidence penalty (-log(c))

Default weights: λ_3D=1.0, λ_2D=0.1, λ_vis=0.1, λ_disp=0.1, λ_normal=0.5, λ_conf=0.2

## Training Details

- Optimizer: AdamW with weight decay 0.03
- Learning rate: Warmup to 1e-4 over 2500 steps, then cosine decay to 1e-6
- Batch size: 1 per device, 64 devices total
- Resolution: 256×256 (resized to square)
- Frames: 48 frames per clip
- Queries per batch: 2048 (30% sampled near depth/motion boundaries)
- Training steps: 500k

## Query Sampling for Different Tasks

| Task | u | v | t_src | t_tgt | t_cam |
|------|---|---|-------|-------|-------|
| Point Track | Fixed | Fixed | Fixed | 1..T | 1..T (=t_tgt) |
| Point Cloud | 1..W | 1..H | 1..T | 1..T (=t_src) | Fixed |
| Depth Map | 1..W | 1..H | 1..T | 1..T (=t_src) | 1..T (=t_src) |
| Extrinsics | 1..h | 1..w | Fixed | 1..T | 1..T |
| Intrinsics | 1..h | 1..w | 1..T | 1..T (=t_src) | 1..T (=t_src) |

## Key Implementation Notes

- Local RGB patch (9×9) dramatically improves performance - critical for fine details
- Encoder initialized from VideoMAE pretrained weights
- Aspect ratio embedded as separate token for arbitrary input ratios
- Camera pose derived via Umeyama algorithm on point correspondences
- For high-res decoding, extract patches from original resolution frames

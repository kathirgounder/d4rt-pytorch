"""Build the D4RT walkthrough notebook from scratch.

Generates notebooks/walkthrough.ipynb with PointOdyssey-focused
data exploration, architecture walkthrough, training loop, and
post-training visualizations (depth, tracks, point cloud).
"""
import json

cells = []

def md(source):
    """Create a markdown cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    })

def code(source):
    """Create a code cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })

# =============================================================================
# Section 1: Setup
# =============================================================================

md("""\
# D4RT: Training on PointOdyssey

End-to-end walkthrough: **data exploration → architecture → training → visualization**.

We train D4RT on a single PointOdyssey sequence (overfitting on purpose) to verify the architecture works
and produce depth maps, point tracks, and 3D point clouds — a toy version of the results on the D4RT project page.

Sections:
1. **PointOdyssey Dataset** — explore RGB, depth, tracks, normals, intrinsics
2. **Architecture** — encoder, query embeddings, decoder, loss
3. **Training** — real training loop with periodic depth snapshots
4. **Depth Prediction** — watch predictions evolve from noise to structure
5. **Point Tracking** — follow points across frames
6. **Point Cloud** — 3D scene reconstruction from video""")

code("""\
import sys, os, time, math, pathlib

# Find project root (directory containing 'models/' and 'data/')
_nb_dir = pathlib.Path(globals().get('__vsc_ipynb_file__', __file__)).resolve().parent if '__file__' in dir() else pathlib.Path.cwd()
_project_root = _nb_dir
for _ in range(5):
    if (_project_root / 'models').is_dir() and (_project_root / 'data').is_dir():
        break
    _project_root = _project_root.parent
os.chdir(str(_project_root))
# Ensure our project is first on sys.path and clear any cached 'data' module
# (some environments have a system-level 'data' package that shadows ours)
if str(_project_root) in sys.path:
    sys.path.remove(str(_project_root))
sys.path.insert(0, str(_project_root))
for _mod in list(sys.modules):
    if _mod == 'data' or _mod.startswith('data.'):
        del sys.modules[_mod]
print(f'Project root: {_project_root}')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

print(f'PyTorch {torch.__version__}')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple MPS')
else:
    device = torch.device('cpu')
    print('Using CPU (training will be slow — reduce train_steps)')
print(f'Device: {device}')""")

code("""\
# ============================================================
# Configuration — adjust these for your hardware
# ============================================================
CFG = {
    # Data
    'img_size': 64,         # 64 for fast iteration, 128+ for better quality
    'num_frames': 16,       # temporal frames to sample from the sequence
    'num_queries': 512,     # queries per training step

    # Model (small for fast training)
    'embed_dim': 256,
    'encoder_depth': 4,
    'decoder_depth': 4,
    'num_heads': 4,
    'patch_size': (2, 8, 8),  # temporal, spatial_h, spatial_w

    # Training
    'train_steps': 1000,    # A100: ~2 min; CPU: ~10 min. Increase for better results.
    'lr': 1e-3,
    'min_lr': 1e-5,
    'warmup_steps': 50,
    'weight_decay': 0.01,
    'grad_clip': 10.0,

    # Visualization
    'snapshot_steps': [0, 50, 100, 200, 500, 1000],  # capture depth at these steps
    'depth_viz_res': (16, 16),  # resolution for depth prediction during training
}

# On A100, you can increase for better results:
# CFG['img_size'] = 128; CFG['train_steps'] = 3000; CFG['depth_viz_res'] = (32, 32)

print('Configuration:')
for k, v in CFG.items():
    print(f'  {k}: {v}')""")

# =============================================================================
# Section 2: PointOdyssey Data Exploration
# =============================================================================

md("""\
---
## 1. PointOdyssey Dataset

**PointOdyssey** is a large-scale synthetic dataset with complex multi-object scenes. Unlike MiDaS pseudo-depth,
it provides **complete ground truth** for every supervision signal D4RT uses:

| Field | Shape | Description |
|-------|-------|-------------|
| RGB frames | `(T, H, W, 3)` | High-quality rendered video |
| Depth | `(T, H, W)` | Metric depth in meters |
| 3D tracks | `(T, N_tracks, 3)` | World-coordinate trajectories |
| 2D tracks | `(T, N_tracks, 2)` | Pixel-coordinate trajectories |
| Visibility | `(T, N_tracks)` | Per-track occlusion labels |
| Normals | `(T, H, W, 3)` | Per-pixel surface normals |
| Intrinsics | `(T, 3, 3)` | Exact camera matrix from renderer |
| Extrinsics | `(T, 4, 4)` | Exact camera pose |

This means **all 6 loss components** (L_3d, L_2d, L_vis, L_disp, L_normal, L_conf) will be active during training.""")

code("""\
# Load PointOdyssey sample via our KubricDataset loader
from data import KubricDataset, collate_fn

po_ds = KubricDataset(
    data_root='data/pointodyssey/sample',
    split='.',
    num_frames=CFG['num_frames'],
    img_size=CFG['img_size'],
    num_queries=CFG['num_queries'],
)
print(f'Sequences found: {len(po_ds)}')
for s in po_ds.sequences:
    contents = sorted([p.name for p in s.iterdir() if not p.name.startswith('.')])
    print(f'  {s.name}/: {contents}')

# Load one sample (this also runs query sampling)
sample = po_ds[0]
print(f'\\n--- Processed sample (after temporal subsampling + resize to {CFG["img_size"]}x{CFG["img_size"]}) ---')
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape} {v.dtype}')
    elif isinstance(v, dict):
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv.shape} {vv.dtype}')

# Show raw annotation shapes
seq_dir = po_ds.sequences[0]
anno = np.load(seq_dir / 'anno.npz', allow_pickle=True)
print(f'\\n--- Raw anno.npz (before any processing) ---')
for k in anno.files:
    print(f'  {k}: {anno[k].shape} {anno[k].dtype}')""")

code("""\
# RGB frames + ground truth metric depth
video = sample['video']  # (T, H, W, 3)
depth = sample['depth']  # (T, H, W)
T = video.shape[0]

fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))
for t in range(T):
    axes[0, t].imshow(video[t].clamp(0, 1).numpy())
    axes[0, t].set_title(f't={t}', fontsize=9)
    axes[0, t].axis('off')
    axes[1, t].imshow(depth[t].numpy(), cmap='plasma')
    axes[1, t].axis('off')
axes[0, 0].set_ylabel('RGB', fontsize=12)
axes[1, 0].set_ylabel('GT Depth\\n(meters)', fontsize=12)
fig.suptitle('PointOdyssey: RGB + Ground Truth Metric Depth', fontsize=14)
plt.tight_layout()
plt.show()

d = depth[depth > 0]
print(f'Depth stats: min={d.min():.2f}m, max={d.max():.2f}m, mean={d.mean():.2f}m')""")

code("""\
# 2D point tracks — temporal correspondence across frames
trajs_2d_raw = anno['trajs_2d']   # (T_total, N_tracks, 2) in pixels
visibs_raw = anno['visibs']        # (T_total, N_tracks)
trajs_3d_raw = anno['trajs_3d']   # (T_total, N_tracks, 3) in world coords

T_total, N_tracks = trajs_2d_raw.shape[:2]
print(f'Tracks: {N_tracks:,} points tracked across {T_total} frames')
print(f'Visibility rate: {visibs_raw.mean():.1%}')

# Pick 15 visible tracks at frame 0
visible_at_0 = np.where(visibs_raw[0] > 0.5)[0]
np.random.seed(42)
selected = np.random.choice(visible_at_0, size=min(15, len(visible_at_0)), replace=False)

# Visualize: 2D tracks on frame 0 (left) and 3D tracks (right)
frame0 = np.array(Image.open(sorted((seq_dir / 'rgbs').glob('*.jpg'))[0]))
colors = plt.cm.rainbow(np.linspace(0, 1, len(selected)))

fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121)
ax1.imshow(frame0)
ax1.set_title(f'2D tracks on frame 0 (first 100 frames)')
ax1.axis('off')

for i, tidx in enumerate(selected):
    n_show = min(100, T_total)
    track = trajs_2d_raw[:n_show, tidx]
    vis = visibs_raw[:n_show, tidx]
    for j in range(n_show - 1):
        if vis[j] > 0.5 and vis[j+1] > 0.5:
            alpha = 0.3 + 0.7 * (j / n_show)
            ax1.plot(track[j:j+2, 0], track[j:j+2, 1], color=colors[i], alpha=alpha, linewidth=1.5)
    ax1.scatter(track[0, 0], track[0, 1], c=[colors[i]], s=30, edgecolors='white', linewidths=0.5, zorder=5)

ax3d = fig.add_subplot(122, projection='3d')
for i, tidx in enumerate(selected):
    track_3d = trajs_3d_raw[:min(100, T_total), tidx]
    vis = visibs_raw[:min(100, T_total), tidx]
    pts = track_3d[vis > 0.5]
    if len(pts) > 1:
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], alpha=0.6, linewidth=1)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title('3D tracks (world coordinates)')
fig.suptitle('PointOdyssey: Point Tracks — Temporal Correspondence', fontsize=14)
plt.tight_layout()
plt.show()""")

code("""\
# Surface normals + camera intrinsics
normal_dir = seq_dir / 'normals'
normal_files = sorted(list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png')))
frame_indices = np.linspace(0, len(normal_files)-1, 4, dtype=int)
rgb_files = sorted((seq_dir / 'rgbs').glob('*.jpg'))

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, fidx in enumerate(frame_indices):
    axes[0, i].imshow(np.array(Image.open(rgb_files[fidx])))
    axes[0, i].set_title(f'Frame {fidx}')
    axes[0, i].axis('off')
    axes[1, i].imshow(np.array(Image.open(normal_files[fidx])))
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('RGB', fontsize=12)
axes[1, 0].set_ylabel('Normals', fontsize=12)
fig.suptitle('PointOdyssey: Per-pixel Surface Normals (RGB-encoded)', fontsize=14)
plt.tight_layout()
plt.show()

# Intrinsics
K = anno['intrinsics'][0]
print(f'Camera intrinsics K:')
print(f'  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}')
print(f'\\nAll 6 GT fields available → all 6 loss components will be active during training')""")

code("""\
# Query type breakdown — depth + tracking + point cloud
t_src = sample['t_src']
t_tgt = sample['t_tgt']
t_cam = sample['t_cam']
tgts = sample['targets']

is_depth = (t_src == t_tgt) & (t_tgt == t_cam)
is_tracking = (t_src != t_tgt) & (t_tgt == t_cam)
is_pc = (t_src == t_tgt) & (t_tgt != t_cam)
n_d, n_t, n_p = is_depth.sum().item(), is_tracking.sum().item(), is_pc.sum().item()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
axes[0].pie([n_d, n_t, n_p],
            labels=[f'Depth ({n_d})', f'Tracking ({n_t})', f'PointCloud ({n_p})'],
            colors=['#4C72B0', '#C44E52', '#CCB974'], autopct='%1.0f%%', startangle=90)
axes[0].set_title('Query type distribution')

# Mask coverage
mask_names = ['mask_3d', 'mask_disp', 'mask_normal']
vals = [tgts[m].sum().item() / len(tgts[m]) * 100 for m in mask_names]
bars = axes[1].bar(['3D position', 'Displacement', 'Normals'], vals, color='steelblue', alpha=0.8)
axes[1].set_ylabel('% queries with valid GT')
axes[1].set_title('Supervision signal coverage')
axes[1].set_ylim(0, 110)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{v:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
print(f'mask_3d: {tgts["mask_3d"].sum().item():.0f}/{len(tgts["mask_3d"])}  '
      f'mask_disp: {tgts["mask_disp"].sum().item():.0f}/{len(tgts["mask_disp"])}  '
      f'mask_normal: {tgts["mask_normal"].sum().item():.0f}/{len(tgts["mask_normal"])}')""")

# =============================================================================
# Section 3: Architecture Walkthrough
# =============================================================================

md("""\
---
## 2. Architecture Walkthrough

D4RT is an encoder-decoder transformer:
1. **Encoder**: ViT with interleaved local (frame-wise) and global self-attention → Global Scene Representation `F`
2. **Decoder**: Cross-attention transformer — each query `(u, v, t_src, t_tgt, t_cam)` attends to `F` independently
3. **6 output heads**: pos_3d, pos_2d, visibility, displacement, normal, confidence

The key insight: all tasks (depth, tracking, point cloud, camera pose) are expressed as different query configurations.""")

code("""\
# Build model
from models import D4RT
from models.encoder import D4RTEncoder
from models.decoder import D4RTDecoder

model = D4RT(
    encoder_variant='base',
    img_size=CFG['img_size'],
    temporal_size=CFG['num_frames'],
    patch_size=CFG['patch_size'],
    decoder_depth=CFG['decoder_depth'],
    max_timesteps=CFG['num_frames'],
    query_patch_size=5,
)
# Override with small encoder/decoder for fast training
model.encoder = D4RTEncoder(
    img_size=CFG['img_size'], temporal_size=CFG['num_frames'],
    patch_size=CFG['patch_size'],
    embed_dim=CFG['embed_dim'], depth=CFG['encoder_depth'], num_heads=CFG['num_heads'],
)
model.decoder = D4RTDecoder(
    embed_dim=CFG['embed_dim'], depth=CFG['decoder_depth'], num_heads=CFG['num_heads'],
    max_timesteps=CFG['num_frames'], patch_size=5,
)
model = model.to(device)

enc_params = sum(p.numel() for p in model.encoder.parameters())
dec_params = sum(p.numel() for p in model.decoder.parameters())
print(f'Encoder: {enc_params:,} params')
print(f'Decoder: {dec_params:,} params')
print(f'Total:   {enc_params + dec_params:,} params')
print(f'\\nEncoder blocks: {len(model.encoder.blocks)}')
for i, b in enumerate(model.encoder.blocks):
    print(f'  [{i}] {b.attention_type} attention + MLP')""")

code("""\
# Encoder: video → Global Scene Representation
batch = collate_fn([sample])
video_b = batch['video'].to(device)  # (1, T, H, W, 3)

with torch.no_grad():
    features = model.encoder(video_b, batch['aspect_ratio'].to(device))
print(f'Input:  video {video_b.shape}')
print(f'Output: features {features.shape} (Global Scene Representation F)')

# PCA visualization of encoder features
feat = features[0].cpu().numpy()
feat_c = feat - feat.mean(axis=0)
_, _, Vt = np.linalg.svd(feat_c, full_matrices=False)
pca = feat_c @ Vt[:3].T
pca = (pca - pca.min(0)) / (pca.max(0) - pca.min(0) + 1e-8)

n_t = model.encoder.num_frames
n_h = model.encoder.patch_embed.num_patches_h
n_w = model.encoder.patch_embed.num_patches_w

fig, axes = plt.subplots(2, n_t, figsize=(3.5 * n_t, 6))
for t_idx in range(n_t):
    frame_t = min(t_idx * (T // n_t), T - 1)
    axes[0, t_idx].imshow(video[frame_t].clamp(0, 1).numpy())
    axes[0, t_idx].set_title(f'Frame {frame_t}')
    axes[0, t_idx].axis('off')
    start = t_idx * n_h * n_w
    axes[1, t_idx].imshow(pca[start:start + n_h*n_w].reshape(n_h, n_w, 3))
    axes[1, t_idx].axis('off')
axes[0, 0].set_ylabel('Input', fontsize=12)
axes[1, 0].set_ylabel('Encoder\\nFeatures', fontsize=12)
fig.suptitle('Encoder: Video → Feature Tokens (PCA → RGB)', fontsize=14)
plt.tight_layout()
plt.show()""")

code("""\
# Query embeddings: Fourier + timestep + RGB patch
from models.embeddings import FourierEmbedding, TimestepEmbedding, PatchEmbeddingFast

fourier = FourierEmbedding(CFG['embed_dim'], num_frequencies=32).to(device)
timestep = TimestepEmbedding(max_timesteps=CFG['num_frames'], embed_dim=CFG['embed_dim']).to(device)
patch_emb = PatchEmbeddingFast(patch_size=5, embed_dim=CFG['embed_dim']).to(device)

coords_b = batch['coords'][:1].to(device)
t_src_b = batch['t_src'][:1].to(device)
t_tgt_b = batch['t_tgt'][:1].to(device)
t_cam_b = batch['t_cam'][:1].to(device)
frames_b = video_b[:1].permute(0, 1, 4, 2, 3)

with torch.no_grad():
    coord_emb = fourier(coords_b)
    src_emb, tgt_emb, cam_emb = timestep(t_src_b, t_tgt_b, t_cam_b)
    patch_feat = patch_emb(frames_b, coords_b, t_src_b)

components = {
    'Fourier(u,v)': coord_emb[0].norm(dim=-1).mean().item(),
    'Timestep(src)': src_emb[0].norm(dim=-1).mean().item(),
    'Timestep(tgt)': tgt_emb[0].norm(dim=-1).mean().item(),
    'Timestep(cam)': cam_emb[0].norm(dim=-1).mean().item(),
    'RGB patch': patch_feat[0].norm(dim=-1).mean().item(),
}
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(list(components.keys()), list(components.values()),
        color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])
ax.set_xlabel('Mean L2 norm')
ax.set_title('Query embedding components (before training)')
plt.tight_layout()
plt.show()
print('Query = Fourier(u,v) + Timestep(src) + Timestep(tgt) + Timestep(cam) + RGBpatch + learnable_token')""")

code("""\
# Decoder outputs (untrained — random predictions)
with torch.no_grad():
    outputs = model(
        video_b.permute(0, 4, 1, 2, 3),
        coords_b, t_src_b, t_tgt_b, t_cam_b,
        batch['aspect_ratio'].to(device)
    )

print('Decoder outputs (untrained):')
for k, v in outputs.items():
    print(f'  {k:15s}: {str(v.shape):25s} range=[{v.min():.3f}, {v.max():.3f}]')""")

code("""\
# Loss computation — all 6 components active
from losses import D4RTLoss

criterion = D4RTLoss(lambda_3d=1.0, lambda_2d=0.1, lambda_vis=0.1,
                     lambda_disp=0.1, lambda_normal=0.5, lambda_conf=0.2)
targets_b = {k: v.to(device) for k, v in batch['targets'].items()}
losses = criterion(outputs, targets_b)

print('Loss breakdown (untrained model):')
weights = {'loss_3d': 1.0, 'loss_2d': 0.1, 'loss_vis': 0.1,
           'loss_disp': 0.1, 'loss_normal': 0.5, 'loss_conf': 0.2}
for k in ['loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf', 'loss']:
    v = losses[k].item()
    w = weights.get(k, '')
    active = 'ACTIVE' if v > 0.001 and k != 'loss' else ''
    print(f'  {k:13s}: {v:.4f}  {"(λ="+str(w)+")" if w else ""} {active}')""")

# =============================================================================
# Section 4: Training Loop
# =============================================================================

md("""\
---
## 3. Training

We train on a single PointOdyssey sequence, deliberately **overfitting** to verify the model can learn all tasks.
The training loop captures depth prediction snapshots at key steps so we can visualize how predictions evolve.

On an **A100**: ~10 steps/sec → 1000 steps in ~2 minutes.
On **CPU**: ~1-2 steps/sec → reduce `train_steps` to 100-200.""")

code("""\
# Training setup
from torch.nn.utils import clip_grad_norm_

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])

# Cosine schedule with warmup
def lr_lambda(step):
    if step < CFG['warmup_steps']:
        return step / max(CFG['warmup_steps'], 1)
    progress = (step - CFG['warmup_steps']) / max(CFG['train_steps'] - CFG['warmup_steps'], 1)
    return max(CFG['min_lr'] / CFG['lr'], 0.5 * (1 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Prepare batch on device
video_input = video_b.permute(0, 4, 1, 2, 3).to(device)  # (1, C, T, H, W)
coords_d = batch['coords'].to(device)
t_src_d = batch['t_src'].to(device)
t_tgt_d = batch['t_tgt'].to(device)
t_cam_d = batch['t_cam'].to(device)
ar_d = batch['aspect_ratio'].to(device)

# For depth prediction snapshots
video_eval = video_b.to(device)  # (1, T, H, W, 3)

# Storage for logging
loss_log = {k: [] for k in ['loss', 'loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf']}
depth_snapshots = {}
snapshot_steps = sorted(set(s for s in CFG['snapshot_steps'] if s <= CFG['train_steps']))
if CFG['train_steps'] not in snapshot_steps:
    snapshot_steps.append(CFG['train_steps'])

print(f'Training for {CFG["train_steps"]} steps')
print(f'Depth snapshots at steps: {snapshot_steps}')
print(f'Optimizer: AdamW (lr={CFG["lr"]}, wd={CFG["weight_decay"]})')""")

code("""\
# Training loop
model.train()
t0 = time.time()

for step in range(CFG['train_steps'] + 1):
    # Capture depth snapshot BEFORE training at step 0, and at other snapshot steps
    if step in snapshot_steps:
        model.eval()
        with torch.no_grad():
            d_snap = model.predict_depth(video_eval, output_resolution=CFG['depth_viz_res'])
        depth_snapshots[step] = d_snap[0].cpu()
        model.train()

    if step == CFG['train_steps']:
        break  # last snapshot captured, done

    # Forward
    preds = model(video_input, coords_d, t_src_d, t_tgt_d, t_cam_d, ar_d)
    all_losses = criterion(preds, targets_b)
    loss = all_losses['loss']

    # Backward
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), CFG['grad_clip'])
    optimizer.step()
    scheduler.step()

    # Log
    for k in loss_log:
        loss_log[k].append(all_losses[k].item())

    if step % max(CFG['train_steps'] // 10, 1) == 0 or step == CFG['train_steps'] - 1:
        elapsed = time.time() - t0
        steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
        print(f'Step {step:5d}/{CFG["train_steps"]}  loss={loss.item():.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.6f}  [{steps_per_sec:.1f} steps/s]')

elapsed = time.time() - t0
print(f'\\nTraining complete: {CFG["train_steps"]} steps in {elapsed:.1f}s ({CFG["train_steps"]/elapsed:.1f} steps/s)')""")

code("""\
# Loss curves — all 6 components
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
loss_keys = ['loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf']
loss_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
loss_weights = [1.0, 0.1, 0.1, 0.1, 0.5, 0.2]

for i, (k, c, w) in enumerate(zip(loss_keys, loss_colors, loss_weights)):
    ax = axes[i // 3, i % 3]
    vals = loss_log[k]
    ax.plot(vals, color=c, alpha=0.3, linewidth=0.5)
    # Smoothed
    window = max(len(vals) // 50, 1)
    if len(vals) > window:
        smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(vals)), smoothed, color=c, linewidth=2)
    ax.set_title(f'{k} (λ={w})')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

fig.suptitle('Training Loss Components', fontsize=14)
plt.tight_layout()
plt.show()

# Total loss
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(loss_log['loss'], 'k-', alpha=0.3, linewidth=0.5)
window = max(len(loss_log['loss']) // 50, 1)
if len(loss_log['loss']) > window:
    smoothed = np.convolve(loss_log['loss'], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(loss_log['loss'])), smoothed, 'k-', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Total Loss')
ax.set_title(f'Total loss: {loss_log["loss"][0]:.4f} → {loss_log["loss"][-1]:.4f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")

# =============================================================================
# Section 5: Depth Prediction Visualization
# =============================================================================

md("""\
---
## 4. Depth Prediction — Watching the Model Learn

The snapshots captured during training show depth predictions evolving from random noise to structured output.
For depth queries: `t_src = t_tgt = t_cam = t`, and we extract the Z component of `pos_3d`.""")

code("""\
# Depth evolution grid: rows = training steps, columns = video frames
gt_depth = depth.numpy()  # (T, H, W)

n_snapshots = len(depth_snapshots)
n_frames = min(T, 8)
frame_idx = np.linspace(0, T-1, n_frames, dtype=int)

fig, axes = plt.subplots(n_snapshots + 1, n_frames, figsize=(2.5 * n_frames, 2.2 * (n_snapshots + 1)))

# GT depth row
for j, fi in enumerate(frame_idx):
    axes[0, j].imshow(gt_depth[fi], cmap='plasma')
    axes[0, j].set_title(f'Frame {fi}', fontsize=9)
    axes[0, j].axis('off')
axes[0, 0].set_ylabel('GT Depth', fontsize=10, fontweight='bold')

# Prediction rows at each snapshot step
sorted_steps = sorted(depth_snapshots.keys())
for row, step in enumerate(sorted_steps):
    snap = depth_snapshots[step]  # (T, res_h, res_w)
    for j, fi in enumerate(frame_idx):
        axes[row + 1, j].imshow(snap[fi].numpy(), cmap='plasma')
        axes[row + 1, j].axis('off')
    axes[row + 1, 0].set_ylabel(f'Step {step}', fontsize=10)

fig.suptitle('Depth Prediction Evolution During Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print('Watch predictions evolve from random noise (step 0) to structured depth!')""")

code("""\
# Final depth: RGB | GT | Predicted — side by side
final_step = sorted_steps[-1]
final_depth = depth_snapshots[final_step]  # (T, res_h, res_w)

fig, axes = plt.subplots(3, n_frames, figsize=(2.8 * n_frames, 7))
for j, fi in enumerate(frame_idx):
    axes[0, j].imshow(video[fi].clamp(0, 1).numpy())
    axes[0, j].set_title(f'Frame {fi}', fontsize=9)
    axes[0, j].axis('off')
    axes[1, j].imshow(gt_depth[fi], cmap='plasma')
    axes[1, j].axis('off')
    axes[2, j].imshow(final_depth[fi].numpy(), cmap='plasma')
    axes[2, j].axis('off')
axes[0, 0].set_ylabel('RGB', fontsize=11)
axes[1, 0].set_ylabel('GT Depth', fontsize=11)
axes[2, 0].set_ylabel(f'Predicted\\n(step {final_step})', fontsize=11)
fig.suptitle('Final Depth Prediction vs Ground Truth', fontsize=14)
plt.tight_layout()
plt.show()

# Compute metrics
from losses.losses import DepthLoss
depth_metric = DepthLoss(scale_invariant=True)
pred_d = final_depth.unsqueeze(0)
gt_d_resized = F.interpolate(
    depth.unsqueeze(1), size=CFG['depth_viz_res'], mode='nearest'
).squeeze(1).unsqueeze(0)  # (1, T, res_h, res_w)

# Average metrics across frames
abs_rels, rmses = [], []
for t_i in range(T):
    mask = gt_d_resized[0, t_i] > 0
    if mask.sum() > 10:
        m = depth_metric(pred_d[0, t_i], gt_d_resized[0, t_i], mask)
        abs_rels.append(m['abs_rel'].item())
        rmses.append(m['rmse'].item())
if abs_rels:
    print(f'Depth metrics (scale-invariant): AbsRel={np.mean(abs_rels):.4f}, RMSE={np.mean(rmses):.4f}')""")

# =============================================================================
# Section 6: Point Track Prediction
# =============================================================================

md("""\
---
## 5. Point Tracking — Following Points Across Frames

For tracking queries: fixed `(u, v, t_src)`, varying `t_tgt = t_cam = 0..T-1`.
The model predicts where each point moves to in every frame.""")

code("""\
# Pick query points from GT tracks (visible at frame 0)
from utils.visualization import visualize_tracks, visualize_3d_tracks

model.eval()
n_track_queries = 20

# Use GT track positions as query points
vis_at_0 = visibs_raw[0] > 0.5
valid_track_idx = np.where(vis_at_0)[0]
np.random.seed(123)
pick = np.random.choice(valid_track_idx, size=min(n_track_queries, len(valid_track_idx)), replace=False)

# Get query coordinates (normalized) and source frames
# We need pixel coords at frame 0 → normalize to [0,1]
raw_H, raw_W = 540, 960  # PointOdyssey raw resolution
query_pts_px = trajs_2d_raw[0, pick]  # (n_track_queries, 2) in pixels
query_pts_norm = query_pts_px.copy()
query_pts_norm[:, 0] /= raw_W
query_pts_norm[:, 1] /= raw_H

query_points = torch.from_numpy(query_pts_norm).float().unsqueeze(0).to(device)  # (1, N, 2)
query_frames = torch.zeros(1, len(pick), dtype=torch.long, device=device)  # all from frame 0

# Predict tracks
with torch.no_grad():
    track_preds = model.predict_point_tracks(video_eval, query_points, query_frames)

pred_tracks_2d = track_preds['tracks_2d'][0].cpu()  # (N, T, 2) normalized
pred_tracks_3d = track_preds['tracks_3d'][0].cpu()  # (N, T, 3)
pred_vis = track_preds['visibility'][0].cpu()         # (N, T)

print(f'Predicted {pred_tracks_2d.shape[0]} tracks across {pred_tracks_2d.shape[1]} frames')

# Visualize predicted 2D tracks
fig = visualize_tracks(video.cpu(), pred_tracks_2d, pred_vis, num_tracks=n_track_queries)
fig.suptitle('Predicted 2D Point Tracks (overlaid on video frames)', fontsize=14)
plt.tight_layout()
plt.show()""")

code("""\
# 3D track comparison: GT vs Predicted
fig = plt.figure(figsize=(14, 6))

# GT tracks (subsampled to our num_frames)
ax1 = fig.add_subplot(121, projection='3d')
# Subsample GT to match our temporal sampling
frame_step = max(T_total // T, 1)
gt_colors = plt.cm.rainbow(np.linspace(0, 1, len(pick)))
for i, tidx in enumerate(pick):
    gt_track = trajs_3d_raw[::frame_step, tidx][:T]
    gt_vis = visibs_raw[::frame_step, tidx][:T]
    pts = gt_track[gt_vis > 0.5]
    if len(pts) > 1:
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=gt_colors[i], alpha=0.7, linewidth=1.5)
        ax1.scatter(pts[0, 0], pts[0, 1], pts[0, 2], c=[gt_colors[i]], s=20, marker='o')
ax1.set_title('GT 3D Tracks')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# Predicted tracks
ax2 = fig.add_subplot(122, projection='3d')
for i in range(len(pick)):
    pts = pred_tracks_3d[i].numpy()
    vis = pred_vis[i].numpy()
    visible = vis > 0.5
    if visible.sum() > 1:
        p = pts[visible]
        ax2.plot(p[:, 0], p[:, 1], p[:, 2], color=gt_colors[i], alpha=0.7, linewidth=1.5)
        ax2.scatter(p[0, 0], p[0, 1], p[0, 2], c=[gt_colors[i]], s=20, marker='o')
ax2.set_title('Predicted 3D Tracks')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

fig.suptitle('3D Point Tracks: Ground Truth vs Predicted', fontsize=14)
plt.tight_layout()
plt.show()
print('With more training steps, predicted tracks will converge toward GT trajectories.')""")

# =============================================================================
# Section 7: Point Cloud Reconstruction
# =============================================================================

md("""\
---
## 6. Point Cloud Reconstruction — 3D Scene from Video

For point cloud queries: varying `(u, v, t_src)`, fixed `t_cam = reference_frame`.
This gives a dense 3D reconstruction of the scene in a single coordinate frame.""")

code("""\
# Predict point cloud
from utils.visualization import visualize_point_cloud

with torch.no_grad():
    pc = model.predict_point_cloud(video_eval, reference_frame=0, stride=2)

points = pc['points'][0].cpu()   # (T*H*W, 3)
colors = pc['colors'][0].cpu()   # (T*H*W, 3)
normals_pc = pc['normals'][0].cpu()

print(f'Point cloud: {points.shape[0]:,} points from {T} frames')

# Filter outliers for cleaner visualization
valid = (points.abs() < 20).all(dim=-1)
pts_clean = points[valid].numpy()
cols_clean = colors[valid].numpy()

# Multiple viewpoints
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
angles = [(30, 45), (60, 135), (15, -45)]
for ax, (elev, azim) in zip(axes, angles):
    ax.scatter(pts_clean[::3, 0], pts_clean[::3, 1], pts_clean[::3, 2],
              c=cols_clean[::3].clip(0, 1), s=0.5, alpha=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    r = max(np.abs(pts_clean).max(), 1)
    ax.set_xlim([-r, r]); ax.set_ylim([-r, r]); ax.set_zlim([-r, r])
fig.suptitle('3D Point Cloud Reconstruction (3 viewpoints)', fontsize=14)
plt.tight_layout()
plt.show()

# Save PLY for external viewer
from utils.visualization import save_point_cloud_ply
os.makedirs('outputs', exist_ok=True)
save_point_cloud_ply('outputs/reconstruction.ply', points[valid], colors[valid], normals_pc[valid])
print('Saved point cloud to outputs/reconstruction.ply')
print('Open in MeshLab, CloudCompare, or any PLY viewer for interactive 3D exploration.')""")

# =============================================================================
# Section 8: Training Progression Animation
# =============================================================================

md("""\
---
## 7. Training Progression — Depth Animation

A filmstrip showing how depth predictions evolve during training, plus an animated GIF.""")

code("""\
# Depth evolution filmstrip for frame 0
frame_idx_anim = 0
sorted_steps_anim = sorted(depth_snapshots.keys())

fig, axes = plt.subplots(1, len(sorted_steps_anim) + 1, figsize=(2.5 * (len(sorted_steps_anim) + 1), 2.5))

# GT
axes[0].imshow(gt_depth[frame_idx_anim], cmap='plasma')
axes[0].set_title('GT', fontsize=10, fontweight='bold')
axes[0].axis('off')

# Snapshots
for i, step in enumerate(sorted_steps_anim):
    axes[i + 1].imshow(depth_snapshots[step][frame_idx_anim].numpy(), cmap='plasma')
    axes[i + 1].set_title(f'Step {step}', fontsize=10)
    axes[i + 1].axis('off')

fig.suptitle(f'Depth Prediction Evolution (Frame {frame_idx_anim})', fontsize=14)
plt.tight_layout()
plt.show()

# Create animated GIF (depth across all frames at final checkpoint)
try:
    from matplotlib.animation import FuncAnimation, PillowWriter

    final_depth_anim = depth_snapshots[sorted_steps_anim[-1]]  # (T, h, w)

    fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    def animate(t):
        ax1.clear(); ax2.clear()
        ax1.imshow(video[t].clamp(0, 1).numpy())
        ax1.set_title(f'RGB (frame {t})')
        ax1.axis('off')
        ax2.imshow(final_depth_anim[t].numpy(), cmap='plasma')
        ax2.set_title(f'Predicted Depth')
        ax2.axis('off')
        return []

    anim = FuncAnimation(fig_anim, animate, frames=T, interval=300)
    anim.save('outputs/depth_animation.gif', writer=PillowWriter(fps=3))
    plt.close(fig_anim)
    print('Saved depth animation to outputs/depth_animation.gif')

    # Display in notebook (if IPython available)
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename='outputs/depth_animation.gif'))
    except ImportError:
        print('(Run in Jupyter to see inline animation)')
except Exception as e:
    print(f'GIF creation skipped: {e}')
    print('Install Pillow for GIF support: pip install Pillow')""")

# =============================================================================
# Section 9: Architecture Summary
# =============================================================================

code("""\
print('=' * 65)
print('D4RT Architecture Summary')
print('=' * 65)
print()
print('ENCODER (ViT with local/global attention)')
print(f'  Input:  Video (B, C, T, H, W)')
print(f'  Patch:  {model.encoder.patch_size} → 3D Conv')
print(f'  Blocks: {len(model.encoder.blocks)} with interleaved attention')
for i, b in enumerate(model.encoder.blocks):
    print(f'    [{i}] {b.attention_type:6s} attention + MLP')
print(f'  Output: Global Scene Representation F ∈ R^(N×{model.encoder.embed_dim})')
print()
print('DECODER (cross-attention transformer)')
print(f'  Query:  (u, v, t_src, t_tgt, t_cam) → {model.decoder.embed_dim}D embedding')
print(f'    Fourier(u,v) + Timestep(src) + Timestep(tgt) + Timestep(cam) + Patch(9×9)')
print(f'  Blocks: {len(model.decoder.blocks)} cross-attention layers')
print(f'  Heads:  pos_3d(3), pos_2d(2), vis(1), disp(3), normal(3), conf(1)')
print()
print('LOSS')
print(f'  L = 1.0·L_3d + 0.1·L_2d + 0.1·L_vis + 0.1·L_disp + 0.5·L_normal + 0.2·L_conf')
print()
print('TASKS (all from same query interface)')
print('  Depth:       t_src = t_tgt = t_cam = t')
print('  Tracking:    fixed (u,v,t_src), varying t_tgt = t_cam')
print('  Point Cloud: varying (u,v,t_src), fixed t_cam')
print('  Extrinsics:  grid queries + Umeyama alignment')
print('=' * 65)""")

# =============================================================================
# Build notebook JSON
# =============================================================================

# Assign IDs
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i}'

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open('notebooks/walkthrough.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'Notebook written: {len(cells)} cells')
for i, c in enumerate(cells):
    ct = c['cell_type']
    src_preview = ''.join(c['source'][:2]).strip()[:90]
    print(f'  Cell {i:2d} [{ct:8s}]: {src_preview}')

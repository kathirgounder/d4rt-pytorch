"""Smoke test: run the critical code paths from the notebook on CPU.

This verifies imports, data loading, model construction, forward pass,
loss computation, a few training steps, and all inference methods work
without errors.
"""
import sys, os, time, math
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

device = torch.device('cpu')
print(f"Device: {device}")

# --- Config (tiny for smoke test) ---
CFG = {
    'img_size': 64,
    'num_frames': 16,
    'num_queries': 64,  # reduced for speed
    'embed_dim': 256,
    'encoder_depth': 4,
    'decoder_depth': 4,
    'num_heads': 4,
    'patch_size': (2, 8, 8),
    'train_steps': 5,  # just 5 steps
    'lr': 1e-3,
    'min_lr': 1e-5,
    'warmup_steps': 2,
    'weight_decay': 0.01,
    'grad_clip': 10.0,
    'snapshot_steps': [0, 5],
    'depth_viz_res': (8, 8),
}

# --- Cell 4: Data loading ---
print("\n=== Data Loading ===")
from data import KubricDataset, collate_fn

po_ds = KubricDataset(
    data_root='data/pointodyssey/sample',
    split='.',
    num_frames=CFG['num_frames'],
    img_size=CFG['img_size'],
    num_queries=CFG['num_queries'],
)
print(f"Sequences found: {len(po_ds)}")
sample = po_ds[0]
print(f"Sample keys: {list(sample.keys())}")
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape} {v.dtype}")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  targets.{kk}: {vv.shape} {vv.dtype}")

# Load raw anno
seq_dir = po_ds.sequences[0]
anno = np.load(seq_dir / 'anno.npz', allow_pickle=True)
print(f"Raw anno keys: {anno.files}")

# --- Cell 5: Video/depth ---
print("\n=== Video + Depth ===")
video = sample['video']
depth = sample['depth']
T = video.shape[0]
print(f"Video: {video.shape}, Depth: {depth.shape}")
d = depth[depth > 0]
print(f"Depth stats: min={d.min():.2f}m, max={d.max():.2f}m, mean={d.mean():.2f}m")

# --- Cell 8: Query breakdown ---
print("\n=== Query Breakdown ===")
t_src = sample['t_src']
t_tgt = sample['t_tgt']
t_cam = sample['t_cam']
tgts = sample['targets']
is_depth = (t_src == t_tgt) & (t_tgt == t_cam)
is_tracking = (t_src != t_tgt) & (t_tgt == t_cam)
is_pc = (t_src == t_tgt) & (t_tgt != t_cam)
print(f"Depth: {is_depth.sum()}, Tracking: {is_tracking.sum()}, PointCloud: {is_pc.sum()}")
print(f"mask_3d: {tgts['mask_3d'].sum():.0f}/{len(tgts['mask_3d'])}")
print(f"mask_disp: {tgts['mask_disp'].sum():.0f}/{len(tgts['mask_disp'])}")
print(f"mask_normal: {tgts['mask_normal'].sum():.0f}/{len(tgts['mask_normal'])}")

# --- Cell 10: Build model ---
print("\n=== Build Model ===")
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
print(f"Encoder: {enc_params:,} params")
print(f"Decoder: {dec_params:,} params")
print(f"Total:   {enc_params + dec_params:,} params")

# --- Cell 11: Encoder forward ---
print("\n=== Encoder Forward ===")
batch = collate_fn([sample])
video_b = batch['video'].to(device)

with torch.no_grad():
    features = model.encoder(video_b, batch['aspect_ratio'].to(device))
print(f"Input: {video_b.shape} -> Features: {features.shape}")

# --- Cell 12: Query embeddings ---
print("\n=== Query Embeddings ===")
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

print(f"Fourier: {coord_emb.shape}, norm={coord_emb[0].norm(dim=-1).mean():.3f}")
print(f"Timestep src: {src_emb.shape}, norm={src_emb[0].norm(dim=-1).mean():.3f}")
print(f"Patch: {patch_feat.shape}, norm={patch_feat[0].norm(dim=-1).mean():.3f}")

# --- Cell 13: Decoder forward ---
print("\n=== Decoder Forward (untrained) ===")
with torch.no_grad():
    outputs = model(
        video_b.permute(0, 4, 1, 2, 3),
        coords_b, t_src_b, t_tgt_b, t_cam_b,
        batch['aspect_ratio'].to(device)
    )
for k, v in outputs.items():
    print(f"  {k:15s}: {str(v.shape):25s} range=[{v.min():.3f}, {v.max():.3f}]")

# --- Cell 14: Loss ---
print("\n=== Loss Computation ===")
from losses import D4RTLoss
criterion = D4RTLoss(lambda_3d=1.0, lambda_2d=0.1, lambda_vis=0.1,
                     lambda_disp=0.1, lambda_normal=0.5, lambda_conf=0.2)
targets_b = {k: v.to(device) for k, v in batch['targets'].items()}
losses = criterion(outputs, targets_b)
for k in ['loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf', 'loss']:
    print(f"  {k:13s}: {losses[k].item():.4f}")

# --- Cell 16-17: Training loop with query pool (5 steps) ---
print("\n=== Loading & processing sequence (cached) ===")
from torch.nn.utils import clip_grad_norm_

# Load & process sequence ONCE (replicating __getitem__ pipeline)
_raw = po_ds._load_sequence(0)
_video = torch.from_numpy(_raw['video']).float()
_orig_h, _orig_w = _raw['original_size']

_depth = torch.from_numpy(_raw['depth']).float() if _raw.get('depth') is not None else None
_normals = torch.from_numpy(_raw['normals']).float() if _raw.get('normals') is not None else None
_intrinsics = torch.from_numpy(_raw['intrinsics']).float() if _raw.get('intrinsics') is not None else None
_extrinsics = torch.from_numpy(_raw['extrinsics']).float() if _raw.get('extrinsics') is not None else None
_tracks_3d = torch.from_numpy(_raw['tracks_3d']).float() if _raw.get('tracks_3d') is not None else None
_tracks_2d = torch.from_numpy(_raw['tracks_2d']).float() if _raw.get('tracks_2d') is not None else None
_visibility = torch.from_numpy(_raw['visibility']).float() if _raw.get('visibility') is not None else None

# Temporal subsampling
_frame_idx = po_ds.temporal_sampler(_video.shape[0], po_ds.num_frames)
_video = _video[_frame_idx]
if _depth is not None: _depth = _depth[_frame_idx]
if _normals is not None: _normals = _normals[_frame_idx]
if _extrinsics is not None: _extrinsics = _extrinsics[_frame_idx]
if _intrinsics is not None and _intrinsics.dim() == 3: _intrinsics = _intrinsics[_frame_idx]
if _tracks_3d is not None: _tracks_3d = _tracks_3d[:, _frame_idx]
if _tracks_2d is not None: _tracks_2d = _tracks_2d[:, _frame_idx]
if _visibility is not None: _visibility = _visibility[:, _frame_idx]

# Normalize + resize
if _video.max() > 1.0: _video = _video / 255.0
_video, _depth, _normals = po_ds._resize_frames(_video, _depth, _normals)
_H, _W = po_ds.img_size, po_ds.img_size
_T = _video.shape[0]

# Scale intrinsics
if _intrinsics is not None:
    _sx, _sy = _W / _orig_w, _H / _orig_h
    _intrinsics = _intrinsics.clone()
    if _intrinsics.dim() == 2:
        _intrinsics[0, 0] *= _sx; _intrinsics[0, 2] *= _sx
        _intrinsics[1, 1] *= _sy; _intrinsics[1, 2] *= _sy
    else:
        _intrinsics[:, 0, 0] *= _sx; _intrinsics[:, 0, 2] *= _sx
        _intrinsics[:, 1, 1] *= _sy; _intrinsics[:, 1, 2] *= _sy

# Scale 2D tracks
if _tracks_2d is not None:
    _tracks_2d = _tracks_2d.clone()
    _tracks_2d[..., 0] *= (_W / _orig_w)
    _tracks_2d[..., 1] *= (_H / _orig_h)

# Aspect ratio
_ar = torch.tensor([_orig_w / max(_orig_w, _orig_h), _orig_h / max(_orig_w, _orig_h)], dtype=torch.float32)

# Shared tensors
video_input = _video.unsqueeze(0).permute(0, 4, 1, 2, 3).to(device)
ar_d = _ar.unsqueeze(0).to(device)
video_for_eval = _video.unsqueeze(0).to(device)
depth_for_eval = _depth

# Generate query pool (fast — no disk I/O)
print("=== Generating query pool ===")
query_pool = []
for _i in range(3):  # small pool for smoke test
    _coords, _t_src, _t_tgt, _t_cam, _targets = po_ds.query_sampler.sample(
        _T, _H, _W, depth=_depth, tracks_3d=_tracks_3d, tracks_2d=_tracks_2d,
        visibility=_visibility, intrinsics=_intrinsics, extrinsics=_extrinsics,
        normals=_normals,
    )
    query_pool.append({
        'coords': _coords.unsqueeze(0).to(device),
        't_src': _t_src.unsqueeze(0).to(device),
        't_tgt': _t_tgt.unsqueeze(0).to(device),
        't_cam': _t_cam.unsqueeze(0).to(device),
        'targets': {k: v.unsqueeze(0).to(device) for k, v in _targets.items()},
    })
print(f"Query pool: {len(query_pool)} batches")

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
def lr_lambda(step):
    if step < CFG['warmup_steps']:
        return step / max(CFG['warmup_steps'], 1)
    progress = (step - CFG['warmup_steps']) / max(CFG['train_steps'] - CFG['warmup_steps'], 1)
    return max(CFG['min_lr'] / CFG['lr'], 0.5 * (1 + math.cos(math.pi * progress)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

loss_log = {k: [] for k in ['loss', 'loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf']}
depth_snapshots = {}
snapshot_steps = sorted(set(s for s in CFG['snapshot_steps'] if s <= CFG['train_steps']))
if CFG['train_steps'] not in snapshot_steps:
    snapshot_steps.append(CFG['train_steps'])

print("\n=== Training (5 steps) ===")
model.train()
t0 = time.time()

for step in range(CFG['train_steps'] + 1):
    if step in snapshot_steps:
        model.eval()
        with torch.no_grad():
            d_snap = model.predict_depth(video_for_eval, output_resolution=CFG['depth_viz_res'])
        depth_snapshots[step] = d_snap[0].cpu()
        model.train()

    if step == CFG['train_steps']:
        break

    qb = query_pool[step % len(query_pool)]
    preds = model(video_input, qb['coords'], qb['t_src'], qb['t_tgt'], qb['t_cam'], ar_d)
    all_losses = criterion(preds, qb['targets'])
    loss = all_losses['loss']

    if torch.isnan(loss) or torch.isinf(loss):
        scheduler.step()
        continue

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), CFG['grad_clip'])
    optimizer.step()
    scheduler.step()

    for k in loss_log:
        loss_log[k].append(all_losses[k].item())

    print(f"  Step {step}: loss={loss.item():.4f}")

elapsed = time.time() - t0
print(f"Training: {CFG['train_steps']} steps in {elapsed:.1f}s")
print(f"Depth snapshots captured: {sorted(depth_snapshots.keys())}")

# --- Cell 23: Point tracking ---
print("\n=== Point Tracking ===")
model.eval()
trajs_2d_raw = anno['trajs_2d']
visibs_raw = anno['visibs']
vis_at_0 = visibs_raw[0] > 0.5
valid_track_idx = np.where(vis_at_0)[0]
np.random.seed(123)
pick = np.random.choice(valid_track_idx, size=min(5, len(valid_track_idx)), replace=False)

raw_H, raw_W = 540, 960
query_pts_px = trajs_2d_raw[0, pick]
query_pts_norm = query_pts_px.copy()
query_pts_norm[:, 0] /= raw_W
query_pts_norm[:, 1] /= raw_H

query_points = torch.from_numpy(query_pts_norm).float().unsqueeze(0).to(device)
query_frames = torch.zeros(1, len(pick), dtype=torch.long, device=device)

with torch.no_grad():
    track_preds = model.predict_point_tracks(video_for_eval, query_points, query_frames)

pred_tracks_2d = track_preds['tracks_2d'][0].cpu()
pred_tracks_3d = track_preds['tracks_3d'][0].cpu()
pred_vis = track_preds['visibility'][0].cpu()
print(f"Tracks: {pred_tracks_2d.shape} 2D, {pred_tracks_3d.shape} 3D, {pred_vis.shape} vis")

# --- Cell 26: Point cloud ---
print("\n=== Point Cloud ===")
with torch.no_grad():
    pc = model.predict_point_cloud(video_for_eval, reference_frame=0, stride=4)

points = pc['points'][0].cpu()
colors = pc['colors'][0].cpu()
normals_pc = pc['normals'][0].cpu()
print(f"Point cloud: {points.shape[0]:,} points")

# --- Depth metrics ---
print("\n=== Depth Metrics ===")
from losses.losses import DepthLoss
depth_metric = DepthLoss(scale_invariant=True)
final_depth = depth_snapshots[sorted(depth_snapshots.keys())[-1]]
pred_d = final_depth.unsqueeze(0)
gt_d_resized = F.interpolate(
    depth_for_eval.unsqueeze(1), size=CFG['depth_viz_res'], mode='nearest'
).squeeze(1).unsqueeze(0)

for t_i in range(min(2, T)):
    mask = gt_d_resized[0, t_i] > 0
    if mask.sum() > 10:
        m = depth_metric(pred_d[0, t_i], gt_d_resized[0, t_i], mask)
        print(f"  Frame {t_i}: AbsRel={m['abs_rel'].item():.4f}, RMSE={m['rmse'].item():.4f}")

# --- Visualization imports ---
print("\n=== Visualization Imports ===")
from utils.visualization import visualize_depth, visualize_tracks, visualize_3d_tracks
from utils.visualization import visualize_point_cloud, save_point_cloud_ply
print("All visualization utilities imported successfully")

print("\n" + "=" * 50)
print("SMOKE TEST PASSED — all code paths work!")
print("=" * 50)

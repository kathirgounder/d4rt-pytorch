"""Add PointOdyssey exploration cells to walkthrough notebook."""
import json
import copy

with open('notebooks/walkthrough.ipynb') as f:
    nb = json.load(f)

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n') if isinstance(source, str) else source
    }

# Fix source: split into lines with \n preserved on each line except last
def src(text):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result

# ================================================================
# Define new cells for Section 11: PointOdyssey Exploration
# ================================================================

new_cells = []

# Cell A: Section header
new_cells.append(make_md_cell(src(
"""---
## 11. Exploring PointOdyssey: A Real Training Dataset

The synthetic dataset above has known geometry but simple scenes. Real training uses **PointOdyssey** — a large-scale synthetic dataset with complex multi-object scenes, precise ground truth depth, 3D/2D point tracks, visibility, and surface normals.

Compared to our city aerial demo (MiDaS pseudo-depth only), PointOdyssey provides **every supervision signal**: metric depth, 3D tracks for correspondence, visibility labels, and per-pixel normals. This means the model gets gradients from *all 6 loss components* — not just L_3d and L_2d.

Let's load one sequence and explore every field.""")))

# Cell B: Load sequence and inspect
new_cells.append(make_code_cell(src(
"""# Load PointOdyssey sample via our updated KubricDataset
from data import KubricDataset
import os
from pathlib import Path

po_data_root = 'data/pointodyssey/sample'
po_ds = KubricDataset(
    data_root=po_data_root,
    split='.',
    num_frames=8,      # subsample for notebook speed
    img_size=64,        # match our small model
    num_queries=256,
)
print(f'PointOdyssey sequences: {len(po_ds)}')
for s in po_ds.sequences:
    # Show directory contents
    contents = [p.name for p in sorted(s.iterdir())]
    print(f'  {s.name}/: {contents}')

# Load a sample
po_sample = po_ds[0]
print(f'\\n--- Sample fields ---')
for k, v in po_sample.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape} {v.dtype}  range=[{v.min():.3f}, {v.max():.3f}]')
    elif isinstance(v, dict):
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv.shape} {vv.dtype}  range=[{vv.min():.3f}, {vv.max():.3f}]')

# Also load raw anno.npz to show original data
seq_dir = po_ds.sequences[0]
anno = np.load(seq_dir / 'anno.npz', allow_pickle=True)
print(f'\\n--- Raw anno.npz fields (before processing) ---')
for k in anno.files:
    arr = anno[k]
    print(f'  {k}: {arr.shape} {arr.dtype}')
print(f'\\nKey difference from city aerial: tracks provide CORRESPONDENCE across frames!')""")))

# Cell C: Visualize RGB frames
new_cells.append(make_code_cell(src(
"""# PointOdyssey RGB frames — complex multi-object scenes
po_video = po_sample['video']  # (T, H, W, 3) float [0,1]
T_po = po_video.shape[0]

fig, axes = plt.subplots(1, T_po, figsize=(2.5 * T_po, 2.5))
for t in range(T_po):
    axes[t].imshow(po_video[t].clamp(0, 1).numpy())
    axes[t].set_title(f't={t}')
    axes[t].axis('off')
fig.suptitle('PointOdyssey: RGB Frames (complex multi-object scenes)', fontsize=14)
plt.tight_layout()
plt.show()""")))

# Cell D: Visualize GT depth
new_cells.append(make_code_cell(src(
"""# Ground truth depth — metric, sharp edges (compare to MiDaS!)
po_depth = po_sample['depth']  # (T, H, W) in meters

fig, axes = plt.subplots(2, T_po, figsize=(2.5 * T_po, 5))
for t in range(T_po):
    axes[0, t].imshow(po_video[t].clamp(0, 1).numpy())
    axes[0, t].set_title(f't={t}', fontsize=10)
    axes[0, t].axis('off')
    im = axes[1, t].imshow(po_depth[t].numpy(), cmap='plasma')
    axes[1, t].axis('off')
axes[0, 0].set_ylabel('RGB', fontsize=12)
axes[1, 0].set_ylabel('GT Depth\\n(meters)', fontsize=12)
fig.suptitle('PointOdyssey: Ground Truth Metric Depth', fontsize=14)
plt.tight_layout()
plt.show()

d = po_depth[po_depth > 0]
print(f'Depth statistics (nonzero):')
print(f'  Min:  {d.min():.2f} m')
print(f'  Max:  {d.max():.2f} m')
print(f'  Mean: {d.mean():.2f} m')
print(f'  Std:  {d.std():.2f} m')
print(f'\\nThis is METRIC depth (real-world meters), unlike MiDaS which gives relative depth.')""")))

# Cell E: Visualize 2D tracks
new_cells.append(make_code_cell(src(
"""# 2D and 3D point tracks — the key feature for correspondence supervision
# Load raw tracks to visualize before subsampling
anno = np.load(po_ds.sequences[0] / 'anno.npz', allow_pickle=True)
trajs_2d_raw = anno['trajs_2d']  # (T_total, N_tracks, 2)
visibs_raw = anno['visibs']       # (T_total, N_tracks)
trajs_3d_raw = anno['trajs_3d']  # (T_total, N_tracks, 3)

T_total, N_tracks = trajs_2d_raw.shape[:2]
print(f'Total tracks: {N_tracks:,}')
print(f'Total frames: {T_total}')

# Pick 15 random tracks that are visible in frame 0
from PIL import Image
frame0 = np.array(Image.open(sorted((po_ds.sequences[0] / 'rgbs').glob('*.jpg'))[0]))

visible_at_0 = np.where(visibs_raw[0] > 0.5)[0]
np.random.seed(42)
selected = np.random.choice(visible_at_0, size=min(15, len(visible_at_0)), replace=False)

# Plot 2D tracks overlaid on frame 0
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(frame0)
axes[0].set_title(f'2D tracks on frame 0 ({len(selected)} tracks)')
axes[0].axis('off')

colors = plt.cm.rainbow(np.linspace(0, 1, len(selected)))
for i, tidx in enumerate(selected):
    # Plot track over first 100 frames (color by time)
    n_show_frames = min(100, T_total)
    track = trajs_2d_raw[:n_show_frames, tidx]  # (100, 2)
    vis = visibs_raw[:n_show_frames, tidx]
    visible = vis > 0.5

    # Plot visible segments
    for j in range(n_show_frames - 1):
        if visible[j] and visible[j+1]:
            alpha = 0.3 + 0.7 * (j / n_show_frames)
            axes[0].plot(track[j:j+2, 0], track[j:j+2, 1],
                        color=colors[i], alpha=alpha, linewidth=1.5)
    # Mark start position
    axes[0].scatter(track[0, 0], track[0, 1], c=[colors[i]], s=30,
                   edgecolors='white', linewidths=0.5, zorder=5)

# 3D track visualization
ax3d = fig.add_subplot(122, projection='3d')
for i, tidx in enumerate(selected):
    n_show_frames = min(100, T_total)
    track_3d = trajs_3d_raw[:n_show_frames, tidx]
    vis = visibs_raw[:n_show_frames, tidx]
    visible = vis > 0.5
    if visible.sum() > 1:
        pts = track_3d[visible]
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], alpha=0.6, linewidth=1)
        ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], c=[colors[i]], s=15)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title('3D tracks (world coords)')

fig.suptitle('PointOdyssey: Point Tracks (temporal correspondence)', fontsize=14)
plt.tight_layout()
plt.show()

# Track statistics
vis_rate = visibs_raw.mean()
mean_visible_len = (visibs_raw.sum(axis=0)).mean()
print(f'\\nTrack statistics:')
print(f'  Mean visibility rate: {vis_rate:.1%}')
print(f'  Mean visible length:  {mean_visible_len:.0f} frames')
print(f'\\nThese tracks give TRACKING queries: the model learns to follow a point across time.')""")))

# Cell F: Visibility heatmap
new_cells.append(make_code_cell(src(
"""# Visibility patterns — when tracks get occluded
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Heatmap for selected tracks
vis_matrix = visibs_raw[:200, selected].T  # (n_tracks, 200_frames)
axes[0].imshow(vis_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
               interpolation='nearest')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Track')
axes[0].set_title(f'Visibility over time ({len(selected)} tracks, first 200 frames)')
axes[0].set_yticks(range(len(selected)))
axes[0].set_yticklabels([f'#{i}' for i in range(len(selected))], fontsize=8)

# Visibility statistics over time
vis_per_frame = visibs_raw.mean(axis=1)  # (T_total,)
axes[1].plot(vis_per_frame[:200], 'g-', alpha=0.8, linewidth=0.8)
axes[1].fill_between(range(min(200, T_total)), vis_per_frame[:200], alpha=0.2, color='green')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Fraction of tracks visible')
axes[1].set_title('Overall visibility rate over time')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print('Green = visible, Red = occluded')
print('Tracks get occluded as objects move behind each other — this is what L_vis learns to predict.')""")))

# Cell G: Surface normals
new_cells.append(make_code_cell(src(
"""# Ground truth surface normals — RGB visualization
# Load raw normals at full resolution for 4 frames
import glob
normal_dir = po_ds.sequences[0] / 'normals'
normal_files = sorted(normal_dir.glob('*.jpg'))

# Show 4 evenly spaced frames
frame_indices = np.linspace(0, len(normal_files)-1, 4, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, fidx in enumerate(frame_indices):
    # RGB frame
    rgb = np.array(Image.open(sorted((po_ds.sequences[0] / 'rgbs').glob('*.jpg'))[fidx]))
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f'Frame {fidx}')
    axes[0, i].axis('off')

    # Normal map: raw JPG is already in [0,255], display directly (RGB = normal direction)
    normal_img = np.array(Image.open(normal_files[fidx]))
    axes[1, i].imshow(normal_img)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('RGB', fontsize=12)
axes[1, 0].set_ylabel('Surface\\nNormals', fontsize=12)
fig.suptitle('PointOdyssey: Per-pixel Surface Normals (RGB-encoded)', fontsize=14)
plt.tight_layout()
plt.show()
print('Normal encoding: R=X, G=Y, B=Z direction')
print('These normals train L_normal (cosine similarity loss) — helps the model understand surface geometry.')""")))

# Cell H: Intrinsics
new_cells.append(make_code_cell(src(
"""# Camera intrinsics — exact from renderer
intrinsics_raw = anno['intrinsics']  # (T_total, 3, 3)
K = intrinsics_raw[0]

print('Camera intrinsics matrix K (frame 0):')
print(f'  [[{K[0,0]:.1f}  {K[0,1]:.1f}  {K[0,2]:.1f}]')
print(f'   [{K[1,0]:.1f}  {K[1,1]:.1f}  {K[1,2]:.1f}]')
print(f'   [{K[2,0]:.1f}  {K[2,1]:.1f}  {K[2,2]:.1f}]]')
print(f'\\n  fx = {K[0,0]:.1f} px  (focal length X)')
print(f'  fy = {K[1,1]:.1f} px  (focal length Y)')
print(f'  cx = {K[0,2]:.1f} px  (principal point X)')
print(f'  cy = {K[1,2]:.1f} px  (principal point Y)')

# Check if intrinsics are constant across frames
K_all = intrinsics_raw.reshape(-1, 9)
K_std = K_all.std(axis=0).max()
print(f'\\n  Intrinsics variation across {T_total} frames: max std = {K_std:.6f}')
print(f'  → {"Constant" if K_std < 0.01 else "Time-varying"} intrinsics')

# Compare with our city aerial estimated intrinsics
city_K = np.load('data_samples/city_aerial/intrinsics.npy')
print(f'\\nComparison with city aerial (estimated):')
print(f'  PointOdyssey: fx={K[0,0]:.0f}, fy={K[1,1]:.0f} (exact, from renderer)')
print(f'  City aerial:  fx={city_K[0,0]:.0f}, fy={city_K[1,1]:.0f} (estimated as 0.8×max(W,H))')
print(f'\\nExact intrinsics → better depth unprojection → cleaner 3D supervision.')""")))

# Cell I: Query type breakdown
new_cells.append(make_code_cell(src(
"""# Query type breakdown — now with TRACKING queries!
t_src_po = po_sample['t_src']
t_tgt_po = po_sample['t_tgt']
t_cam_po = po_sample['t_cam']

is_depth_po = (t_src_po == t_tgt_po) & (t_tgt_po == t_cam_po)
is_tracking_po = (t_src_po != t_tgt_po) & (t_tgt_po == t_cam_po)
is_pc_po = (t_src_po == t_tgt_po) & (t_tgt_po != t_cam_po)

n_d = is_depth_po.sum().item()
n_t = is_tracking_po.sum().item()
n_p = is_pc_po.sum().item()

tgts_po = po_sample['targets']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Query type pie chart
labels = [f'Depth ({n_d})', f'Tracking ({n_t})', f'PointCloud ({n_p})']
sizes = [n_d, n_t, n_p]
colors_pie = ['#4C72B0', '#C44E52', '#CCB974']
axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
axes[0].set_title('Query type distribution')

# Scatter on frame 0
coords_po = po_sample['coords']
H_po, W_po = 64, 64
frame0_mask_po = t_src_po == 0
for label, mask, color in [
    ('Depth', is_depth_po & frame0_mask_po, '#4C72B0'),
    ('Tracking', is_tracking_po & frame0_mask_po, '#C44E52'),
    ('PointCloud', is_pc_po & frame0_mask_po, '#CCB974'),
]:
    if mask.sum() > 0:
        pts = coords_po[mask]
        axes[1].scatter(pts[:, 0] * W_po, pts[:, 1] * H_po,
                       c=color, s=15, alpha=0.7, label=label, edgecolors='k', linewidths=0.3)
axes[1].imshow(po_video[0].clamp(0, 1).numpy(), alpha=0.3)
axes[1].legend(fontsize=9)
axes[1].set_title('Query locations on frame 0')
axes[1].axis('off')

# Mask comparison: PointOdyssey vs City Aerial
mask_names = ['mask_3d', 'mask_disp', 'mask_normal']
po_vals = [tgts_po[m].sum().item() / tgts_po[m].shape[0] * 100 for m in mask_names]
# City aerial had 0 tracking and 0 normals
city_vals = [100.0, 0.0, 0.0]  # approx from city aerial session

x = np.arange(len(mask_names))
width = 0.35
axes[2].bar(x - width/2, city_vals, width, label='City Aerial', color='orange', alpha=0.7)
axes[2].bar(x + width/2, po_vals, width, label='PointOdyssey', color='steelblue', alpha=0.7)
axes[2].set_ylabel('% valid')
axes[2].set_title('Supervision signal coverage')
axes[2].set_xticks(x)
axes[2].set_xticklabels(['3D pos', 'Displacement', 'Normals'])
axes[2].legend()
axes[2].set_ylim(0, 110)

plt.tight_layout()
plt.show()

print(f'PointOdyssey mask statistics:')
print(f'  mask_3d:     {tgts_po["mask_3d"].sum().item():.0f}/{len(tgts_po["mask_3d"])} valid')
print(f'  mask_disp:   {tgts_po["mask_disp"].sum().item():.0f}/{len(tgts_po["mask_disp"])} valid')
print(f'  mask_normal: {tgts_po["mask_normal"].sum().item():.0f}/{len(tgts_po["mask_normal"])} valid')
print(f'\\nCity aerial had mask_disp=0 and mask_normal=0 (no tracks, no normals).')
print(f'With PointOdyssey, L_disp and L_normal will be NONZERO — richer training signal!')""")))

# Cell J: Full pipeline with loss
new_cells.append(make_code_cell(src(
"""# Full pipeline: encode → decode → loss on PointOdyssey
# Uses same small model from earlier
po_batch = collate_fn([po_sample])

model.eval()
with torch.no_grad():
    po_input = po_batch['video'].permute(0, 4, 1, 2, 3)  # (1, C, T, H, W)
    po_preds = model(
        po_input,
        po_batch['coords'],
        po_batch['t_src'],
        po_batch['t_tgt'],
        po_batch['t_cam'],
        po_batch['aspect_ratio']
    )

po_losses = criterion(po_preds, po_batch['targets'])

# Compare losses: PointOdyssey vs City Aerial (from earlier)
print('Loss comparison (untrained model):')
print(f'{"Component":15s} {"City Aerial":>12s} {"PointOdyssey":>12s}')
print('-' * 42)
# City aerial values from earlier
city_losses = {
    'loss_3d': 0.228, 'loss_2d': 0.251, 'loss_vis': 0.008,
    'loss_disp': 0.000, 'loss_normal': 0.000, 'loss_conf': 0.217
}
for k in ['loss_3d', 'loss_2d', 'loss_vis', 'loss_disp', 'loss_normal', 'loss_conf', 'loss']:
    city_v = city_losses.get(k, sum(city_losses.values()))
    po_v = po_losses[k].item()
    marker = ' ← NOW ACTIVE!' if (city_v == 0 and po_v > 0) else ''
    print(f'  {k:13s} {city_v:12.4f} {po_v:12.4f}{marker}')

print(f'\\nL_disp and L_normal are now nonzero — the model gets 6/6 supervision signals!')""")))

# Cell K: Training comparison
new_cells.append(make_code_cell(src(
"""# Train 20 steps on PointOdyssey and compare loss curves
model.train()
po_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
po_loss_history = []

for step in range(20):
    v = po_batch['video'].permute(0, 4, 1, 2, 3)
    preds_po = model(
        v, po_batch['coords'], po_batch['t_src'],
        po_batch['t_tgt'], po_batch['t_cam'], po_batch['aspect_ratio']
    )
    loss_po = criterion(preds_po, po_batch['targets'])['loss']

    po_optimizer.zero_grad()
    loss_po.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    po_optimizer.step()

    po_loss_history.append(loss_po.item())
    if step % 5 == 0:
        print(f'Step {step:3d}: loss={loss_po.item():.4f}')

print(f'\\nPO loss: {po_loss_history[0]:.4f} → {po_loss_history[-1]:.4f}')

# Plot all 3 training curves
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(loss_history, 'b-o', markersize=3, label='Synthetic', alpha=0.8)
ax.plot(real_loss_history, 'r-s', markersize=3, label='City Aerial (pseudo-GT)', alpha=0.8)
ax.plot(po_loss_history, 'g-^', markersize=3, label='PointOdyssey (full GT)', alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Total Loss')
ax.set_title('Training loss: Synthetic vs City Aerial vs PointOdyssey')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print('PointOdyssey provides richer gradients (6 losses active vs 3 for city aerial).')""")))

# Cell L: Depth prediction comparison
new_cells.append(make_code_cell(src(
"""# Depth prediction on PointOdyssey: before vs after training
model.eval()
po_video_b = po_batch['video']  # (1, T, H, W, 3)

with torch.no_grad():
    po_depth_pred = model.predict_depth(po_video_b, output_resolution=(16, 16))

n_show = min(T_po, 8)
fig, axes = plt.subplots(3, n_show, figsize=(2.8 * n_show, 7.5))
for t in range(n_show):
    axes[0, t].imshow(po_video[t].clamp(0, 1).numpy())
    axes[0, t].set_title(f't={t}', fontsize=10)
    axes[0, t].axis('off')
    axes[1, t].imshow(po_depth[t].numpy(), cmap='plasma')
    axes[1, t].axis('off')
    axes[2, t].imshow(po_depth_pred[0, t].numpy(), cmap='plasma')
    axes[2, t].axis('off')

axes[0, 0].set_ylabel('RGB', fontsize=11)
axes[1, 0].set_ylabel('GT Depth\\n(metric)', fontsize=11)
axes[2, 0].set_ylabel('Model\\nPrediction', fontsize=11)
fig.suptitle('Depth: Ground Truth vs Model Prediction (PointOdyssey)', fontsize=14)
plt.tight_layout()
plt.show()
print('With proper training data (full GT), the model can learn to match metric depth.')
print('After 500k steps on the full dataset, predictions should closely match GT.')""")))

# ================================================================
# Insert new cells after cell 34 (training loss plot)
# ================================================================
insert_idx = 35  # insert before current cell 35
for cell in new_cells:
    nb['cells'].insert(insert_idx, cell)
    insert_idx += 1

# ================================================================
# Renumber existing sections
# ================================================================
# The old "Real Video Pipeline Demo" (now at higher index) becomes Section 12
# The old "Model Architecture Summary" (Section 11) becomes Section 13
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src_text = ''.join(cell['source'])
        if '## Real Video Pipeline Demo' in src_text:
            # Renumber to Section 12
            cell['source'] = [
                '---\n',
                '## 12. Real Video Pipeline Demo (Pseudo-GT)\n',
                '\n',
                'Everything above used synthetic or PointOdyssey data. Now let\'s run the full pipeline on **real drone/aerial video** with MiDaS monocular depth estimates — no ground truth tracks or normals available.\n',
                '\n',
                'Dataset: 48 frames (720p) from a city aerial drone video, with per-frame depth maps generated by MiDaS v2.1.'
            ]
        elif '## 11. Model Architecture Summary' in src_text:
            cell['source'] = [
                '---\n',
                '## 13. Model Architecture Summary\n',
                '\n',
                'Quick reference for the full D4RT pipeline.'
            ]

# Reassign cell IDs
for i, cell in enumerate(nb['cells']):
    cell['id'] = f'cell-{i}'

with open('notebooks/walkthrough.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Done! Notebook now has {len(nb["cells"])} cells.')
print(f'Inserted {len(new_cells)} new cells at position 35.')
print('Renumbered: Real Video → Section 12, Architecture Summary → Section 13.')

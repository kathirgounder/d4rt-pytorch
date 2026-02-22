"""
Determine the depth scale factor for PointOdyssey 16-bit PNG depth maps.

Compares raw depth pixel values against Z-in-camera-frame derived from
3D track annotations + extrinsics, for multiple visible tracks at frame 0.
"""

import numpy as np
from PIL import Image
import os

DATA_DIR = "/Users/kathir/Desktop/d4rt-pytorch/data/pointodyssey/sample/r4_new_f"

# ---------- 1. Load depth map for frame 0 ----------
depth_files = sorted(os.listdir(os.path.join(DATA_DIR, "depths")))
depth_path = os.path.join(DATA_DIR, "depths", depth_files[0])
depth_img = Image.open(depth_path)
depth_arr = np.array(depth_img)

print("=" * 70)
print("STEP 1: Depth map info")
print(f"  File      : {depth_files[0]}")
print(f"  dtype     : {depth_arr.dtype}")
print(f"  shape     : {depth_arr.shape}")
print(f"  min       : {depth_arr.min()}")
print(f"  max       : {depth_arr.max()}")
print(f"  mean      : {depth_arr.mean():.2f}")
print(f"  nonzero%  : {(depth_arr > 0).mean() * 100:.1f}%")
print()

# ---------- 2. Load annotations ----------
anno = np.load(os.path.join(DATA_DIR, "anno.npz"), allow_pickle=True)
print("STEP 2: Annotation keys and shapes")
for k in anno.files:
    v = anno[k]
    print(f"  {k:15s} : shape={v.shape}, dtype={v.dtype}")

intrinsics = anno["intrinsics"]   # (T, 3, 3)
extrinsics = anno["extrinsics"]   # (T, 4, 4)
trajs_3d = anno["trajs_3d"]      # (T, N, 3)
trajs_2d = anno["trajs_2d"]      # (T, N, 2)
visibs = anno["visibs"]          # (T, N)
valids = anno["valids"]          # (T, N)

K = intrinsics[0]
E = extrinsics[0]

print(f"\n  intrinsics[0]:\n{K}")
print(f"\n  extrinsics[0]:\n{E}")
print()

# ---------- 3-4. Pick visible tracks at frame 0 ----------
frame_idx = 0
vis_mask = (visibs[frame_idx] > 0.5) & (valids[frame_idx] > 0.5)
vis_indices = np.where(vis_mask)[0]
print(f"STEP 3: Number of visible+valid tracks at frame {frame_idx}: {len(vis_indices)}")

# Pick up to 20 tracks spread across the index range
num_tracks = min(20, len(vis_indices))
chosen = vis_indices[np.linspace(0, len(vis_indices) - 1, num_tracks, dtype=int)]

H, W = depth_arr.shape[:2]

print(f"\nSTEP 4-8: Comparing depth_pixel vs Z_cam for {num_tracks} tracks")
print("-" * 70)

# Extract R, t from extrinsics
R_ext = E[:3, :3]
t_ext = E[:3, 3]

# Interpretation A: extrinsics = world-to-camera  [p_cam = R @ p_world + t]
# Interpretation B: extrinsics = camera-to-world  [p_world = R @ p_cam + t]
#   => w2c: R_w2c = R.T, t_w2c = -R.T @ t

ratios_A = []
ratios_B = []
results = []

for track_idx in chosen:
    uv = trajs_2d[frame_idx, track_idx]   # (2,) - (x, y) i.e. (col, row)
    p3d = trajs_3d[frame_idx, track_idx]  # (3,) - world coords

    u_px = uv[0]  # x = column
    v_px = uv[1]  # y = row

    # Clamp to image bounds
    col = int(np.clip(np.round(u_px), 0, W - 1))
    row = int(np.clip(np.round(v_px), 0, H - 1))

    depth_pixel = float(depth_arr[row, col])

    # Interpretation A: E is w2c
    p_cam_A = R_ext @ p3d + t_ext
    z_cam_A = p_cam_A[2]

    # Interpretation B: E is c2w => invert
    R_w2c = R_ext.T
    t_w2c = -R_ext.T @ t_ext
    p_cam_B = R_w2c @ p3d + t_w2c
    z_cam_B = p_cam_B[2]

    ratio_A = depth_pixel / z_cam_A if abs(z_cam_A) > 1e-6 else float("nan")
    ratio_B = depth_pixel / z_cam_B if abs(z_cam_B) > 1e-6 else float("nan")

    ratios_A.append(ratio_A)
    ratios_B.append(ratio_B)

    results.append({
        "track": track_idx,
        "uv": (u_px, v_px),
        "pixel_rc": (row, col),
        "depth_pixel": depth_pixel,
        "z_cam_A": z_cam_A,
        "z_cam_B": z_cam_B,
        "ratio_A": ratio_A,
        "ratio_B": ratio_B,
    })

# Print table
header = f"{'Track':>7} {'u':>7} {'v':>7} {'depth_px':>10} {'Z_A(w2c)':>10} {'Z_B(c2w)':>10} {'ratio_A':>10} {'ratio_B':>10}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['track']:7d} {r['uv'][0]:7.1f} {r['uv'][1]:7.1f} "
          f"{r['depth_pixel']:10.1f} {r['z_cam_A']:10.4f} {r['z_cam_B']:10.4f} "
          f"{r['ratio_A']:10.2f} {r['ratio_B']:10.2f}")

print()
print("=" * 70)
print("STEP 9: CONCLUSIONS")
print("=" * 70)

ratios_A = np.array(ratios_A)
ratios_B = np.array(ratios_B)

# Filter out NaN and negative ratios
valid_A = ratios_A[np.isfinite(ratios_A) & (ratios_A > 0)]
valid_B = ratios_B[np.isfinite(ratios_B) & (ratios_B > 0)]

if len(valid_A) > 0:
    print(f"\nInterpretation A (extrinsics = w2c):")
    print(f"  ratio mean   = {valid_A.mean():.4f}")
    print(f"  ratio std    = {valid_A.std():.4f}")
    print(f"  ratio median = {np.median(valid_A):.4f}")
    print(f"  ratio min    = {valid_A.min():.4f}")
    print(f"  ratio max    = {valid_A.max():.4f}")
    print(f"  num valid    = {len(valid_A)}/{len(ratios_A)}")
    z_A_vals = np.array([r['z_cam_A'] for r in results])
    print(f"  Z_cam range  = [{z_A_vals.min():.4f}, {z_A_vals.max():.4f}]")
else:
    print(f"\nInterpretation A: no valid positive ratios")

if len(valid_B) > 0:
    print(f"\nInterpretation B (extrinsics = c2w, inverted to w2c):")
    print(f"  ratio mean   = {valid_B.mean():.4f}")
    print(f"  ratio std    = {valid_B.std():.4f}")
    print(f"  ratio median = {np.median(valid_B):.4f}")
    print(f"  ratio min    = {valid_B.min():.4f}")
    print(f"  ratio max    = {valid_B.max():.4f}")
    print(f"  num valid    = {len(valid_B)}/{len(ratios_B)}")
    z_B_vals = np.array([r['z_cam_B'] for r in results])
    print(f"  Z_cam range  = [{z_B_vals.min():.4f}, {z_B_vals.max():.4f}]")
else:
    print(f"\nInterpretation B: no valid positive ratios")

# Determine which interpretation gives consistent ratios
print("\n" + "-" * 70)

best = None
for label, valid_ratios in [("A (w2c)", valid_A), ("B (c2w inverted)", valid_B)]:
    if len(valid_ratios) > 0 and valid_ratios.std() / valid_ratios.mean() < 0.1:
        scale = np.median(valid_ratios)
        print(f"Interpretation {label} has consistent ratios!")
        print(f"  => depth_pixel / Z_cam ~ {scale:.2f}")
        print(f"  => To convert depth PNG to meters: depth_meters = depth_pixel / {scale:.2f}")
        print(f"  => Or equivalently: depth_scale = {1.0/scale:.6f}")
        best = (label, scale)

if best is None:
    print("Neither interpretation gives perfectly consistent ratios.")
    print("Trying to find the better one...")
    for label, valid_ratios in [("A (w2c)", valid_A), ("B (c2w inverted)", valid_B)]:
        if len(valid_ratios) > 0:
            cv = valid_ratios.std() / valid_ratios.mean()
            print(f"  {label}: CV = {cv:.4f} (lower is better)")

# Also try: maybe depth is already in the same units, just check if scale ~ 1
print()
print("Quick sanity checks:")
print(f"  depth_pixel range:  [{depth_arr.min()}, {depth_arr.max()}]")
print(f"  If uint16 max is 65535, fraction used: {depth_arr.max() / 65535.0:.4f}")

# Also check if the 3D points are in world or camera coords by projecting with K
print("\n--- Projection verification (using intrinsics) ---")
for r in results[:3]:
    track_idx = r['track']
    p3d = trajs_3d[frame_idx, track_idx]
    uv_gt = trajs_2d[frame_idx, track_idx]

    # Try A: w2c
    p_cam = R_ext @ p3d + t_ext
    if p_cam[2] > 0:
        proj = K @ p_cam
        proj_uv = proj[:2] / proj[2]
        err_A = np.linalg.norm(proj_uv - uv_gt)
    else:
        proj_uv = np.array([float('nan'), float('nan')])
        err_A = float('nan')

    # Try B: c2w inverted
    p_cam_b = R_ext.T @ p3d + (-R_ext.T @ t_ext)
    if p_cam_b[2] > 0:
        proj_b = K @ p_cam_b
        proj_uv_b = proj_b[:2] / proj_b[2]
        err_B = np.linalg.norm(proj_uv_b - uv_gt)
    else:
        proj_uv_b = np.array([float('nan'), float('nan')])
        err_B = float('nan')

    print(f"  Track {track_idx}: uv_gt=({uv_gt[0]:.1f}, {uv_gt[1]:.1f})")
    print(f"    A(w2c): proj=({proj_uv[0]:.1f}, {proj_uv[1]:.1f}), err={err_A:.2f} px")
    print(f"    B(c2w): proj=({proj_uv_b[0]:.1f}, {proj_uv_b[1]:.1f}), err={err_B:.2f} px")

print("\nDone.")

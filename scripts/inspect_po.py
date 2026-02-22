"""Inspect PointOdyssey sample data structure."""
import numpy as np
import os
import json
from PIL import Image

seq = '/Users/kathir/Desktop/d4rt-pytorch/data/pointodyssey/sample/r4_new_f'

# RGB frames
rgbs = sorted(os.listdir(os.path.join(seq, 'rgbs')))
print(f'RGB frames: {len(rgbs)}, first: {rgbs[0]}, last: {rgbs[-1]}')
img = Image.open(os.path.join(seq, 'rgbs', rgbs[0]))
print(f'  Resolution: {img.size}')

# Depths
depths_dir = os.path.join(seq, 'depths')
dfiles = sorted(os.listdir(depths_dir))
print(f'\nDepth files: {len(dfiles)}, first: {dfiles[0]}')
# PNG depth - load and check
d_img = Image.open(os.path.join(depths_dir, dfiles[0]))
d_arr = np.array(d_img)
print(f'  PIL mode: {d_img.mode}, size: {d_img.size}')
print(f'  Array shape: {d_arr.shape}, dtype: {d_arr.dtype}')
print(f'  Range: [{d_arr.min()}, {d_arr.max()}]')

# Check if 16-bit
d_img16 = Image.open(os.path.join(depths_dir, dfiles[0]))
print(f'  PIL info: {d_img16.info}')

# Normals
normals_dir = os.path.join(seq, 'normals')
nfiles = sorted(os.listdir(normals_dir))
print(f'\nNormal files: {len(nfiles)}, first: {nfiles[0]}')
n_img = Image.open(os.path.join(normals_dir, nfiles[0]))
n_arr = np.array(n_img)
print(f'  PIL mode: {n_img.mode}, size: {n_img.size}')
print(f'  Array shape: {n_arr.shape}, dtype: {n_arr.dtype}')
print(f'  Range: [{n_arr.min()}, {n_arr.max()}]')

# Annotations
print('\n--- anno.npz ---')
anno = np.load(os.path.join(seq, 'anno.npz'))
print(f'Keys: {list(anno.keys())}')
for k in anno.keys():
    v = anno[k]
    print(f'  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]')

# Info
print('\n--- info.npz ---')
info = np.load(os.path.join(seq, 'info.npz'), allow_pickle=True)
print(f'Keys: {list(info.keys())}')
for k in info.keys():
    v = info[k]
    if hasattr(v, 'shape') and v.shape != ():
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        if v.size < 20:
            print(f'    values: {v}')
    else:
        print(f'  {k}: {v}')

# Scene info JSON
print('\n--- scene_info.json ---')
with open(os.path.join(seq, 'scene_info.json')) as f:
    sj = json.load(f)
for k, v in sj.items():
    if isinstance(v, (list, dict)):
        print(f'  {k}: {type(v).__name__}, len={len(v)}')
        if isinstance(v, list) and len(v) <= 10:
            print(f'    {v}')
        elif isinstance(v, dict):
            for kk, vv in list(v.items())[:5]:
                print(f'    {kk}: {vv}')
    else:
        print(f'  {k}: {v}')

# Masks
masks_dir = os.path.join(seq, 'masks')
mfiles = sorted(os.listdir(masks_dir))
print(f'\nMask files: {len(mfiles)}, first: {mfiles[0]}')
m_img = Image.open(os.path.join(masks_dir, mfiles[0]))
m_arr = np.array(m_img)
print(f'  Mode: {m_img.mode}, shape: {m_arr.shape}, dtype: {m_arr.dtype}')
print(f'  Unique values: {np.unique(m_arr)[:20]}')

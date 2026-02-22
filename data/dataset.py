"""Base dataset class, query sampler, and collate function for D4RT."""

import random
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augmentations import TemporalSubsampling


class QuerySampler:
    """Mixed-task boundary-aware query sampler following the D4RT paper.

    Samples queries as (u, v, t_src, t_tgt, t_cam) with ground truth targets.
    Task mix: ~50% depth, ~30% tracking, ~20% point cloud.
    30% of spatial samples are drawn from depth/motion boundaries.
    """

    def __init__(
        self,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        task_mix: Optional[dict] = None,
    ):
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.task_mix = task_mix or {'depth': 0.5, 'tracking': 0.3, 'pointcloud': 0.2}

    def sample(
        self,
        T: int,
        H: int,
        W: int,
        depth: Optional[torch.Tensor] = None,
        tracks_3d: Optional[torch.Tensor] = None,
        tracks_2d: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ):
        """Sample queries with ground truth targets.

        Args:
            T, H, W: Video dimensions.
            depth: (T, H, W) depth maps, optional.
            tracks_3d: (N_tracks, T, 3) 3D track positions, optional.
            tracks_2d: (N_tracks, T, 2) 2D track positions in pixels, optional.
            visibility: (N_tracks, T) track visibility, optional.
            intrinsics: (3, 3) or (T, 3, 3) camera intrinsics.
            extrinsics: (T, 4, 4) camera-to-world extrinsics, optional.
            normals: (T, H, W, 3) surface normals, optional.

        Returns:
            coords: (N_q, 2) normalized coordinates in [0, 1]
            t_src: (N_q,) long
            t_tgt: (N_q,) long
            t_cam: (N_q,) long
            targets: dict of tensors
        """
        has_tracks = tracks_3d is not None and visibility is not None
        has_depth = depth is not None
        has_extrinsics = extrinsics is not None

        # Determine actual task counts based on available data
        n_depth = int(self.num_queries * self.task_mix['depth'])
        n_tracking = int(self.num_queries * self.task_mix['tracking']) if has_tracks else 0
        n_pointcloud = int(self.num_queries * self.task_mix['pointcloud']) if (has_depth and has_extrinsics) else 0
        # Give remaining to depth
        n_depth = self.num_queries - n_tracking - n_pointcloud

        # Compute boundary map for boundary-aware sampling
        boundary_map = self._compute_boundary_map(depth) if has_depth else None

        all_coords = []
        all_t_src = []
        all_t_tgt = []
        all_t_cam = []
        all_pos_3d = []
        all_pos_2d = []
        all_vis = []
        all_disp = []
        all_normal = []
        all_mask_3d = []
        all_mask_disp = []
        all_mask_normal = []

        # --- Depth queries ---
        if n_depth > 0:
            d = self._sample_depth_queries(
                n_depth, T, H, W, depth, intrinsics, normals, boundary_map
            )
            all_coords.append(d['coords'])
            all_t_src.append(d['t_src'])
            all_t_tgt.append(d['t_tgt'])
            all_t_cam.append(d['t_cam'])
            all_pos_3d.append(d['pos_3d'])
            all_pos_2d.append(d['pos_2d'])
            all_vis.append(d['visibility'])
            all_disp.append(d['displacement'])
            all_normal.append(d['normal'])
            all_mask_3d.append(d['mask_3d'])
            all_mask_disp.append(d['mask_disp'])
            all_mask_normal.append(d['mask_normal'])

        # --- Tracking queries ---
        if n_tracking > 0:
            t = self._sample_tracking_queries(
                n_tracking, T, H, W, tracks_3d, tracks_2d, visibility, intrinsics, extrinsics
            )
            all_coords.append(t['coords'])
            all_t_src.append(t['t_src'])
            all_t_tgt.append(t['t_tgt'])
            all_t_cam.append(t['t_cam'])
            all_pos_3d.append(t['pos_3d'])
            all_pos_2d.append(t['pos_2d'])
            all_vis.append(t['visibility'])
            all_disp.append(t['displacement'])
            all_normal.append(t['normal'])
            all_mask_3d.append(t['mask_3d'])
            all_mask_disp.append(t['mask_disp'])
            all_mask_normal.append(t['mask_normal'])

        # --- Point cloud queries ---
        if n_pointcloud > 0:
            p = self._sample_pointcloud_queries(
                n_pointcloud, T, H, W, depth, intrinsics, extrinsics, normals, boundary_map
            )
            all_coords.append(p['coords'])
            all_t_src.append(p['t_src'])
            all_t_tgt.append(p['t_tgt'])
            all_t_cam.append(p['t_cam'])
            all_pos_3d.append(p['pos_3d'])
            all_pos_2d.append(p['pos_2d'])
            all_vis.append(p['visibility'])
            all_disp.append(p['displacement'])
            all_normal.append(p['normal'])
            all_mask_3d.append(p['mask_3d'])
            all_mask_disp.append(p['mask_disp'])
            all_mask_normal.append(p['mask_normal'])

        coords = torch.cat(all_coords, dim=0)
        t_src = torch.cat(all_t_src, dim=0)
        t_tgt = torch.cat(all_t_tgt, dim=0)
        t_cam = torch.cat(all_t_cam, dim=0)

        targets = {
            'pos_3d': torch.cat(all_pos_3d, dim=0),
            'pos_2d': torch.cat(all_pos_2d, dim=0),
            'visibility': torch.cat(all_vis, dim=0),
            'displacement': torch.cat(all_disp, dim=0),
            'normal': torch.cat(all_normal, dim=0),
            'mask_3d': torch.cat(all_mask_3d, dim=0),
            'mask_disp': torch.cat(all_mask_disp, dim=0),
            'mask_normal': torch.cat(all_mask_normal, dim=0),
        }

        # Shuffle all queries together
        perm = torch.randperm(self.num_queries)
        coords = coords[perm]
        t_src = t_src[perm]
        t_tgt = t_tgt[perm]
        t_cam = t_cam[perm]
        for k in targets:
            targets[k] = targets[k][perm]

        return coords, t_src, t_tgt, t_cam, targets

    def _compute_boundary_map(self, depth):
        """Compute boundary probability map from depth using Sobel filter.

        Returns (T, H, W) float tensor with higher values at boundaries.
        """
        if depth is None:
            return None
        T, H, W = depth.shape
        # Sobel on each frame
        d = depth.unsqueeze(1).float()  # (T, 1, H, W)
        # Simple 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        sobel_x = sobel_x.view(1, 1, 3, 3).to(depth.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(depth.device)

        gx = F.conv2d(d, sobel_x, padding=1)
        gy = F.conv2d(d, sobel_y, padding=1)
        gradient_mag = (gx ** 2 + gy ** 2).sqrt().squeeze(1)  # (T, H, W)

        # Normalize to probability-like map
        for t in range(T):
            g = gradient_mag[t]
            gmax = g.max()
            if gmax > 0:
                gradient_mag[t] = g / gmax

        return gradient_mag

    def _sample_spatial_points(self, n, H, W, boundary_map_frame=None):
        """Sample n (u, v) coordinates, boundary_ratio from boundary regions."""
        if boundary_map_frame is not None and self.boundary_ratio > 0:
            n_boundary = max(int(n * self.boundary_ratio), 0)
            n_uniform = n - n_boundary

            if n_boundary > 0:
                # Boundary sampling: use boundary_map as probability
                probs = boundary_map_frame.reshape(-1)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    indices = torch.multinomial(probs, n_boundary, replacement=True)
                    v_boundary = (indices // W).float() / max(H - 1, 1)
                    u_boundary = (indices % W).float() / max(W - 1, 1)
                else:
                    u_boundary = torch.rand(n_boundary)
                    v_boundary = torch.rand(n_boundary)

                # Uniform sampling
                u_uniform = torch.rand(n_uniform)
                v_uniform = torch.rand(n_uniform)

                u = torch.cat([u_boundary, u_uniform])
                v = torch.cat([v_boundary, v_uniform])
            else:
                u = torch.rand(n)
                v = torch.rand(n)
        else:
            u = torch.rand(n)
            v = torch.rand(n)

        return torch.stack([u, v], dim=-1)  # (n, 2)

    def _unproject_pixel(self, u_norm, v_norm, t, depth, intrinsics):
        """Unproject a normalized pixel coordinate to 3D in camera frame.

        Args:
            u_norm, v_norm: Normalized coords in [0, 1].
            t: Frame index.
            depth: (T, H, W) depth maps.
            intrinsics: (3, 3) or (T, 3, 3).

        Returns:
            pos_3d: (3,) 3D position or zeros if invalid.
            valid: bool
        """
        H, W = depth.shape[1], depth.shape[2]
        px = int(u_norm * (W - 1))
        py = int(v_norm * (H - 1))
        px = min(max(px, 0), W - 1)
        py = min(max(py, 0), H - 1)

        d = depth[t, py, px].item()
        if d <= 0 or not math.isfinite(d):
            return torch.zeros(3), False

        K = intrinsics[t] if intrinsics.dim() == 3 else intrinsics
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()

        # Pixel coordinates
        pixel_x = u_norm * (W - 1)
        pixel_y = v_norm * (H - 1)

        x = (pixel_x - cx) * d / fx
        y = (pixel_y - cy) * d / fy
        z = d

        return torch.tensor([x, y, z], dtype=torch.float32), True

    def _sample_depth_queries(self, n, T, H, W, depth, intrinsics, normals, boundary_map):
        """Depth queries: t_src = t_tgt = t_cam."""
        coords_list = []
        t_src_list = []
        pos_3d_list = []
        pos_2d_list = []
        vis_list = []
        normal_list = []
        mask_3d_list = []
        mask_normal_list = []

        # Sample frames and points
        frames = torch.randint(0, T, (n,))
        for i in range(n):
            t = frames[i].item()
            bmap = boundary_map[t] if boundary_map is not None else None
            coord = self._sample_spatial_points(1, H, W, bmap).squeeze(0)  # (2,)
            coords_list.append(coord)
            t_src_list.append(t)

            if depth is not None and intrinsics is not None:
                p3d, valid = self._unproject_pixel(
                    coord[0].item(), coord[1].item(), t, depth, intrinsics
                )
                pos_3d_list.append(p3d)
                pos_2d_list.append(coord.clone())
                vis_list.append(1.0 if valid else 0.0)
                mask_3d_list.append(1.0 if valid else 0.0)
            else:
                pos_3d_list.append(torch.zeros(3))
                pos_2d_list.append(coord.clone())
                vis_list.append(1.0)
                mask_3d_list.append(0.0)

            if normals is not None:
                py = int(coord[1].item() * (H - 1))
                px = int(coord[0].item() * (W - 1))
                py = min(max(py, 0), H - 1)
                px = min(max(px, 0), W - 1)
                normal_list.append(normals[t, py, px])
                mask_normal_list.append(1.0)
            else:
                normal_list.append(torch.zeros(3))
                mask_normal_list.append(0.0)

        t_src = torch.tensor(t_src_list, dtype=torch.long)
        return {
            'coords': torch.stack(coords_list),
            't_src': t_src,
            't_tgt': t_src.clone(),
            't_cam': t_src.clone(),
            'pos_3d': torch.stack(pos_3d_list),
            'pos_2d': torch.stack(pos_2d_list),
            'visibility': torch.tensor(vis_list),
            'displacement': torch.zeros(n, 3),  # no displacement for depth queries
            'normal': torch.stack(normal_list),
            'mask_3d': torch.tensor(mask_3d_list),
            'mask_disp': torch.zeros(n),
            'mask_normal': torch.tensor(mask_normal_list),
        }

    def _sample_tracking_queries(self, n, T, H, W, tracks_3d, tracks_2d, visibility, intrinsics, extrinsics):
        """Tracking queries: fixed (u,v,t_src), varying t_tgt = t_cam."""
        N_tracks = tracks_3d.shape[0]

        # Pick random track points and target frames
        track_indices = torch.randint(0, N_tracks, (n,))
        src_frames = torch.randint(0, T, (n,))
        tgt_frames = torch.randint(0, T, (n,))

        coords_list = []
        t_src_list = []
        t_tgt_list = []
        pos_3d_list = []
        pos_2d_list = []
        vis_list = []
        disp_list = []
        mask_3d_list = []
        mask_disp_list = []

        for i in range(n):
            ti = track_indices[i].item()
            ts = src_frames[i].item()
            tt = tgt_frames[i].item()

            # Source 2D coordinate (normalized)
            if tracks_2d is not None:
                px_2d = tracks_2d[ti, ts]  # (2,) in pixel coords
                u_norm = px_2d[0].item() / max(W - 1, 1)
                v_norm = px_2d[1].item() / max(H - 1, 1)
            else:
                u_norm = random.random()
                v_norm = random.random()

            coords_list.append(torch.tensor([u_norm, v_norm], dtype=torch.float32))
            t_src_list.append(ts)
            t_tgt_list.append(tt)

            # Target 3D position at t_tgt in t_tgt's camera frame
            pos_3d_tgt = tracks_3d[ti, tt]  # (3,)
            vis_src = visibility[ti, ts].item() if visibility is not None else 1.0
            vis_tgt = visibility[ti, tt].item() if visibility is not None else 1.0
            valid = vis_src > 0.5 and vis_tgt > 0.5

            # If we have extrinsics, transform to t_tgt camera frame
            if extrinsics is not None:
                # tracks_3d is assumed to be in world coordinates
                # Transform to t_tgt camera frame
                ext_tgt = extrinsics[tt]  # (4, 4) camera-to-world
                R = ext_tgt[:3, :3]
                t_vec = ext_tgt[:3, 3]
                # world-to-camera: p_cam = R^T @ (p_world - t)
                pos_3d_cam = R.T @ (pos_3d_tgt - t_vec)
                pos_3d_list.append(pos_3d_cam)
            else:
                # Assume tracks_3d is already in camera coordinates
                pos_3d_list.append(pos_3d_tgt)

            # 2D projection at target frame
            if tracks_2d is not None:
                tgt_2d = tracks_2d[ti, tt]
                pos_2d_list.append(torch.tensor(
                    [tgt_2d[0].item() / max(W - 1, 1), tgt_2d[1].item() / max(H - 1, 1)],
                    dtype=torch.float32
                ))
            else:
                pos_2d_list.append(torch.tensor([u_norm, v_norm], dtype=torch.float32))

            vis_list.append(1.0 if valid else 0.0)
            mask_3d_list.append(1.0 if valid else 0.0)

            # Displacement: 3D position change from src to tgt
            pos_3d_src = tracks_3d[ti, ts]
            disp = pos_3d_tgt - pos_3d_src
            disp_list.append(disp)
            mask_disp_list.append(1.0 if valid else 0.0)

        return {
            'coords': torch.stack(coords_list),
            't_src': torch.tensor(t_src_list, dtype=torch.long),
            't_tgt': torch.tensor(t_tgt_list, dtype=torch.long),
            't_cam': torch.tensor(t_tgt_list, dtype=torch.long),  # t_cam = t_tgt for tracking
            'pos_3d': torch.stack(pos_3d_list),
            'pos_2d': torch.stack(pos_2d_list),
            'visibility': torch.tensor(vis_list),
            'displacement': torch.stack(disp_list),
            'normal': torch.zeros(n, 3),
            'mask_3d': torch.tensor(mask_3d_list),
            'mask_disp': torch.tensor(mask_disp_list),
            'mask_normal': torch.zeros(n),
        }

    def _sample_pointcloud_queries(self, n, T, H, W, depth, intrinsics, extrinsics, normals, boundary_map):
        """Point cloud queries: varying (u,v,t_src), fixed t_cam = reference frame."""
        ref_frame = random.randint(0, T - 1)

        coords_list = []
        t_src_list = []
        pos_3d_list = []
        pos_2d_list = []
        vis_list = []
        normal_list = []
        mask_3d_list = []
        mask_normal_list = []

        frames = torch.randint(0, T, (n,))
        for i in range(n):
            t = frames[i].item()
            bmap = boundary_map[t] if boundary_map is not None else None
            coord = self._sample_spatial_points(1, H, W, bmap).squeeze(0)
            coords_list.append(coord)
            t_src_list.append(t)

            # Unproject in source camera frame
            p3d_src, valid = self._unproject_pixel(
                coord[0].item(), coord[1].item(), t, depth, intrinsics
            )

            if valid and extrinsics is not None:
                # Transform: source camera -> world -> reference camera
                ext_src = extrinsics[t]
                ext_ref = extrinsics[ref_frame]
                R_src, t_src_vec = ext_src[:3, :3], ext_src[:3, 3]
                R_ref, t_ref_vec = ext_ref[:3, :3], ext_ref[:3, 3]

                # Camera to world
                p_world = R_src @ p3d_src + t_src_vec
                # World to reference camera
                p_ref = R_ref.T @ (p_world - t_ref_vec)
                pos_3d_list.append(p_ref)
                mask_3d_list.append(1.0)
            elif valid:
                pos_3d_list.append(p3d_src)
                mask_3d_list.append(1.0)
            else:
                pos_3d_list.append(torch.zeros(3))
                mask_3d_list.append(0.0)

            pos_2d_list.append(coord.clone())
            vis_list.append(1.0 if valid else 0.0)

            if normals is not None:
                py = int(coord[1].item() * (H - 1))
                px = int(coord[0].item() * (W - 1))
                py = min(max(py, 0), H - 1)
                px = min(max(px, 0), W - 1)
                normal_list.append(normals[t, py, px])
                mask_normal_list.append(1.0)
            else:
                normal_list.append(torch.zeros(3))
                mask_normal_list.append(0.0)

        t_src_tensor = torch.tensor(t_src_list, dtype=torch.long)
        return {
            'coords': torch.stack(coords_list),
            't_src': t_src_tensor,
            't_tgt': t_src_tensor.clone(),  # t_tgt = t_src for point cloud
            't_cam': torch.full((n,), ref_frame, dtype=torch.long),
            'pos_3d': torch.stack(pos_3d_list),
            'pos_2d': torch.stack(pos_2d_list),
            'visibility': torch.tensor(vis_list),
            'displacement': torch.zeros(n, 3),
            'normal': torch.stack(normal_list),
            'mask_3d': torch.tensor(mask_3d_list),
            'mask_disp': torch.zeros(n),
            'mask_normal': torch.tensor(mask_normal_list),
        }


class BaseD4RTDataset(Dataset):
    """Abstract base class for all D4RT datasets.

    Subclasses must implement _discover_sequences() and _load_sequence(idx).
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 48,
        img_size: int = 256,
        num_queries: int = 2048,
        transform=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_queries = num_queries
        self.transform = transform
        self.temporal_sampler = TemporalSubsampling('random_contiguous')
        self.query_sampler = QuerySampler(num_queries) if num_queries > 0 else None
        self.sequences = []

    def __len__(self):
        return len(self.sequences)

    def _discover_sequences(self):
        """Populate self.sequences. Must be implemented by subclasses."""
        raise NotImplementedError

    def _load_sequence(self, idx):
        """Load raw data for sequence idx.

        Must return a dict with at least:
            'video': np.ndarray (T_total, H, W, 3) uint8
            'original_size': (H, W) original resolution

        Optional keys:
            'depth': np.ndarray (T_total, H, W)
            'intrinsics': np.ndarray (3, 3) or (T_total, 3, 3)
            'extrinsics': np.ndarray (T_total, 4, 4)
            'tracks_3d': np.ndarray (N_tracks, T_total, 3)
            'tracks_2d': np.ndarray (N_tracks, T_total, 2)
            'visibility': np.ndarray (N_tracks, T_total)
            'normals': np.ndarray (T_total, H, W, 3)
        """
        raise NotImplementedError

    def _resize_frames(self, video, depth=None, normals=None):
        """Resize video and aligned maps to (img_size, img_size).

        Args:
            video: (T, H, W, 3) float32
            depth: (T, H, W) optional
            normals: (T, H, W, 3) optional

        Returns:
            Resized tensors.
        """
        S = self.img_size
        T, H, W, C = video.shape
        if H == S and W == S:
            return video, depth, normals

        video_t = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video_t = F.interpolate(video_t, size=(S, S), mode='bilinear', align_corners=False)
        video = video_t.permute(0, 2, 3, 1)  # (T, S, S, C)

        if depth is not None:
            depth = depth.unsqueeze(1)
            depth = F.interpolate(depth, size=(S, S), mode='nearest')
            depth = depth.squeeze(1)

        if normals is not None:
            normals_t = normals.permute(0, 3, 1, 2)
            normals_t = F.interpolate(normals_t, size=(S, S), mode='bilinear', align_corners=False)
            normals = normals_t.permute(0, 2, 3, 1)
            normals = F.normalize(normals, dim=-1)

        return video, depth, normals

    def __getitem__(self, idx):
        data = self._load_sequence(idx)

        # Convert to tensors
        video = torch.from_numpy(data['video']).float()
        orig_h, orig_w = data['original_size']
        T_total = video.shape[0]

        depth = torch.from_numpy(data['depth']).float() if data.get('depth') is not None else None
        normals = torch.from_numpy(data['normals']).float() if data.get('normals') is not None else None
        intrinsics = torch.from_numpy(data['intrinsics']).float() if data.get('intrinsics') is not None else None
        extrinsics = torch.from_numpy(data['extrinsics']).float() if data.get('extrinsics') is not None else None
        tracks_3d = torch.from_numpy(data['tracks_3d']).float() if data.get('tracks_3d') is not None else None
        tracks_2d = torch.from_numpy(data['tracks_2d']).float() if data.get('tracks_2d') is not None else None
        visibility = torch.from_numpy(data['visibility']).float() if data.get('visibility') is not None else None

        # Temporal subsampling
        frame_indices = self.temporal_sampler(T_total, self.num_frames)
        video = video[frame_indices]
        if depth is not None:
            depth = depth[frame_indices]
        if normals is not None:
            normals = normals[frame_indices]
        if extrinsics is not None:
            extrinsics = extrinsics[frame_indices]
        if intrinsics is not None and intrinsics.dim() == 3:
            intrinsics = intrinsics[frame_indices]
        if tracks_3d is not None:
            tracks_3d = tracks_3d[:, frame_indices]
        if tracks_2d is not None:
            tracks_2d = tracks_2d[:, frame_indices]
        if visibility is not None:
            visibility = visibility[:, frame_indices]

        T = video.shape[0]

        # Normalize video to [0, 1]
        if video.max() > 1.0:
            video = video / 255.0

        # Resize to img_size
        video, depth, normals = self._resize_frames(video, depth, normals)
        H, W = self.img_size, self.img_size

        # Scale intrinsics for resize
        if intrinsics is not None:
            scale_x = W / orig_w
            scale_y = H / orig_h
            if intrinsics.dim() == 2:
                intrinsics = intrinsics.clone()
                intrinsics[0, 0] *= scale_x
                intrinsics[0, 2] *= scale_x
                intrinsics[1, 1] *= scale_y
                intrinsics[1, 2] *= scale_y
            else:
                intrinsics = intrinsics.clone()
                intrinsics[:, 0, 0] *= scale_x
                intrinsics[:, 0, 2] *= scale_x
                intrinsics[:, 1, 1] *= scale_y
                intrinsics[:, 1, 2] *= scale_y

        # Scale 2D tracks for resize
        if tracks_2d is not None:
            tracks_2d = tracks_2d.clone()
            tracks_2d[..., 0] *= (W / orig_w)
            tracks_2d[..., 1] *= (H / orig_h)

        # Apply augmentations
        if self.transform is not None:
            video, depth, normals = self.transform(video, depth, normals)

        # Aspect ratio
        aspect_ratio = torch.tensor(
            [orig_w / max(orig_w, orig_h), orig_h / max(orig_w, orig_h)],
            dtype=torch.float32
        )

        # Training mode: sample queries
        if self.num_queries > 0 and self.query_sampler is not None:
            coords, t_src, t_tgt, t_cam, targets = self.query_sampler.sample(
                T, H, W,
                depth=depth,
                tracks_3d=tracks_3d,
                tracks_2d=tracks_2d,
                visibility=visibility,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                normals=normals,
            )
            result = {
                'video': video,
                'coords': coords,
                't_src': t_src,
                't_tgt': t_tgt,
                't_cam': t_cam,
                'aspect_ratio': aspect_ratio,
                'targets': targets,
            }
            # Also include raw data for potential eval during training
            if depth is not None:
                result['depth'] = depth
            return result

        # Evaluation mode: return raw data
        result = {
            'video': video,
            'aspect_ratio': aspect_ratio,
        }
        if depth is not None:
            result['depth'] = depth
        if extrinsics is not None:
            result['extrinsics'] = extrinsics
        if tracks_3d is not None:
            result['tracks'] = tracks_3d
        if visibility is not None:
            result['visibility'] = visibility
        return result


def collate_fn(batch_list):
    """Custom collate function for D4RT DataLoader.

    Stacks all tensor fields along batch dimension.
    Handles optional fields gracefully.
    """
    if not batch_list:
        return {}

    result = {}
    first = batch_list[0]

    for key in first:
        if key == 'targets':
            # Nested dict of tensors
            result['targets'] = {}
            for tkey in first['targets']:
                result['targets'][tkey] = torch.stack(
                    [b['targets'][tkey] for b in batch_list]
                )
        elif isinstance(first[key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch_list])
        else:
            # Skip non-tensor, non-dict fields
            result[key] = [b[key] for b in batch_list]

    return result

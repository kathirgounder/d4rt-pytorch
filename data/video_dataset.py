"""Concrete dataset implementations for D4RT training and evaluation."""

import math
import random
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dataset import BaseD4RTDataset, QuerySampler


# =============================================================================
# Synthetic Dataset (zero-download sanity check)
# =============================================================================


class SyntheticDataset(Dataset):
    """On-the-fly synthetic dataset for sanity checking.

    Generates random geometric scenes (ground plane + cuboids) with known
    3D ground truth. No external data needed.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 16,
        img_size: int = 128,
        num_queries: int = 512,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_queries = num_queries
        self.query_sampler = QuerySampler(num_queries)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = torch.Generator()
        rng.manual_seed(idx)

        T, H, W = self.num_frames, self.img_size, self.img_size

        # Generate camera trajectory (circular arc)
        poses = self._generate_camera_trajectory(T, rng)
        K = self._generate_intrinsics(H, W)

        # Generate scene and render
        video, depth_maps = self._render_scene(T, H, W, K, poses, rng)

        # Sample queries
        coords, t_src, t_tgt, t_cam, targets = self.query_sampler.sample(
            T, H, W,
            depth=depth_maps,
            intrinsics=K,
            extrinsics=poses,
        )

        aspect_ratio = torch.tensor([1.0, 1.0], dtype=torch.float32)

        return {
            'video': video,
            'coords': coords,
            't_src': t_src,
            't_tgt': t_tgt,
            't_cam': t_cam,
            'aspect_ratio': aspect_ratio,
            'targets': targets,
        }

    def _generate_camera_trajectory(self, T, rng):
        """Generate smooth camera trajectory (circular arc around origin).

        Uses OpenCV convention: x=right, y=down, z=forward (into scene).
        Pose is camera-to-world: p_world = R @ p_cam + t.
        """
        poses = []
        angles = torch.linspace(0, math.pi * 0.3, T)
        radius = 4.0

        for t in range(T):
            angle = angles[t].item()
            eye = torch.tensor([
                radius * math.cos(angle),
                1.5 + 0.3 * math.sin(angle * 2),
                radius * math.sin(angle),
            ])
            target = torch.zeros(3)

            # Look-at matrix (OpenCV: x=right, y=down, z=forward)
            forward = F.normalize(target - eye, dim=0)
            world_up = torch.tensor([0.0, 1.0, 0.0])
            right = F.normalize(torch.linalg.cross(forward, world_up), dim=0)
            down = F.normalize(torch.linalg.cross(forward, right), dim=0)

            R = torch.stack([right, down, forward], dim=1)  # (3, 3)
            pose = torch.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = eye
            poses.append(pose)

        return torch.stack(poses)  # (T, 4, 4)

    def _generate_intrinsics(self, H, W):
        """Generate pinhole camera intrinsics."""
        fx = fy = W * 0.8
        cx, cy = W / 2.0, H / 2.0
        return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    def _render_scene(self, T, H, W, K, poses, rng):
        """Render a simple scene via ray casting against analytical primitives.

        Produces clean, gap-free images with a ground plane, box, and sphere.
        Depth maps store z-depth in camera frame (positive forward).
        """
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()

        # Scene primitives: axis-aligned box on ground plane + sphere
        box_center = torch.tensor([
            torch.rand(1, generator=rng).item() * 2 - 1,
            0.4,
            torch.rand(1, generator=rng).item() * 2 - 1,
        ])
        box_half = torch.tensor([0.3, 0.4, 0.3])
        box_color = torch.tensor([
            torch.rand(1, generator=rng).item() * 0.4 + 0.5,
            torch.rand(1, generator=rng).item() * 0.3 + 0.2,
            torch.rand(1, generator=rng).item() * 0.4 + 0.5,
        ])

        sph_center = torch.tensor([
            torch.rand(1, generator=rng).item() * 2 - 1,
            0.5,
            torch.rand(1, generator=rng).item() * 2 - 1,
        ])
        sph_radius = 0.35 + torch.rand(1, generator=rng).item() * 0.2
        sph_color = torch.tensor([
            torch.rand(1, generator=rng).item() * 0.3 + 0.2,
            torch.rand(1, generator=rng).item() * 0.4 + 0.5,
            torch.rand(1, generator=rng).item() * 0.4 + 0.5,
        ])

        # Pixel grid → camera-space ray directions (z=1 so depth = t_hit)
        v_idx, u_idx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij',
        )
        ray_cam = torch.stack([
            (u_idx - cx) / fx,
            (v_idx - cy) / fy,
            torch.ones(H, W),
        ], dim=-1)  # (H, W, 3)

        light_dir = F.normalize(torch.tensor([1.0, -1.0, -0.5]), dim=0)

        video = torch.zeros(T, H, W, 3)
        depth_maps = torch.zeros(T, H, W)

        for t in range(T):
            R = poses[t, :3, :3]
            eye = poses[t, :3, 3]

            # Camera to world
            dirs = ray_cam @ R.T  # (H, W, 3) world-space ray directions

            t_hit = torch.full((H, W), float('inf'))

            # Background: sky gradient (blue at top, lighter at horizon)
            sky_frac = v_idx / max(H - 1, 1)
            colors = torch.stack([
                0.55 - 0.15 * sky_frac,
                0.70 - 0.20 * sky_frac,
                0.95 - 0.15 * sky_frac,
            ], dim=-1)  # (H, W, 3)

            # --- Ground plane at y=0 ---
            dir_y = dirs[..., 1]
            safe_dir_y = torch.where(dir_y.abs() < 1e-8, torch.full_like(dir_y, 1e-8), dir_y)
            gt = -eye[1] / safe_dir_y
            gx = eye[0] + gt * dirs[..., 0]
            gz = eye[2] + gt * dirs[..., 2]
            g_ok = (gt > 0.01) & (dir_y.abs() > 1e-6) & (gx.abs() < 8) & (gz.abs() < 8)

            # Checkerboard pattern
            checker = ((gx + 100).floor() + (gz + 100).floor()).long() % 2
            gc = torch.stack([
                torch.where(checker == 0, torch.tensor(0.45), torch.tensor(0.55)),
                torch.where(checker == 0, torch.tensor(0.65), torch.tensor(0.75)),
                torch.where(checker == 0, torch.tensor(0.35), torch.tensor(0.45)),
            ], dim=-1)  # (H, W, 3)

            upd = g_ok & (gt < t_hit)
            t_hit = torch.where(upd, gt, t_hit)
            colors = torch.where(upd.unsqueeze(-1).expand_as(colors), gc, colors)

            # --- Axis-aligned box ---
            bmin = box_center - box_half
            bmax = box_center + box_half
            safe_dirs = torch.where(dirs.abs() < 1e-8, torch.full_like(dirs, 1e-8), dirs)
            inv_d = 1.0 / safe_dirs
            t1 = (bmin - eye).view(1, 1, 3) * inv_d  # (H, W, 3)
            t2 = (bmax - eye).view(1, 1, 3) * inv_d
            t_lo = torch.minimum(t1, t2)
            t_hi = torch.maximum(t1, t2)
            t_near = t_lo.max(dim=-1).values  # (H, W)
            t_far = t_hi.min(dim=-1).values
            b_ok = (t_near < t_far) & (t_far > 0.01)
            bt = torch.where(t_near > 0.01, t_near, t_far)

            # Simple face shading: darken side/bottom faces
            face_axis = t_lo.argmax(dim=-1)  # which axis determined entry
            shade = torch.where(face_axis == 1, torch.tensor(0.8),
                    torch.where(face_axis == 0, torch.tensor(0.9), torch.tensor(1.0)))
            shaded_box = box_color.view(1, 1, 3) * shade.unsqueeze(-1)

            upd = b_ok & (bt < t_hit)
            t_hit = torch.where(upd, bt, t_hit)
            colors = torch.where(upd.unsqueeze(-1).expand_as(colors), shaded_box.expand_as(colors), colors)

            # --- Sphere ---
            oc = eye - sph_center  # (3,)
            a_coeff = (dirs * dirs).sum(dim=-1)  # (H, W)
            b_coeff = 2.0 * (dirs * oc.view(1, 1, 3)).sum(dim=-1)
            c_coeff = (oc * oc).sum() - sph_radius ** 2
            disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
            s_ok = disc > 0
            sqrt_disc = torch.sqrt(disc.clamp(min=0))
            st = (-b_coeff - sqrt_disc) / (2.0 * a_coeff + 1e-10)
            s_ok = s_ok & (st > 0.01)

            # Diffuse shading
            hit_pts = eye.view(1, 1, 3) + st.unsqueeze(-1) * dirs
            normals = F.normalize(hit_pts - sph_center.view(1, 1, 3), dim=-1)
            shade = (normals * light_dir.view(1, 1, 3)).sum(dim=-1).clamp(0.15, 1.0)
            shaded_sph = sph_color.view(1, 1, 3) * shade.unsqueeze(-1)

            upd = s_ok & (st < t_hit)
            t_hit = torch.where(upd, st, t_hit)
            colors = torch.where(upd.unsqueeze(-1).expand_as(colors), shaded_sph.expand_as(colors), colors)

            video[t] = colors
            # Depth: z-depth in camera frame. t_hit equals z_cam since ray_cam has z=1.
            depth_maps[t] = torch.where(t_hit < float('inf'), t_hit, torch.zeros_like(t_hit))

        video = video.clamp(0, 1)
        return video, depth_maps


# =============================================================================
# Kubric / PointOdyssey Dataset
# =============================================================================


class KubricDataset(BaseD4RTDataset):
    """Kubric MOVi-F / PointOdyssey dataset.

    Expected directory structure:
        data_root/{split}/
            sequence_NNN/
                rgbs/*.jpg or *.png
                depths/*.npy
                normals/ (optional)
                anno.npz -> trajs_2d, trajs_3d, visibility
                intrinsics.npy (optional)
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
        super().__init__(data_root, split, num_frames, img_size, num_queries, transform)
        self._discover_sequences()

    def _discover_sequences(self):
        """Find all sequence directories."""
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            # Try data_root directly (flat structure)
            split_dir = self.data_root

        self.sequences = sorted([
            d for d in split_dir.iterdir()
            if d.is_dir() and (
                (d / 'rgbs').exists() or (d / 'frames').exists() or (d / 'images').exists()
            )
        ])

        if not self.sequences:
            raise FileNotFoundError(
                f"No sequences found in {split_dir}. "
                f"Expected directories with 'rgbs/' subdirectory."
            )

    def _load_sequence(self, idx):
        seq_dir = self.sequences[idx]

        # Find RGB directory
        for name in ['rgbs', 'frames', 'images']:
            rgb_dir = seq_dir / name
            if rgb_dir.exists():
                break

        rgb_files = sorted(list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png')))

        # Load frames
        from PIL import Image
        frames = []
        for f in rgb_files:
            img = np.array(Image.open(f).convert('RGB'))
            frames.append(img)
        video = np.stack(frames)  # (T, H, W, 3) uint8
        orig_h, orig_w = video.shape[1], video.shape[2]

        # Load depth — supports .npy (Kubric) and .png (PointOdyssey 16-bit)
        depth = None
        depth_dir = seq_dir / 'depths'
        if depth_dir.exists():
            depth_npy = sorted(depth_dir.glob('*.npy'))
            depth_png = sorted(depth_dir.glob('*.png'))
            if depth_npy:
                depth = np.stack([np.load(f) for f in depth_npy])  # (T, H, W)
            elif depth_png:
                # PointOdyssey: 16-bit PNG, depth_meters = pixel_value / 65.535
                depth_frames = []
                for f in depth_png:
                    d = np.array(Image.open(f)).astype(np.float32) / 65.535
                    depth_frames.append(d)
                depth = np.stack(depth_frames)  # (T, H, W)

        # Load normals — supports .npy (Kubric) and .jpg/.png images (PointOdyssey)
        normals = None
        normal_dir = seq_dir / 'normals'
        if normal_dir.exists():
            normal_npy = sorted(normal_dir.glob('*.npy'))
            normal_img = sorted(
                list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
            )
            if normal_npy:
                normals = np.stack([np.load(f) for f in normal_npy])  # (T, H, W, 3)
            elif normal_img:
                # PointOdyssey: RGB-encoded normals, map [0,255] → [-1,1]
                normal_frames = []
                for f in normal_img:
                    n = np.array(Image.open(f).convert('RGB')).astype(np.float32)
                    n = n / 255.0 * 2.0 - 1.0  # [0,255] → [-1,1]
                    normal_frames.append(n)
                normals = np.stack(normal_frames)  # (T, H, W, 3)

        # Load annotations (tracks)
        tracks_3d = None
        tracks_2d = None
        vis = None
        intrinsics = None
        extrinsics = None
        anno_file = seq_dir / 'anno.npz'
        if anno_file.exists():
            anno = np.load(anno_file, allow_pickle=True)
            if 'trajs_3d' in anno:
                tracks_3d = anno['trajs_3d']
            if 'trajs_2d' in anno:
                tracks_2d = anno['trajs_2d']
            # Visibility: try multiple key names
            for vis_key in ['visibility', 'visibilities', 'visibs']:
                if vis_key in anno:
                    vis = anno[vis_key]
                    break
            # PointOdyssey stores intrinsics/extrinsics in anno.npz
            if 'intrinsics' in anno:
                intrinsics = anno['intrinsics']
            if 'extrinsics' in anno:
                extrinsics = anno['extrinsics']

        # Handle track axis order: PointOdyssey uses (T, N, dim),
        # our pipeline expects (N, T, dim)
        if tracks_3d is not None and tracks_3d.ndim == 3:
            T_total = video.shape[0]
            if tracks_3d.shape[0] == T_total and tracks_3d.shape[1] != T_total:
                # Shape is (T, N, dim) — transpose to (N, T, dim)
                tracks_3d = np.transpose(tracks_3d, (1, 0, 2))
        if tracks_2d is not None and tracks_2d.ndim == 3:
            T_total = video.shape[0]
            if tracks_2d.shape[0] == T_total and tracks_2d.shape[1] != T_total:
                tracks_2d = np.transpose(tracks_2d, (1, 0, 2))
        if vis is not None and vis.ndim == 2:
            T_total = video.shape[0]
            if vis.shape[0] == T_total and vis.shape[1] != T_total:
                vis = np.transpose(vis, (1, 0))

        # PointOdyssey extrinsics are world-to-camera (w2c).
        # Our pipeline expects camera-to-world (c2w): p_world = R @ p_cam + t.
        # Convert: R_c2w = R_w2c^T, t_c2w = -R_w2c^T @ t_w2c
        if extrinsics is not None and extrinsics.ndim == 3:
            # Heuristic: if loaded from anno.npz, assume w2c and convert
            if anno_file.exists() and 'extrinsics' in np.load(anno_file, allow_pickle=True):
                c2w = np.zeros_like(extrinsics)
                for i in range(extrinsics.shape[0]):
                    R_w2c = extrinsics[i, :3, :3]
                    t_w2c = extrinsics[i, :3, 3]
                    c2w[i, :3, :3] = R_w2c.T
                    c2w[i, :3, 3] = -R_w2c.T @ t_w2c
                    c2w[i, 3, 3] = 1.0
                extrinsics = c2w

        # Load intrinsics from separate file (Kubric format, fallback)
        if intrinsics is None:
            intrinsics_file = seq_dir / 'intrinsics.npy'
            if intrinsics_file.exists():
                intrinsics = np.load(intrinsics_file)  # (3, 3) or (T, 3, 3)

        # Load extrinsics from separate file (Kubric format, fallback)
        if extrinsics is None:
            extrinsics_file = seq_dir / 'extrinsics.npy'
            if extrinsics_file.exists():
                extrinsics = np.load(extrinsics_file)  # (T, 4, 4)

        return {
            'video': video,
            'original_size': (orig_h, orig_w),
            'depth': depth,
            'normals': normals,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tracks_3d': tracks_3d,
            'tracks_2d': tracks_2d,
            'visibility': vis,
        }


# =============================================================================
# Sintel Dataset
# =============================================================================


class SintelDataset(BaseD4RTDataset):
    """MPI Sintel dataset for evaluation.

    Expected directory structure:
        data_root/training/
            clean/scene_name/frame_NNNN.png
            final/scene_name/frame_NNNN.png
            depth/scene_name/frame_NNNN.dpt
            camdata_left/scene_name/frame_NNNN.cam
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'training',
        pass_name: str = 'final',
        num_frames: int = 48,
        img_size: int = 256,
        num_queries: int = 2048,
        transform=None,
    ):
        self.pass_name = pass_name
        super().__init__(data_root, split, num_frames, img_size, num_queries, transform)
        self._discover_sequences()

    def _discover_sequences(self):
        pass_dir = self.data_root / self.split / self.pass_name
        if not pass_dir.exists():
            # Try without split subdirectory
            pass_dir = self.data_root / self.pass_name
        if not pass_dir.exists():
            self.sequences = []
            return

        self.sequences = sorted([d for d in pass_dir.iterdir() if d.is_dir()])

    def _load_sequence(self, idx):
        scene_dir = self.sequences[idx]
        scene_name = scene_dir.name

        # Load RGB frames
        frame_files = sorted(scene_dir.glob('frame_*.png'))
        from PIL import Image
        frames = []
        for f in frame_files:
            img = np.array(Image.open(f).convert('RGB'))
            frames.append(img)
        video = np.stack(frames)
        orig_h, orig_w = video.shape[1], video.shape[2]

        # Load depth
        depth = None
        depth_dir = self.data_root / self.split / 'depth' / scene_name
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob('frame_*.dpt'))
            if depth_files:
                depths = [self._read_dpt(str(f)) for f in depth_files]
                depth = np.stack(depths)

        # Load camera data
        intrinsics = None
        extrinsics = None
        cam_dir = self.data_root / self.split / 'camdata_left' / scene_name
        if cam_dir.exists():
            cam_files = sorted(cam_dir.glob('frame_*.cam'))
            if cam_files:
                K_list, E_list = [], []
                for cf in cam_files:
                    K, E = self._read_cam(str(cf))
                    K_list.append(K)
                    E_list.append(E)
                intrinsics = np.stack(K_list)  # (T, 3, 3)
                extrinsics = np.stack(E_list)  # (T, 4, 4)

        return {
            'video': video,
            'original_size': (orig_h, orig_w),
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'normals': None,
            'tracks_3d': None,
            'tracks_2d': None,
            'visibility': None,
        }

    @staticmethod
    def _read_dpt(filename):
        """Read Sintel .dpt depth file."""
        with open(filename, 'rb') as f:
            magic = struct.unpack('f', f.read(4))[0]
            if magic != 202021.25:
                raise ValueError(f"Invalid .dpt file: {filename}")
            w = struct.unpack('i', f.read(4))[0]
            h = struct.unpack('i', f.read(4))[0]
            data = np.fromfile(f, dtype=np.float32).reshape(h, w)
        return data

    @staticmethod
    def _read_cam(filename):
        """Read Sintel camera file. Returns intrinsics (3,3) and extrinsics (4,4)."""
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Parse intrinsics (first 3 lines)
        K = np.eye(3, dtype=np.float32)
        for i in range(3):
            vals = [float(x) for x in lines[i].split()]
            K[i] = vals

        # Parse extrinsics (next 4 lines)
        E = np.eye(4, dtype=np.float32)
        for i in range(4):
            vals = [float(x) for x in lines[3 + i].split()]
            E[i] = vals

        return K, E


# =============================================================================
# ScanNet Dataset
# =============================================================================


class ScanNetDataset(BaseD4RTDataset):
    """ScanNet indoor RGB-D dataset.

    Expected directory structure:
        data_root/
            scene0000_00/
                color/*.jpg
                depth/*.png (16-bit, millimeters)
                pose/*.txt (4x4 matrices)
                intrinsic/intrinsic_depth.txt
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 48,
        img_size: int = 256,
        num_queries: int = 2048,
        frame_skip: int = 10,
        transform=None,
    ):
        self.frame_skip = frame_skip
        super().__init__(data_root, split, num_frames, img_size, num_queries, transform)
        self._discover_sequences()

    def _discover_sequences(self):
        """Discover ScanNet scene directories."""
        # Try split file
        split_file = self.data_root.parent / f'scannetv2_{self.split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            self.sequences = [
                self.data_root / sid for sid in scene_ids
                if (self.data_root / sid / 'color').exists()
            ]
        else:
            self.sequences = sorted([
                d for d in self.data_root.iterdir()
                if d.is_dir() and (d / 'color').exists()
            ])

    def _load_sequence(self, idx):
        scene_dir = self.sequences[idx]

        # Get frame list with skip
        color_dir = scene_dir / 'color'
        all_frames = sorted(color_dir.glob('*.jpg'))
        if not all_frames:
            all_frames = sorted(color_dir.glob('*.png'))
        selected = all_frames[::self.frame_skip]

        # Load RGB
        from PIL import Image
        frames = []
        for f in selected:
            img = np.array(Image.open(f).convert('RGB'))
            frames.append(img)
        video = np.stack(frames)
        orig_h, orig_w = video.shape[1], video.shape[2]

        # Load depth (16-bit PNG, millimeters)
        depth = None
        depth_dir = scene_dir / 'depth'
        if depth_dir.exists():
            depths = []
            for f in selected:
                frame_id = f.stem
                depth_file = depth_dir / f'{frame_id}.png'
                if depth_file.exists():
                    d = np.array(Image.open(depth_file)).astype(np.float32) / 1000.0
                    depths.append(d)
                else:
                    depths.append(np.zeros((orig_h, orig_w), dtype=np.float32))
            depth = np.stack(depths)

        # Load intrinsics
        intrinsics = None
        intr_file = scene_dir / 'intrinsic' / 'intrinsic_depth.txt'
        if intr_file.exists():
            K = np.loadtxt(str(intr_file))[:3, :3].astype(np.float32)
            intrinsics = K

        # Load poses
        extrinsics = None
        pose_dir = scene_dir / 'pose'
        if pose_dir.exists():
            poses = []
            for f in selected:
                frame_id = f.stem
                pose_file = pose_dir / f'{frame_id}.txt'
                if pose_file.exists():
                    pose = np.loadtxt(str(pose_file)).astype(np.float32)
                    if pose.shape == (4, 4):
                        poses.append(pose)
                    else:
                        poses.append(np.eye(4, dtype=np.float32))
                else:
                    poses.append(np.eye(4, dtype=np.float32))
            extrinsics = np.stack(poses)

        return {
            'video': video,
            'original_size': (orig_h, orig_w),
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'normals': None,
            'tracks_3d': None,
            'tracks_2d': None,
            'visibility': None,
        }


# =============================================================================
# Generic Video Dataset
# =============================================================================


class VideoDataset(BaseD4RTDataset):
    """Generic video dataset from a directory of frame sequences.

    Expected structure:
        data_root/{split}/
            sequence_001/
                rgbs/ (or frames/ or images/)
                    frame_0001.png, ...
                depths/ (optional)
                anno.npz (optional)
                intrinsics.npy (optional)

    Falls back to data_root/ directly if split subdirectory doesn't exist.
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
        super().__init__(data_root, split, num_frames, img_size, num_queries, transform)
        self._discover_sequences()

    def _discover_sequences(self):
        search_dir = self.data_root / self.split
        if not search_dir.exists():
            search_dir = self.data_root

        self.sequences = sorted([
            d for d in search_dir.iterdir()
            if d.is_dir() and any(
                (d / name).exists() for name in ['rgbs', 'frames', 'images']
            )
        ])

        # Also check for video files directly
        if not self.sequences:
            video_files = sorted(
                list(search_dir.glob('*.mp4')) +
                list(search_dir.glob('*.avi'))
            )
            self.sequences = video_files

    def _load_sequence(self, idx):
        seq_path = self.sequences[idx]

        if seq_path.is_file():
            # Load from video file
            return self._load_video_file(seq_path)

        # Load from directory
        for name in ['rgbs', 'frames', 'images']:
            rgb_dir = seq_path / name
            if rgb_dir.exists():
                break

        from PIL import Image
        frame_files = sorted(
            list(rgb_dir.glob('*.jpg')) +
            list(rgb_dir.glob('*.png'))
        )

        frames = []
        for f in frame_files:
            img = np.array(Image.open(f).convert('RGB'))
            frames.append(img)
        video = np.stack(frames)
        orig_h, orig_w = video.shape[1], video.shape[2]

        # Try to load depth
        depth = None
        depth_dir = seq_path / 'depths'
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob('*.npy'))
            if depth_files:
                depth = np.stack([np.load(f) for f in depth_files])

        # Try to load annotations
        tracks_3d = tracks_2d = vis = None
        anno_file = seq_path / 'anno.npz'
        if anno_file.exists():
            anno = np.load(anno_file, allow_pickle=True)
            tracks_3d = anno.get('trajs_3d')
            tracks_2d = anno.get('trajs_2d')
            vis = anno.get('visibility', anno.get('visibilities'))

        # Try to load intrinsics
        intrinsics = None
        intr_file = seq_path / 'intrinsics.npy'
        if intr_file.exists():
            intrinsics = np.load(intr_file)

        extrinsics = None
        ext_file = seq_path / 'extrinsics.npy'
        if ext_file.exists():
            extrinsics = np.load(ext_file)

        return {
            'video': video,
            'original_size': (orig_h, orig_w),
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'normals': None,
            'tracks_3d': tracks_3d,
            'tracks_2d': tracks_2d,
            'visibility': vis,
        }

    def _load_video_file(self, video_path):
        """Load from a video file using OpenCV or torchvision."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            video = np.stack(frames)
        except ImportError:
            raise ImportError("OpenCV required for video file loading: pip install opencv-python")

        orig_h, orig_w = video.shape[1], video.shape[2]
        return {
            'video': video,
            'original_size': (orig_h, orig_w),
            'depth': None,
            'intrinsics': None,
            'extrinsics': None,
            'normals': None,
            'tracks_3d': None,
            'tracks_2d': None,
            'visibility': None,
        }

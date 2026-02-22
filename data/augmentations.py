"""Video augmentation pipeline for D4RT training.

Spatial transforms are applied identically across all frames to preserve
temporal consistency. Color transforms can vary slightly per frame.
"""

import random
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class AugmentationConfig:
    """Configuration for video augmentations."""
    # Spatial
    random_crop: bool = True
    crop_scale: tuple = (0.8, 1.0)
    random_flip: bool = True
    flip_prob: float = 0.5
    # Color
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


class VideoAugmentation:
    """Apply augmentations consistently across all frames of a video.

    Spatial transforms (crop, flip) use the same random parameters for all
    frames so that depth maps, normals, and coordinate annotations stay aligned.
    """

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()

    def __call__(self, video, depth=None, normals=None):
        """
        Args:
            video: (T, H, W, 3) float32 in [0, 1]
            depth: (T, H, W) float32, optional
            normals: (T, H, W, 3) float32, optional

        Returns:
            Augmented (video, depth, normals) with same shapes.
        """
        # Random crop (same region for all frames)
        if self.config.random_crop:
            video, depth, normals = self._random_crop(video, depth, normals)

        # Random horizontal flip (same for all frames)
        if self.config.random_flip and random.random() < self.config.flip_prob:
            video = video.flip(dims=[2])  # flip W
            if depth is not None:
                depth = depth.flip(dims=[2])
            if normals is not None:
                normals = normals.flip(dims=[2])
                normals[..., 0] = -normals[..., 0]  # negate X component

        # Color jitter (consistent parameters across frames)
        if self.config.color_jitter:
            video = self._color_jitter(video)

        return video, depth, normals

    def _random_crop(self, video, depth, normals):
        """Random crop + resize back to original size, same region all frames."""
        T, H, W, C = video.shape
        scale = random.uniform(*self.config.crop_scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        video = video[:, top:top + crop_h, left:left + crop_w, :]
        if depth is not None:
            depth = depth[:, top:top + crop_h, left:left + crop_w]
        if normals is not None:
            normals = normals[:, top:top + crop_h, left:left + crop_w, :]

        # Resize back to (H, W)
        # video: (T, h, w, C) -> (T, C, h, w) for interpolate -> (T, C, H, W) -> back
        video = video.permute(0, 3, 1, 2)  # (T, C, h, w)
        video = F.interpolate(video, size=(H, W), mode='bilinear', align_corners=False)
        video = video.permute(0, 2, 3, 1)  # (T, H, W, C)

        if depth is not None:
            depth = depth.unsqueeze(1)  # (T, 1, h, w)
            depth = F.interpolate(depth, size=(H, W), mode='nearest')
            depth = depth.squeeze(1)

        if normals is not None:
            normals = normals.permute(0, 3, 1, 2)
            normals = F.interpolate(normals, size=(H, W), mode='bilinear', align_corners=False)
            normals = normals.permute(0, 2, 3, 1)
            normals = F.normalize(normals, dim=-1)  # re-normalize after interpolation

        return video, depth, normals

    def _color_jitter(self, video):
        """Apply color jitter with same params across all frames."""
        # Sample jitter parameters once for temporal consistency
        brightness = 1.0 + random.uniform(-self.config.brightness, self.config.brightness)
        contrast = 1.0 + random.uniform(-self.config.contrast, self.config.contrast)
        saturation = 1.0 + random.uniform(-self.config.saturation, self.config.saturation)
        hue = random.uniform(-self.config.hue, self.config.hue)

        # Apply brightness
        video = video * brightness

        # Apply contrast
        gray = video.mean(dim=-1, keepdim=True)
        video = gray + contrast * (video - gray)

        # Apply saturation
        gray = video.mean(dim=-1, keepdim=True)
        video = gray + saturation * (video - gray)

        # Apply hue shift (simplified: rotate in RGB space)
        if abs(hue) > 1e-4:
            angle = hue * 2 * math.pi
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            # Approximate hue rotation via luminance-preserving rotation
            r, g, b = video[..., 0], video[..., 1], video[..., 2]
            nr = r * (cos_a + (1 - cos_a) / 3) + g * ((1 - cos_a) / 3 - sin_a * math.sqrt(1/3)) + b * ((1 - cos_a) / 3 + sin_a * math.sqrt(1/3))
            ng = r * ((1 - cos_a) / 3 + sin_a * math.sqrt(1/3)) + g * (cos_a + (1 - cos_a) / 3) + b * ((1 - cos_a) / 3 - sin_a * math.sqrt(1/3))
            nb = r * ((1 - cos_a) / 3 - sin_a * math.sqrt(1/3)) + g * ((1 - cos_a) / 3 + sin_a * math.sqrt(1/3)) + b * (cos_a + (1 - cos_a) / 3)
            video = torch.stack([nr, ng, nb], dim=-1)

        return video.clamp(0.0, 1.0)


class TemporalSubsampling:
    """Temporal subsampling strategies for video clips."""

    def __init__(self, strategy='random_contiguous'):
        """
        Args:
            strategy: 'random_contiguous' | 'uniform' | 'random'
        """
        self.strategy = strategy

    def __call__(self, total_frames, num_frames):
        """Return list of frame indices to sample.

        Args:
            total_frames: Total number of available frames.
            num_frames: Desired number of frames.

        Returns:
            List of integer frame indices (sorted).
        """
        if total_frames <= num_frames:
            indices = list(range(total_frames))
            # Pad by repeating last frame
            while len(indices) < num_frames:
                indices.append(indices[-1])
            return indices

        if self.strategy == 'random_contiguous':
            start = random.randint(0, total_frames - num_frames)
            return list(range(start, start + num_frames))
        elif self.strategy == 'uniform':
            import numpy as np
            return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
        elif self.strategy == 'random':
            return sorted(random.sample(range(total_frames), num_frames))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

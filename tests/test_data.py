"""Tests for D4RT data pipeline."""

import pytest
import torch
import numpy as np

from data import (
    VideoDataset, KubricDataset, SintelDataset, ScanNetDataset,
    SyntheticDataset, collate_fn,
    VideoAugmentation, TemporalSubsampling, AugmentationConfig,
)
from data.dataset import QuerySampler, BaseD4RTDataset


class TestAugmentations:
    """Tests for video augmentation pipeline."""

    def test_augmentation_config_defaults(self):
        config = AugmentationConfig()
        assert config.random_crop is True
        assert config.random_flip is True
        assert config.color_jitter is True

    def test_video_augmentation_shapes(self):
        aug = VideoAugmentation(AugmentationConfig())
        video = torch.rand(8, 64, 64, 3)
        depth = torch.rand(8, 64, 64)
        normals = torch.randn(8, 64, 64, 3)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        v_out, d_out, n_out = aug(video, depth, normals)

        assert v_out.shape == (8, 64, 64, 3)
        assert d_out.shape == (8, 64, 64)
        assert n_out.shape == (8, 64, 64, 3)

    def test_video_augmentation_no_depth(self):
        aug = VideoAugmentation(AugmentationConfig())
        video = torch.rand(8, 64, 64, 3)

        v_out, d_out, n_out = aug(video, None, None)

        assert v_out.shape == (8, 64, 64, 3)
        assert d_out is None
        assert n_out is None

    def test_video_augmentation_value_range(self):
        aug = VideoAugmentation(AugmentationConfig())
        video = torch.rand(4, 32, 32, 3)

        v_out, _, _ = aug(video)

        assert v_out.min() >= 0.0
        assert v_out.max() <= 1.0

    def test_temporal_subsampling_contiguous(self):
        sampler = TemporalSubsampling('random_contiguous')
        indices = sampler(100, 16)

        assert len(indices) == 16
        # Check contiguous
        for i in range(1, len(indices)):
            assert indices[i] == indices[i - 1] + 1

    def test_temporal_subsampling_uniform(self):
        sampler = TemporalSubsampling('uniform')
        indices = sampler(100, 10)

        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_temporal_subsampling_padding(self):
        sampler = TemporalSubsampling('random_contiguous')
        indices = sampler(5, 10)

        assert len(indices) == 10
        # Last elements should repeat the final frame
        assert indices[-1] == 4


class TestQuerySampler:
    """Tests for the query sampling strategy."""

    def test_basic_sampling_depth_only(self):
        sampler = QuerySampler(num_queries=100)
        T, H, W = 8, 64, 64
        depth = torch.rand(T, H, W) * 5.0 + 0.1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])

        coords, t_src, t_tgt, t_cam, targets = sampler.sample(
            T, H, W, depth=depth, intrinsics=K
        )

        assert coords.shape == (100, 2)
        assert t_src.shape == (100,)
        assert t_tgt.shape == (100,)
        assert t_cam.shape == (100,)
        assert targets['pos_3d'].shape == (100, 3)
        assert targets['pos_2d'].shape == (100, 2)
        assert targets['visibility'].shape == (100,)
        assert targets['mask_3d'].shape == (100,)

    def test_sampling_with_tracks(self):
        sampler = QuerySampler(num_queries=100)
        T, H, W = 8, 64, 64
        depth = torch.rand(T, H, W) * 5.0 + 0.1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
        tracks_3d = torch.randn(50, T, 3)
        tracks_2d = torch.rand(50, T, 2) * 63
        visibility = torch.ones(50, T)

        coords, t_src, t_tgt, t_cam, targets = sampler.sample(
            T, H, W,
            depth=depth,
            tracks_3d=tracks_3d,
            tracks_2d=tracks_2d,
            visibility=visibility,
            intrinsics=K,
        )

        assert coords.shape == (100, 2)
        assert targets['displacement'].shape == (100, 3)

    def test_coord_ranges(self):
        sampler = QuerySampler(num_queries=200)
        T, H, W = 4, 32, 32
        depth = torch.rand(T, H, W) * 3.0 + 0.1
        K = torch.tensor([[25.0, 0, 16], [0, 25.0, 16], [0, 0, 1]])

        coords, t_src, t_tgt, t_cam, targets = sampler.sample(
            T, H, W, depth=depth, intrinsics=K
        )

        assert coords.min() >= 0.0
        assert coords.max() <= 1.0
        assert t_src.min() >= 0
        assert t_src.max() < T
        assert t_tgt.min() >= 0
        assert t_tgt.max() < T
        assert t_cam.min() >= 0
        assert t_cam.max() < T

    def test_no_data_sampling(self):
        """Test sampling with no depth/tracks — should still produce valid shapes."""
        sampler = QuerySampler(num_queries=50)
        T, H, W = 4, 32, 32

        coords, t_src, t_tgt, t_cam, targets = sampler.sample(T, H, W)

        assert coords.shape == (50, 2)
        assert targets['pos_3d'].shape == (50, 3)
        # mask_3d should be mostly 0 since no depth available
        assert targets['mask_3d'].sum() == 0


class TestSyntheticDataset:
    """Tests for the on-the-fly synthetic dataset."""

    def test_getitem_shapes(self):
        ds = SyntheticDataset(num_samples=5, num_frames=4, img_size=32, num_queries=64)

        sample = ds[0]

        assert sample['video'].shape == (4, 32, 32, 3)
        assert sample['coords'].shape == (64, 2)
        assert sample['t_src'].shape == (64,)
        assert sample['t_tgt'].shape == (64,)
        assert sample['t_cam'].shape == (64,)
        assert sample['aspect_ratio'].shape == (2,)
        assert sample['targets']['pos_3d'].shape == (64, 3)
        assert sample['targets']['mask_3d'].shape == (64,)

    def test_deterministic(self):
        ds = SyntheticDataset(num_samples=5, num_frames=4, img_size=32, num_queries=32)

        s1 = ds[0]
        s2 = ds[0]

        assert torch.allclose(s1['video'], s2['video'])

    def test_video_range(self):
        ds = SyntheticDataset(num_samples=3, num_frames=4, img_size=32, num_queries=32)

        sample = ds[0]

        assert sample['video'].min() >= 0.0
        assert sample['video'].max() <= 1.0

    def test_length(self):
        ds = SyntheticDataset(num_samples=42)
        assert len(ds) == 42


class TestCollate:
    """Tests for the collate function."""

    def test_collate_training_batch(self):
        ds = SyntheticDataset(num_samples=4, num_frames=4, img_size=32, num_queries=32)

        items = [ds[i] for i in range(3)]
        batch = collate_fn(items)

        assert batch['video'].shape == (3, 4, 32, 32, 3)
        assert batch['coords'].shape == (3, 32, 2)
        assert batch['t_src'].shape == (3, 32)
        assert batch['aspect_ratio'].shape == (3, 2)
        assert batch['targets']['pos_3d'].shape == (3, 32, 3)
        assert batch['targets']['mask_3d'].shape == (3, 32)

    def test_collate_with_dataloader(self):
        from torch.utils.data import DataLoader

        ds = SyntheticDataset(num_samples=8, num_frames=4, img_size=32, num_queries=32)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, num_workers=0)

        batch = next(iter(loader))

        assert batch['video'].shape[0] == 2
        assert batch['targets']['pos_3d'].shape[0] == 2


class TestImports:
    """Test that all expected imports work."""

    def test_train_imports(self):
        """Verify the exact imports train.py uses."""
        from data import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset, collate_fn
        from data.augmentations import VideoAugmentation, TemporalSubsampling, AugmentationConfig

        assert VideoDataset is not None
        assert KubricDataset is not None
        assert SintelDataset is not None
        assert ScanNetDataset is not None
        assert collate_fn is not None
        assert VideoAugmentation is not None
        assert TemporalSubsampling is not None
        assert AugmentationConfig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

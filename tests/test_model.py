"""Tests for D4RT model components."""

import pytest
import torch
import torch.nn as nn

from models import D4RT, D4RTEncoder, D4RTDecoder
from models.embeddings import FourierEmbedding, TimestepEmbedding, PatchEmbeddingFast
from losses import D4RTLoss


class TestEmbeddings:
    """Tests for embedding modules."""

    def test_fourier_embedding(self):
        embed = FourierEmbedding(embed_dim=256, num_frequencies=32)
        coords = torch.rand(2, 100, 2)  # (B, N, 2)

        output = embed(coords)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_timestep_embedding(self):
        embed = TimestepEmbedding(max_timesteps=64, embed_dim=256)
        t_src = torch.randint(0, 64, (2, 100))
        t_tgt = torch.randint(0, 64, (2, 100))
        t_cam = torch.randint(0, 64, (2, 100))

        src_emb, tgt_emb, cam_emb = embed(t_src, t_tgt, t_cam)

        assert src_emb.shape == (2, 100, 256)
        assert tgt_emb.shape == (2, 100, 256)
        assert cam_emb.shape == (2, 100, 256)

    def test_patch_embedding(self):
        embed = PatchEmbeddingFast(patch_size=9, embed_dim=256)
        frames = torch.rand(2, 16, 3, 64, 64)  # (B, T, C, H, W)
        coords = torch.rand(2, 50, 2)  # (B, N, 2)
        t_src = torch.randint(0, 16, (2, 50))

        output = embed(frames, coords, t_src)

        assert output.shape == (2, 50, 256)
        assert not torch.isnan(output).any()


class TestEncoder:
    """Tests for the encoder."""

    def test_encoder_forward(self):
        encoder = D4RTEncoder(
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            embed_dim=256,
            depth=4,
            num_heads=4
        )

        video = torch.rand(2, 3, 16, 64, 64)  # (B, C, T, H, W)
        aspect_ratio = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        output = encoder(video, aspect_ratio)

        # Expected: (num_frames / 2) * (img_size / 8)^2 patches
        expected_patches = (16 // 2) * (64 // 8) * (64 // 8)
        assert output.shape == (2, expected_patches, 256)

    def test_encoder_no_aspect_ratio(self):
        encoder = D4RTEncoder(
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            embed_dim=256,
            depth=4,
            num_heads=4
        )

        video = torch.rand(2, 3, 16, 64, 64)

        output = encoder(video)

        expected_patches = (16 // 2) * (64 // 8) * (64 // 8)
        assert output.shape == (2, expected_patches, 256)


class TestDecoder:
    """Tests for the decoder."""

    def test_decoder_forward(self):
        decoder = D4RTDecoder(
            embed_dim=256,
            depth=4,
            num_heads=4,
            max_timesteps=32,
            patch_size=5
        )

        encoder_features = torch.rand(2, 512, 256)  # (B, N_enc, C)
        frames = torch.rand(2, 16, 3, 64, 64)  # (B, T, C, H, W)
        coords = torch.rand(2, 100, 2)
        t_src = torch.randint(0, 16, (2, 100))
        t_tgt = torch.randint(0, 16, (2, 100))
        t_cam = torch.randint(0, 16, (2, 100))

        outputs = decoder(encoder_features, frames, coords, t_src, t_tgt, t_cam)

        assert 'pos_3d' in outputs
        assert 'pos_2d' in outputs
        assert 'visibility' in outputs
        assert 'displacement' in outputs
        assert 'normal' in outputs
        assert 'confidence' in outputs

        assert outputs['pos_3d'].shape == (2, 100, 3)
        assert outputs['pos_2d'].shape == (2, 100, 2)
        assert outputs['confidence'].shape == (2, 100, 1)


class TestD4RT:
    """Tests for the full D4RT model."""

    def test_forward(self):
        model = D4RT(
            encoder_variant='base',
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            decoder_depth=4,
            max_timesteps=32,
            query_patch_size=5
        )

        # Reduce model size for testing
        model.encoder = D4RTEncoder(
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            embed_dim=256,
            depth=4,
            num_heads=4
        )
        model.decoder = D4RTDecoder(
            embed_dim=256,
            depth=4,
            num_heads=4,
            max_timesteps=32,
            patch_size=5
        )

        video = torch.rand(2, 16, 64, 64, 3)  # (B, T, H, W, C)
        coords = torch.rand(2, 100, 2)
        t_src = torch.randint(0, 16, (2, 100))
        t_tgt = torch.randint(0, 16, (2, 100))
        t_cam = torch.randint(0, 16, (2, 100))

        outputs = model(video, coords, t_src, t_tgt, t_cam)

        assert outputs['pos_3d'].shape == (2, 100, 3)

    def test_encode_decode_separate(self):
        """Test that encode and decode can be called separately."""
        model = D4RT(
            encoder_variant='base',
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            decoder_depth=4,
            max_timesteps=32,
            query_patch_size=5
        )

        # Use smaller model
        model.encoder = D4RTEncoder(
            img_size=64,
            temporal_size=16,
            patch_size=(2, 8, 8),
            embed_dim=256,
            depth=4,
            num_heads=4
        )
        model.decoder = D4RTDecoder(
            embed_dim=256,
            depth=4,
            num_heads=4,
            max_timesteps=32,
            patch_size=5
        )

        video = torch.rand(2, 16, 64, 64, 3)

        # Encode
        features = model.encode(video)
        assert features.dim() == 3

        # Decode with different queries
        frames = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        coords1 = torch.rand(2, 50, 2)
        coords2 = torch.rand(2, 100, 2)
        t_idx = torch.randint(0, 16, (2, 50))
        t_idx2 = torch.randint(0, 16, (2, 100))

        out1 = model.decode(features, frames, coords1, t_idx, t_idx, t_idx)
        out2 = model.decode(features, frames, coords2, t_idx2, t_idx2, t_idx2)

        assert out1['pos_3d'].shape == (2, 50, 3)
        assert out2['pos_3d'].shape == (2, 100, 3)


class TestLoss:
    """Tests for loss functions."""

    def test_d4rt_loss(self):
        criterion = D4RTLoss()

        predictions = {
            'pos_3d': torch.rand(2, 100, 3),
            'pos_2d': torch.rand(2, 100, 2),
            'visibility': torch.rand(2, 100, 1),
            'displacement': torch.rand(2, 100, 3),
            'normal': torch.rand(2, 100, 3),
            'confidence': torch.sigmoid(torch.rand(2, 100, 1))
        }

        targets = {
            'pos_3d': torch.rand(2, 100, 3),
            'pos_2d': torch.rand(2, 100, 2),
            'visibility': torch.randint(0, 2, (2, 100)).float(),
            'displacement': torch.rand(2, 100, 3),
            'normal': torch.rand(2, 100, 3),
            'mask_3d': torch.ones(2, 100),
            'mask_disp': torch.ones(2, 100),
            'mask_normal': torch.ones(2, 100)
        }

        losses = criterion(predictions, targets)

        assert 'loss' in losses
        assert 'loss_3d' in losses
        assert 'loss_2d' in losses
        assert 'loss_vis' in losses
        assert losses['loss'].requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

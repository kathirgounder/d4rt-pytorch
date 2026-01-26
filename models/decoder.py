"""D4RT Decoder: Lightweight cross-attention transformer for point queries."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import FourierEmbedding, TimestepEmbedding, PatchEmbeddingFast


class CrossAttention(nn.Module):
    """Efficient cross-attention using PyTorch's scaled_dot_product_attention.

    Automatically uses FlashAttention or memory-efficient attention when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            key_value: (B, N_kv, C) key-value tokens (encoder features)
            mask: Optional attention mask

        Returns:
            out: (B, N_q, C)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's efficient attention (FlashAttention when available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with cross-attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        query: torch.Tensor,
        encoder_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            encoder_features: (B, N_kv, C) encoder output (Global Scene Representation)

        Returns:
            out: (B, N_q, C)
        """
        # Cross-attention
        query = query + self.cross_attn(
            self.norm1(query),
            self.norm_kv(encoder_features)
        )
        # MLP
        query = query + self.mlp(self.norm2(query))

        return query


class D4RTDecoder(nn.Module):
    """D4RT Pointwise Decoder.

    Lightweight cross-attention transformer that decodes queries independently.
    Each query (u, v, t_src, t_tgt, t_cam) is decoded to predict:
    - 3D position (X, Y, Z)
    - 2D position (u, v) reprojection
    - Visibility flag
    - Motion displacement
    - Surface normal
    - Confidence score
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_timesteps: int = 128,
        patch_size: int = 9,
        num_fourier_freqs: int = 64,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Query embeddings
        self.fourier_embed = FourierEmbedding(embed_dim, num_fourier_freqs)
        self.timestep_embed = TimestepEmbedding(max_timesteps, embed_dim)
        self.patch_embed = PatchEmbeddingFast(patch_size, embed_dim)

        # Learnable query token (base)
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, True, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        self.head_3d = nn.Linear(embed_dim, 3)  # 3D position
        self.head_2d = nn.Linear(embed_dim, 2)  # 2D position
        self.head_vis = nn.Linear(embed_dim, 1)  # Visibility
        self.head_disp = nn.Linear(embed_dim, 3)  # Displacement/motion
        self.head_normal = nn.Linear(embed_dim, 3)  # Surface normal
        self.head_conf = nn.Linear(embed_dim, 1)  # Confidence

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def build_query(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor
    ) -> torch.Tensor:
        """Build query embeddings from components.

        Args:
            frames: (B, T, C, H, W) video frames for patch extraction
            coords: (B, N, 2) normalized (u, v) coordinates
            t_src: (B, N) source timestep indices
            t_tgt: (B, N) target timestep indices
            t_cam: (B, N) camera reference timestep indices

        Returns:
            query: (B, N, embed_dim)
        """
        B, N = coords.shape[:2]

        # Fourier embedding of coordinates
        coord_emb = self.fourier_embed(coords)  # (B, N, embed_dim)

        # Timestep embeddings
        src_emb, tgt_emb, cam_emb = self.timestep_embed(t_src, t_tgt, t_cam)

        # Local RGB patch embedding
        # Reshape frames for patch extraction: (B, T, C, H, W)
        if frames.dim() == 5 and frames.shape[-1] == 3:
            frames = frames.permute(0, 1, 4, 2, 3)  # (B, T, H, W, C) -> (B, T, C, H, W)

        patch_emb = self.patch_embed(frames, coords, t_src)  # (B, N, embed_dim)

        # Combine all embeddings
        query = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb

        # Add learnable query token
        query = query + self.query_token.expand(B, N, -1)

        return query

    def forward(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: (B, N_enc, embed_dim) Global Scene Representation
            frames: (B, T, C, H, W) video frames
            coords: (B, N_q, 2) normalized query coordinates
            t_src: (B, N_q) source timesteps
            t_tgt: (B, N_q) target timesteps
            t_cam: (B, N_q) camera reference timesteps

        Returns:
            Dictionary with predictions:
                - pos_3d: (B, N_q, 3) 3D positions
                - pos_2d: (B, N_q, 2) 2D positions
                - visibility: (B, N_q, 1) visibility logits
                - displacement: (B, N_q, 3) motion displacement
                - normal: (B, N_q, 3) surface normals
                - confidence: (B, N_q, 1) confidence scores
        """
        # Build query embeddings
        query = self.build_query(frames, coords, t_src, t_tgt, t_cam)

        # Apply decoder blocks (cross-attention into encoder features)
        for block in self.blocks:
            query = block(query, encoder_features)

        query = self.norm(query)

        # Predict outputs
        pos_3d = self.head_3d(query)
        pos_2d = self.head_2d(query)
        visibility = self.head_vis(query)
        displacement = self.head_disp(query)
        normal = self.head_normal(query)
        normal = F.normalize(normal, dim=-1)  # Normalize to unit vectors
        confidence = torch.sigmoid(self.head_conf(query))

        return {
            'pos_3d': pos_3d,
            'pos_2d': pos_2d,
            'visibility': visibility,
            'displacement': displacement,
            'normal': normal,
            'confidence': confidence
        }

    def decode_3d_position(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get only 3D positions.

        Returns:
            pos_3d: (B, N_q, 3) 3D positions
        """
        outputs = self.forward(encoder_features, frames, coords, t_src, t_tgt, t_cam)
        return outputs['pos_3d']

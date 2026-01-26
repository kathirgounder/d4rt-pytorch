"""D4RT Encoder: Video encoder using timm ViT backbone with local/global attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import partial

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer, Block
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Using custom implementation.")

# VideoMAE support via HuggingFace transformers
try:
    from transformers import VideoMAEModel, VideoMAEConfig
    VIDEOMAE_AVAILABLE = True
except ImportError:
    VIDEOMAE_AVAILABLE = False
    print("Warning: transformers not available. VideoMAE encoder disabled.")


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video.

    Converts video to sequence of patch embeddings with spatio-temporal patches.
    """

    def __init__(
        self,
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.temporal_size = temporal_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches_t = temporal_size // patch_size[0]
        self.num_patches_h = img_size // patch_size[1]
        self.num_patches_w = img_size // patch_size[2]
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor

        Returns:
            patches: (B, N, embed_dim) where N = num_patches
        """
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class EfficientAttention(nn.Module):
    """Efficient multi-head attention using PyTorch's scaled_dot_product_attention.

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)

        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's efficient attention (FlashAttention when available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocalAttention(nn.Module):
    """Frame-wise local self-attention using efficient attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.attention = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        patches_per_frame: int
    ) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B * num_frames, patches_per_frame, C)
        x = self.attention(x)
        x = x.view(B, N, C)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

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


class EncoderBlock(nn.Module):
    """Encoder block with either local or global attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attention_type: str = 'global'
    ):
        super().__init__()
        self.attention_type = attention_type

        self.norm1 = nn.LayerNorm(dim)
        if attention_type == 'local':
            self.attn = LocalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        else:
            self.attn = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: Optional[int] = None,
        patches_per_frame: Optional[int] = None
    ) -> torch.Tensor:
        if self.attention_type == 'local':
            x = x + self.attn(self.norm1(x), num_frames, patches_per_frame)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class D4RTEncoder(nn.Module):
    """D4RT Video Encoder.

    Vision Transformer with interleaved local (frame-wise) and global self-attention.
    Produces the Global Scene Representation F.

    Can optionally use timm pretrained weights for initialization.
    """

    def __init__(
        self,
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        local_global_pattern: str = 'interleaved',
        use_timm_init: bool = False,
        timm_model: str = 'vit_base_patch16_224'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_timm_init = use_timm_init

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size, temporal_size, patch_size, in_channels, embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.num_frames = self.patch_embed.num_patches_t
        self.patches_per_frame = self.patch_embed.num_patches_h * self.patch_embed.num_patches_w

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Aspect ratio token
        self.aspect_ratio_embed = nn.Linear(2, embed_dim)

        # Encoder blocks with interleaved local and global attention
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if local_global_pattern == 'interleaved':
                attn_type = 'local' if i % 2 == 0 else 'global'
            elif local_global_pattern == 'local_first':
                attn_type = 'local' if i < depth // 2 else 'global'
            else:
                attn_type = 'global' if i < depth // 2 else 'local'

            self.blocks.append(EncoderBlock(
                embed_dim, num_heads, mlp_ratio, qkv_bias,
                drop_rate, attn_drop_rate, attn_type
            ))

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        # Optionally load timm weights
        if use_timm_init and TIMM_AVAILABLE:
            self._load_timm_weights(timm_model)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _load_timm_weights(self, model_name: str):
        """Load pretrained weights from timm ViT model."""
        if not TIMM_AVAILABLE:
            print("timm not available, skipping pretrained weight loading")
            return

        print(f"Loading pretrained weights from timm model: {model_name}")
        timm_model = timm.create_model(model_name, pretrained=True)

        # Copy block weights (only for matching layers)
        timm_blocks = list(timm_model.blocks.children())
        for i, (our_block, timm_block) in enumerate(zip(self.blocks, timm_blocks)):
            if i >= len(timm_blocks):
                break

            # Copy attention weights
            if hasattr(our_block.attn, 'qkv'):
                our_block.attn.qkv.weight.data.copy_(timm_block.attn.qkv.weight.data)
                our_block.attn.qkv.bias.data.copy_(timm_block.attn.qkv.bias.data)
                our_block.attn.proj.weight.data.copy_(timm_block.attn.proj.weight.data)
                our_block.attn.proj.bias.data.copy_(timm_block.attn.proj.bias.data)
            elif hasattr(our_block.attn, 'attention'):  # LocalAttention wrapper
                our_block.attn.attention.qkv.weight.data.copy_(timm_block.attn.qkv.weight.data)
                our_block.attn.attention.qkv.bias.data.copy_(timm_block.attn.qkv.bias.data)
                our_block.attn.attention.proj.weight.data.copy_(timm_block.attn.proj.weight.data)
                our_block.attn.attention.proj.bias.data.copy_(timm_block.attn.proj.bias.data)

            # Copy MLP weights
            our_block.mlp.fc1.weight.data.copy_(timm_block.mlp.fc1.weight.data)
            our_block.mlp.fc1.bias.data.copy_(timm_block.mlp.fc1.bias.data)
            our_block.mlp.fc2.weight.data.copy_(timm_block.mlp.fc2.weight.data)
            our_block.mlp.fc2.bias.data.copy_(timm_block.mlp.fc2.bias.data)

            # Copy LayerNorm weights
            our_block.norm1.weight.data.copy_(timm_block.norm1.weight.data)
            our_block.norm1.bias.data.copy_(timm_block.norm1.bias.data)
            our_block.norm2.weight.data.copy_(timm_block.norm2.weight.data)
            our_block.norm2.bias.data.copy_(timm_block.norm2.bias.data)

        # Copy final norm
        self.norm.weight.data.copy_(timm_model.norm.weight.data)
        self.norm.bias.data.copy_(timm_model.norm.bias.data)

        print(f"Loaded pretrained weights for {min(len(self.blocks), len(timm_blocks))} blocks")

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W) or (B, T, H, W, C) video tensor
            aspect_ratio: (B, 2) original aspect ratio [width, height]

        Returns:
            features: (B, N, embed_dim) Global Scene Representation F
        """
        # Handle different input formats
        if video.dim() == 5 and video.shape[-1] == 3:
            video = video.permute(0, 4, 1, 2, 3)

        B = video.shape[0]

        # Patch embedding
        x = self.patch_embed(video)

        # Add positional embedding
        x = x + self.pos_embed

        # Add aspect ratio token if provided
        if aspect_ratio is not None:
            ar_token = self.aspect_ratio_embed(aspect_ratio)
            ar_token = ar_token.unsqueeze(1)
            x = torch.cat([ar_token, x], dim=1)

        # Apply encoder blocks
        for block in self.blocks:
            if aspect_ratio is not None:
                ar_token = x[:, :1]
                patches = x[:, 1:]
                patches = block(patches, self.num_frames, self.patches_per_frame)
                x = torch.cat([ar_token, patches], dim=1)
            else:
                x = block(x, self.num_frames, self.patches_per_frame)

        x = self.norm(x)

        if aspect_ratio is not None:
            x = x[:, 1:]

        return x


class TimmVideoEncoder(nn.Module):
    """Video encoder that wraps a timm ViT model.

    Processes video frame-by-frame with a timm ViT, then applies
    temporal attention to aggregate features across frames.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        temporal_size: int = 48,
        temporal_stride: int = 2,
        freeze_backbone: bool = False
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TimmVideoEncoder")

        # Create timm model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        self.embed_dim = self.backbone.embed_dim

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal aggregation
        self.temporal_stride = temporal_stride
        self.num_frames = temporal_size // temporal_stride

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frames, self.embed_dim)
        )

        # Temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            EncoderBlock(
                self.embed_dim, num_heads=12, mlp_ratio=4.0,
                attention_type='global'
            )
            for _ in range(4)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W) or (B, T, H, W, C) video tensor

        Returns:
            features: (B, T'*N, embed_dim) where T' = T // temporal_stride
        """
        if video.dim() == 5 and video.shape[-1] == 3:
            video = video.permute(0, 4, 1, 2, 3)

        B, C, T, H, W = video.shape

        # Subsample temporally
        frame_indices = list(range(0, T, self.temporal_stride))[:self.num_frames]
        video = video[:, :, frame_indices]
        T_sub = len(frame_indices)

        # Process each frame with backbone
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T_sub, C, H, W)

        # Get features from timm backbone
        features = self.backbone.forward_features(video_flat)  # (B*T, N, C)
        N = features.shape[1]

        # Reshape back: (B*T, N, C) -> (B, T, N, C)
        features = features.view(B, T_sub, N, self.embed_dim)

        # Add temporal position embedding
        features = features + self.temporal_pos_embed[:, :T_sub].unsqueeze(2)

        # Reshape for temporal attention: (B, T*N, C)
        features = features.view(B, T_sub * N, self.embed_dim)

        # Apply temporal attention blocks
        for block in self.temporal_blocks:
            features = block(features)

        features = self.norm(features)

        return features


class VideoMAEEncoder(nn.Module):
    """Video encoder using pretrained VideoMAE from HuggingFace.

    VideoMAE is specifically pretrained on video data using masked autoencoding,
    making it well-suited for video understanding tasks.

    Supported pretrained models:
        - MCG-NJU/videomae-base (ViT-B, 86M params)
        - MCG-NJU/videomae-large (ViT-L, 305M params)
        - MCG-NJU/videomae-huge (ViT-H, 633M params)
        - MCG-NJU/videomae-base-finetuned-kinetics (finetuned on K400)
        - MCG-NJU/videomae-large-finetuned-kinetics (finetuned on K400)
    """

    def __init__(
        self,
        model_name: str = 'MCG-NJU/videomae-base',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_frames: int = 16,
        use_mean_pooling: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            num_frames: Number of frames VideoMAE expects (default 16)
            use_mean_pooling: Whether to mean pool over time dimension
        """
        super().__init__()

        if not VIDEOMAE_AVAILABLE:
            raise ImportError(
                "transformers library required for VideoMAE. "
                "Install with: pip install transformers"
            )

        print(f"Loading VideoMAE encoder: {model_name}")

        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(model_name)
        else:
            config = VideoMAEConfig.from_pretrained(model_name)
            self.backbone = VideoMAEModel(config)

        self.embed_dim = self.backbone.config.hidden_size
        self.num_frames = num_frames
        self.use_mean_pooling = use_mean_pooling
        self.patch_size = (
            self.backbone.config.tubelet_size,
            self.backbone.config.patch_size,
            self.backbone.config.patch_size
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("VideoMAE backbone frozen")

        # Aspect ratio embedding
        self.aspect_ratio_embed = nn.Linear(2, self.embed_dim)

        # Additional projection for interleaved local/global attention pattern
        # VideoMAE already has global attention, we add local attention layers
        self.local_attention_layers = nn.ModuleList([
            EncoderBlock(
                self.embed_dim,
                num_heads=self.backbone.config.num_attention_heads,
                mlp_ratio=4.0,
                attention_type='local'
            )
            for _ in range(4)  # Add 4 local attention layers
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W) or (B, T, H, W, C) video tensor
            aspect_ratio: (B, 2) original aspect ratio

        Returns:
            features: (B, N, embed_dim) Global Scene Representation F
        """
        # Handle different input formats
        if video.dim() == 5 and video.shape[-1] == 3:
            # (B, T, H, W, C) -> (B, C, T, H, W)
            video = video.permute(0, 4, 1, 2, 3)

        B, C, T, H, W = video.shape

        # VideoMAE expects (B, T, C, H, W) format
        video_mae_input = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Subsample or pad frames to match VideoMAE's expected num_frames
        if T != self.num_frames:
            if T > self.num_frames:
                # Subsample frames uniformly
                indices = torch.linspace(0, T - 1, self.num_frames).long()
                video_mae_input = video_mae_input[:, indices]
            else:
                # Pad by repeating last frame
                pad_frames = self.num_frames - T
                last_frame = video_mae_input[:, -1:].expand(-1, pad_frames, -1, -1, -1)
                video_mae_input = torch.cat([video_mae_input, last_frame], dim=1)

        # Get VideoMAE features
        outputs = self.backbone(video_mae_input, return_dict=True)
        features = outputs.last_hidden_state  # (B, N, embed_dim)

        # Calculate spatial dimensions for local attention
        num_patches_t = self.num_frames // self.patch_size[0]
        num_patches_h = H // self.patch_size[1]
        num_patches_w = W // self.patch_size[2]
        patches_per_frame = num_patches_h * num_patches_w

        # Apply interleaved local attention layers
        for local_layer in self.local_attention_layers:
            features = local_layer(features, num_patches_t, patches_per_frame)

        # Add aspect ratio if provided
        if aspect_ratio is not None:
            ar_embed = self.aspect_ratio_embed(aspect_ratio)  # (B, embed_dim)
            ar_token = ar_embed.unsqueeze(1)  # (B, 1, embed_dim)
            features = torch.cat([ar_token, features], dim=1)

        features = self.norm(features)

        # Remove aspect ratio token from output
        if aspect_ratio is not None:
            features = features[:, 1:]

        return features


def create_encoder(
    variant: str = 'base',
    use_timm: bool = False,
    use_videomae: bool = True,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """Create encoder with predefined configurations.

    Args:
        variant: One of 'base', 'large', 'huge', 'giant'
        use_timm: Whether to use timm-based encoder
        use_videomae: Whether to use VideoMAE encoder (recommended)
        pretrained: Whether to load pretrained weights

    Returns:
        Configured encoder
    """
    configs = {
        'base': dict(embed_dim=768, depth=12, num_heads=12),
        'large': dict(embed_dim=1024, depth=24, num_heads=16),
        'huge': dict(embed_dim=1280, depth=32, num_heads=16),
        'giant': dict(embed_dim=1408, depth=40, num_heads=16),
    }

    timm_models = {
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224',
        'huge': 'vit_huge_patch14_224',
        'giant': 'vit_giant_patch14_224',
    }

    videomae_models = {
        'base': 'MCG-NJU/videomae-base',
        'large': 'MCG-NJU/videomae-large',
        'huge': 'MCG-NJU/videomae-huge',
        'giant': 'MCG-NJU/videomae-huge',  # No giant, use huge
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    # Priority: VideoMAE > timm > custom
    if use_videomae and VIDEOMAE_AVAILABLE and pretrained:
        print(f"Using VideoMAE encoder: {videomae_models[variant]}")
        return VideoMAEEncoder(
            model_name=videomae_models[variant],
            pretrained=True,
            **{k: v for k, v in kwargs.items() if k in ['freeze_backbone', 'num_frames', 'use_mean_pooling']}
        )
    elif use_timm and TIMM_AVAILABLE and pretrained:
        # Use D4RTEncoder with timm weight initialization
        config = configs[variant]
        config.update(kwargs)
        config['use_timm_init'] = True
        config['timm_model'] = timm_models.get(variant, 'vit_base_patch16_224')
        return D4RTEncoder(**config)
    else:
        # Use custom encoder without pretrained weights
        config = configs[variant]
        config.update(kwargs)
        return D4RTEncoder(**config)

"""Loss functions for D4RT training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def normalize_points(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize points by their mean depth.

    Args:
        points: (B, N, 3) point positions
        eps: Small value for numerical stability

    Returns:
        normalized_points: (B, N, 3) normalized points
    """
    # Mean depth (Z coordinate)
    mean_depth = points[..., 2].mean(dim=-1, keepdim=True)  # (B, 1)
    mean_depth = mean_depth.unsqueeze(-1)  # (B, 1, 1)

    # Clamp to prevent division by near-zero (can happen early in training
    # when predicted Z values average near zero with diverse queries)
    mean_depth = mean_depth.abs().clamp(min=0.1)

    # Normalize by mean depth
    return points / mean_depth


def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply log transform to dampen influence of far points.

    Transform: sign(x) * log(1 + |x|)

    Args:
        x: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def compute_3d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    use_log_transform: bool = True
) -> torch.Tensor:
    """Compute L1 loss on 3D positions.

    Args:
        pred: (B, N, 3) predicted 3D positions
        target: (B, N, 3) target 3D positions
        mask: (B, N) optional validity mask
        normalize: Whether to normalize by mean depth
        use_log_transform: Whether to apply log transform

    Returns:
        loss: Scalar loss value
    """
    if normalize:
        pred = normalize_points(pred)
        target = normalize_points(target)

    if use_log_transform:
        pred = log_transform(pred)
        target = log_transform(target)

    # L1 loss
    loss = torch.abs(pred - target)

    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


def compute_2d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on 2D reprojection.

    Args:
        pred: (B, N, 2) predicted 2D positions
        target: (B, N, 2) target 2D positions
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


def compute_visibility_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute binary cross-entropy loss for visibility prediction.

    Args:
        pred: (B, N, 1) predicted visibility logits
        target: (B, N) target visibility (0 or 1)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    pred = pred.squeeze(-1)  # (B, N)

    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


def compute_displacement_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on motion displacement.

    Args:
        pred: (B, N, 3) predicted displacement
        target: (B, N, 3) target displacement
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


def compute_normal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute cosine similarity loss for surface normals.

    Args:
        pred: (B, N, 3) predicted normals (should be normalized)
        target: (B, N, 3) target normals (should be normalized)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value (1 - cosine_similarity)
    """
    # Normalize to unit vectors
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity
    cos_sim = (pred * target).sum(dim=-1)  # (B, N)

    # Loss is 1 - cosine_similarity
    loss = 1.0 - cos_sim

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


def compute_confidence_loss(
    confidence: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute confidence penalty loss: -log(confidence).

    This encourages the model to be confident in its predictions.

    Args:
        confidence: (B, N, 1) confidence scores in (0, 1)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    confidence = confidence.squeeze(-1)  # (B, N)

    # Clamp for numerical stability
    confidence = torch.clamp(confidence, min=1e-6, max=1.0 - 1e-6)

    loss = -torch.log(confidence)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)

    return loss.mean()


class D4RTLoss(nn.Module):
    """Combined loss function for D4RT training.

    Computes weighted sum of:
    - L_3D: L1 loss on normalized, log-transformed 3D positions (weighted by confidence)
    - L_2D: L1 loss on 2D reprojection
    - L_vis: BCE loss for visibility
    - L_disp: L1 loss on motion displacement
    - L_normal: Cosine similarity loss for surface normals
    - L_conf: Confidence penalty -log(c)
    """

    def __init__(
        self,
        lambda_3d: float = 1.0,
        lambda_2d: float = 0.1,
        lambda_vis: float = 0.1,
        lambda_disp: float = 0.1,
        lambda_normal: float = 0.5,
        lambda_conf: float = 0.2
    ):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_2d = lambda_2d
        self.lambda_vis = lambda_vis
        self.lambda_disp = lambda_disp
        self.lambda_normal = lambda_normal
        self.lambda_conf = lambda_conf

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            predictions: Dictionary with model outputs:
                - pos_3d: (B, N, 3) predicted 3D positions
                - pos_2d: (B, N, 2) predicted 2D positions
                - visibility: (B, N, 1) predicted visibility logits
                - displacement: (B, N, 3) predicted displacement
                - normal: (B, N, 3) predicted normals
                - confidence: (B, N, 1) predicted confidence

            targets: Dictionary with ground truth:
                - pos_3d: (B, N, 3) target 3D positions
                - pos_2d: (B, N, 2) target 2D positions (optional)
                - visibility: (B, N) target visibility
                - displacement: (B, N, 3) target displacement (optional)
                - normal: (B, N, 3) target normals (optional)
                - mask_3d: (B, N) validity mask for 3D loss
                - mask_2d: (B, N) validity mask for 2D loss (optional)
                - mask_vis: (B, N) validity mask for visibility (optional)
                - mask_disp: (B, N) validity mask for displacement (optional)
                - mask_normal: (B, N) validity mask for normals (optional)

        Returns:
            Dictionary with:
                - loss: Total weighted loss
                - loss_3d: 3D position loss
                - loss_2d: 2D position loss
                - loss_vis: Visibility loss
                - loss_disp: Displacement loss
                - loss_normal: Normal loss
                - loss_conf: Confidence penalty
        """
        losses = {}

        # Get masks (default to all valid if not provided)
        mask_3d = targets.get('mask_3d')
        mask_2d = targets.get('mask_2d', mask_3d)
        mask_vis = targets.get('mask_vis', mask_3d)
        mask_disp = targets.get('mask_disp')
        mask_normal = targets.get('mask_normal')

        # 3D position loss (weighted by confidence)
        confidence = predictions['confidence'].squeeze(-1)  # (B, N)

        loss_3d_unweighted = compute_3d_loss(
            predictions['pos_3d'],
            targets['pos_3d'],
            mask=None,  # We apply confidence weighting instead
            normalize=True,
            use_log_transform=True
        )

        # Apply confidence weighting
        if mask_3d is not None:
            # Per-point loss with confidence weighting
            pred_norm = normalize_points(predictions['pos_3d'])
            target_norm = normalize_points(targets['pos_3d'])
            pred_log = log_transform(pred_norm)
            target_log = log_transform(target_norm)

            point_loss = torch.abs(pred_log - target_log).mean(dim=-1)  # (B, N)
            weighted_loss = confidence * point_loss

            losses['loss_3d'] = (weighted_loss * mask_3d).sum() / (mask_3d.sum() + 1e-6)
        else:
            losses['loss_3d'] = loss_3d_unweighted

        # 2D position loss (if target provided)
        if 'pos_2d' in targets:
            losses['loss_2d'] = compute_2d_loss(
                predictions['pos_2d'],
                targets['pos_2d'],
                mask=mask_2d
            )
        else:
            losses['loss_2d'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Visibility loss
        if 'visibility' in targets:
            losses['loss_vis'] = compute_visibility_loss(
                predictions['visibility'],
                targets['visibility'],
                mask=mask_vis
            )
        else:
            losses['loss_vis'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Displacement loss (if target provided)
        if 'displacement' in targets and mask_disp is not None:
            losses['loss_disp'] = compute_displacement_loss(
                predictions['displacement'],
                targets['displacement'],
                mask=mask_disp
            )
        else:
            losses['loss_disp'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Normal loss (if target provided)
        if 'normal' in targets and mask_normal is not None:
            losses['loss_normal'] = compute_normal_loss(
                predictions['normal'],
                targets['normal'],
                mask=mask_normal
            )
        else:
            losses['loss_normal'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Confidence penalty
        losses['loss_conf'] = compute_confidence_loss(
            predictions['confidence'],
            mask=mask_3d
        )

        # Total loss
        losses['loss'] = (
            self.lambda_3d * losses['loss_3d'] +
            self.lambda_2d * losses['loss_2d'] +
            self.lambda_vis * losses['loss_vis'] +
            self.lambda_disp * losses['loss_disp'] +
            self.lambda_normal * losses['loss_normal'] +
            self.lambda_conf * losses['loss_conf']
        )

        return losses


class DepthLoss(nn.Module):
    """Loss function for depth estimation evaluation."""

    def __init__(self, scale_invariant: bool = True):
        super().__init__()
        self.scale_invariant = scale_invariant

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Compute depth metrics.

        Args:
            pred: (B, H, W) predicted depth
            target: (B, H, W) target depth
            mask: (B, H, W) validity mask

        Returns:
            Dictionary with metrics
        """
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)

        # Flatten
        pred_flat = pred[mask]
        target_flat = target[mask]

        if self.scale_invariant:
            # Scale alignment
            scale = (target_flat / (pred_flat + 1e-6)).median()
            pred_flat = pred_flat * scale

        # AbsRel
        abs_rel = torch.abs(pred_flat - target_flat) / (target_flat + 1e-6)
        abs_rel = abs_rel.mean()

        # RMSE
        rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean())

        # Log RMSE
        log_rmse = torch.sqrt(((torch.log(pred_flat + 1e-6) - torch.log(target_flat + 1e-6)) ** 2).mean())

        return {
            'abs_rel': abs_rel,
            'rmse': rmse,
            'log_rmse': log_rmse
        }

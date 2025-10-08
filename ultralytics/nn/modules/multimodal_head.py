# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Enhanced multimodal head modules for brain tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Proto
from .conv import Conv
from .head import Segment


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention mechanism for CT-MRI feature fusion.

    This module enables information exchange between CT and MRI feature representations at the same spatial resolution,
    allowing the model to leverage complementary information from both modalities.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-modal attention.

        Args:
            source: Source modality features [B, C, H, W]
            target: Target modality features [B, C, H, W]

        Returns:
            Enhanced source features with target information
        """
        B, C, H, W = source.shape

        # Reshape for attention computation
        source_flat = source.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        target_flat = target.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Multi-head attention
        residual = source_flat
        source_flat = self.norm1(source_flat)

        # Compute Q from source, K and V from target
        Q = self.q_proj(source_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(target_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(target_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, C)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        enhanced = residual + attn_output

        # Feed-forward network with residual
        residual = enhanced
        enhanced = self.norm2(enhanced)
        enhanced = residual + self.ffn(enhanced)

        # Reshape back to spatial format
        enhanced = enhanced.permute(0, 2, 1).view(B, C, H, W)

        return enhanced


class MultiModalFusionModule(nn.Module):
    """
    Multi-modal fusion module that combines CT and MRI features.

    Uses cross-modal attention for deep feature interaction followed by feature aggregation and refinement.
    """

    def __init__(self, channels: int, fusion_type: str = "attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.channels = channels

        if fusion_type == "attention":
            # Cross-modal attention layers
            self.ct_to_mri_attention = CrossModalAttention(channels)
            self.mri_to_ct_attention = CrossModalAttention(channels)

            # Feature aggregation
            self.fusion_conv = nn.Sequential(
                Conv(channels * 2, channels, 1),
                Conv(channels, channels, 3),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        elif fusion_type == "concat":
            # Simple concatenation + convolution
            self.fusion_conv = nn.Sequential(Conv(channels * 2, channels, 1), Conv(channels, channels, 3))

        elif fusion_type == "add":
            # Element-wise addition with weighting
            self.ct_weight = nn.Parameter(torch.ones(1))
            self.mri_weight = nn.Parameter(torch.ones(1))
            self.fusion_conv = Conv(channels, channels, 3)

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, ct_features: torch.Tensor, mri_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse CT and MRI features.

        Args:
            ct_features: CT feature maps [B, C, H, W]
            mri_features: MRI feature maps [B, C, H, W]

        Returns:
            Fused feature maps [B, C, H, W]
        """
        if self.fusion_type == "attention":
            # Cross-modal attention
            ct_enhanced = self.ct_to_mri_attention(ct_features, mri_features)
            mri_enhanced = self.mri_to_ct_attention(mri_features, ct_features)

            # Concatenate and fuse
            fused = torch.cat([ct_enhanced, mri_enhanced], dim=1)
            fused = self.fusion_conv(fused)

        elif self.fusion_type == "concat":
            # Simple concatenation
            fused = torch.cat([ct_features, mri_features], dim=1)
            fused = self.fusion_conv(fused)

        elif self.fusion_type == "add":
            # Weighted addition
            fused = self.ct_weight * ct_features + self.mri_weight * mri_features
            fused = self.fusion_conv(fused)

        return fused


class MultiModalSegment(Segment):
    """
    Enhanced YOLO Segment head for multimodal brain tumor segmentation.

    Extends the standard YOLO segmentation head to handle CT+MRI inputs with cross-modal attention and medical-specific
    optimizations.
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=(), fusion_type="attention", uncertainty=False):
        """
        Initialize multimodal segmentation head.

        Args:
            nc (int): Number of classes (4 for brain tumor: background, core, edema, enhancing)
            nm (int): Number of masks
            npr (int): Number of protos
            ch (tuple): Input channels from backbone
            fusion_type (str): Type of fusion ('attention', 'concat', 'add')
            uncertainty (bool): Whether to include uncertainty estimation
        """
        super().__init__(nc, nm, npr, ch)

        self.fusion_type = fusion_type
        self.uncertainty = uncertainty

        # Multi-modal fusion modules for each scale
        self.fusion_modules = nn.ModuleList([MultiModalFusionModule(c, fusion_type) for c in ch])

        # Uncertainty estimation head (if enabled)
        if uncertainty:
            self.uncertainty_convs = nn.ModuleList([nn.Sequential(Conv(c, c // 2, 3), Conv(c // 2, nc, 1)) for c in ch])

        # Medical-specific prototype generation
        self.medical_proto = Proto(ch[0], npr, nm)

        # Attention weights for combining multi-scale features
        self.scale_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(len(ch), len(ch), 1), nn.Sigmoid())

    def forward(self, ct_features: list[torch.Tensor], mri_features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for multimodal segmentation.

        Args:
            ct_features: List of CT feature maps from backbone
            mri_features: List of MRI feature maps from backbone

        Returns:
            Dictionary containing segmentation outputs
        """
        assert len(ct_features) == len(mri_features), "CT and MRI must have same number of feature levels"

        # Fuse features at each scale
        fused_features = []
        for i, (ct_feat, mri_feat) in enumerate(zip(ct_features, mri_features)):
            fused_feat = self.fusion_modules[i](ct_feat, mri_feat)
            fused_features.append(fused_feat)

        # Standard YOLO segmentation forward pass with fused features
        p = []
        for i, x in enumerate(fused_features):
            p.append(torch.cat((self.cv2[i](x), self.cv3[i](x), self.cv4[i](x)), 1))

        # Generate prototypes using the largest feature map
        proto = self.medical_proto(fused_features[0])

        results = {"predictions": p, "prototypes": proto}

        # Add uncertainty estimation if enabled
        if self.uncertainty:
            uncertainty_maps = []
            for i, x in enumerate(fused_features):
                uncertainty_map = self.uncertainty_convs[i](x)
                uncertainty_maps.append(uncertainty_map)
            results["uncertainty"] = uncertainty_maps

        return results

    def get_attention_maps(
        self, ct_features: list[torch.Tensor], mri_features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Extract attention maps for visualization and analysis.

        Args:
            ct_features: List of CT feature maps
            mri_features: List of MRI feature maps

        Returns:
            List of attention maps
        """
        attention_maps = []

        for i, (ct_feat, mri_feat) in enumerate(zip(ct_features, mri_features)):
            if self.fusion_type == "attention":
                fusion_module = self.fusion_modules[i]
                if hasattr(fusion_module, "ct_to_mri_attention"):
                    # Get attention weights from cross-modal attention
                    B, C, H, W = ct_feat.shape
                    ct_flat = ct_feat.view(B, C, H * W).permute(0, 2, 1)
                    mri_flat = mri_feat.view(B, C, H * W).permute(0, 2, 1)

                    with torch.no_grad():
                        attention_module = fusion_module.ct_to_mri_attention
                        Q = attention_module.q_proj(attention_module.norm1(ct_flat))
                        K = attention_module.k_proj(attention_module.norm1(mri_flat))

                        Q = Q.view(B, H * W, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)
                        K = K.view(B, H * W, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)

                        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * attention_module.scale
                        attn_weights = F.softmax(attn_weights, dim=-1)

                        # Average across heads and reshape
                        attn_map = attn_weights.mean(dim=1)  # [B, HW, HW]
                        attn_map = attn_map.mean(dim=-1).view(B, 1, H, W)  # [B, 1, H, W]
                        attention_maps.append(attn_map)

        return attention_maps


class BrainTumorLoss(nn.Module):
    """
    Specialized loss function for brain tumor segmentation.

    Combines multiple loss components optimized for medical image segmentation:
    - Dice loss for overlap maximization
    - Focal loss for hard example mining
    - Boundary loss for edge preservation
    - Uncertainty regularization
    """

    def __init__(self, num_classes: int = 4, class_weights: list[float] | None = None):
        super().__init__()
        self.num_classes = num_classes

        # Class weights for imbalanced data (tumor regions are typically small)
        if class_weights is None:
            # Default weights: less weight for background, more for tumor regions
            class_weights = [0.1, 1.0, 1.5, 2.0]  # background, core, edema, enhancing

        self.register_buffer("class_weights", torch.tensor(class_weights))

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Multi-class Dice loss."""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dice_scores = []
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target_one_hot[:, i]

            intersection = torch.sum(pred_i * target_i, dim=(1, 2))
            union = torch.sum(pred_i, dim=(1, 2)) + torch.sum(target_i, dim=(1, 2))

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice * self.class_weights[i])

        return 1.0 - torch.stack(dice_scores, dim=1).mean()

    def focal_loss(
        self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss for addressing class imbalance."""
        ce_loss = F.cross_entropy(pred, target, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Boundary loss for edge preservation."""
        pred_soft = F.softmax(pred, dim=1)

        # Compute gradients
        pred_grad_x = torch.abs(pred_soft[:, :, :, :-1] - pred_soft[:, :, :, 1:])
        pred_grad_y = torch.abs(pred_soft[:, :, :-1, :] - pred_soft[:, :, 1:, :])

        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        target_grad_x = torch.abs(target_one_hot[:, :, :, :-1] - target_one_hot[:, :, :, 1:])
        target_grad_y = torch.abs(target_one_hot[:, :, :-1, :] - target_one_hot[:, :, 1:, :])

        boundary_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        return boundary_loss

    def forward(self, predictions: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Model predictions
            targets: Ground truth masks

        Returns:
            Dictionary of loss components
        """
        pred = predictions["predictions"][0]  # Use first prediction scale

        # Resize prediction to match target if needed
        if pred.shape[-2:] != targets.shape[-2:]:
            pred = F.interpolate(pred, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        # Extract segmentation logits (assuming they're in the prediction)
        seg_pred = pred[:, : self.num_classes]  # First nc channels are segmentation

        # Compute loss components
        dice_loss = self.dice_loss(seg_pred, targets)
        focal_loss = self.focal_loss(seg_pred, targets)
        boundary_loss = self.boundary_loss(seg_pred, targets)

        # Combine losses
        total_loss = dice_loss + 0.5 * focal_loss + 0.1 * boundary_loss

        return {
            "total_loss": total_loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "boundary_loss": boundary_loss,
        }


# Export the new classes
__all__ = ["CrossModalAttention", "MultiModalFusionModule", "MultiModalSegment", "BrainTumorLoss"]

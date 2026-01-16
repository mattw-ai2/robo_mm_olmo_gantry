"""
Point Cloud Backbone for processing 3D point clouds.

Uses Point Transformer V3 for encoding, voxelization for spatial binning,
max pooling over voxels, and linear projection to LLM dimension.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo.config import BaseConfig

log = logging.getLogger(__name__)


@dataclass
class PointCloudBackboneConfig(BaseConfig):
    """Configuration for Point Cloud Backbone"""
    
    # Voxelization parameters
    voxel_size: float = 0.1  # Meters per voxel
    grid_range: float = 10.0  # Grid extent [-R, R] in meters
    
    # Point Transformer V3 parameters
    ptv3_channels: int = 512  # PTv3 output feature dimension
    ptv3_num_layers: int = 4  # Number of PTv3 layers
    ptv3_num_heads: int = 8  # Number of attention heads
    
    # Whether to use pretrained PTv3 weights
    use_pretrained_ptv3: bool = False
    ptv3_checkpoint: Optional[str] = None
    
    # Output projection
    dropout: float = 0.0
    
    def build(self, llm_dim: int, device=None):
        return PointCloudBackbone(self, llm_dim, device)


class VoxelPooling(nn.Module):
    """
    Voxelizes point cloud and performs max pooling over each voxel.
    
    Divides 3D space into a grid and pools point features within each cell.
    """
    
    def __init__(self, voxel_size: float, grid_range: float):
        super().__init__()
        self.voxel_size = voxel_size
        self.grid_range = grid_range
        
        # Compute grid dimensions
        self.grid_size = int(2 * grid_range / voxel_size)
        
    def forward(
        self, 
        points: torch.Tensor,  # [B, N, 3] - xyz coordinates
        features: torch.Tensor,  # [B, N, C] - point features
        mask: Optional[torch.Tensor] = None  # [B, N] - valid point mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: Point positions [B, N, 3]
            features: Point features [B, N, C]
            mask: Valid point mask [B, N], True for valid points
            
        Returns:
            voxel_features: Pooled features [B, num_voxels, C]
            voxel_mask: Valid voxel mask [B, num_voxels]
        """
        B, N, C = features.shape
        device = features.device
        
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # Compute voxel indices for each point
        # Shift coordinates to [0, 2*grid_range] then divide by voxel_size
        voxel_coords = ((points + self.grid_range) / self.voxel_size).long()
        
        # Clamp to valid range
        voxel_coords = voxel_coords.clamp(0, self.grid_size - 1)
        
        # Convert 3D coords to flat index: x + y*size + z*size^2
        voxel_idx = (
            voxel_coords[..., 0] + 
            voxel_coords[..., 1] * self.grid_size + 
            voxel_coords[..., 2] * self.grid_size * self.grid_size
        )  # [B, N]
        
        num_voxels = self.grid_size ** 3
        
        # Initialize output tensors
        voxel_features = torch.zeros(B, num_voxels, C, device=device, dtype=features.dtype)
        voxel_counts = torch.zeros(B, num_voxels, device=device, dtype=torch.long)
        
        # Max pool features into voxels
        # We use scatter_reduce for efficient batched aggregation
        for b in range(B):
            valid_mask = mask[b]
            valid_idx = voxel_idx[b][valid_mask]  # [num_valid]
            valid_features = features[b][valid_mask]  # [num_valid, C]
            
            if valid_idx.numel() > 0:
                # Max pooling via scatter
                voxel_features[b].scatter_reduce_(
                    0, 
                    valid_idx.unsqueeze(-1).expand(-1, C),
                    valid_features,
                    reduce="amax",
                    include_self=False
                )
                
                # Count points per voxel
                voxel_counts[b].scatter_add_(
                    0,
                    valid_idx,
                    torch.ones_like(valid_idx)
                )
        
        # Create voxel validity mask (voxels with at least one point)
        voxel_mask = voxel_counts > 0  # [B, num_voxels]
        
        # Only keep non-empty voxels (sparse representation)
        # Find max number of non-empty voxels across batch
        max_valid_voxels = voxel_mask.sum(dim=1).max().item()
        
        if max_valid_voxels == 0:
            # No valid voxels, return empty
            return torch.zeros(B, 1, C, device=device, dtype=features.dtype), \
                   torch.zeros(B, 1, dtype=torch.bool, device=device)
        
        # Gather non-empty voxel features
        output_features = torch.zeros(B, max_valid_voxels, C, device=device, dtype=features.dtype)
        output_mask = torch.zeros(B, max_valid_voxels, dtype=torch.bool, device=device)
        
        for b in range(B):
            valid_indices = voxel_mask[b].nonzero(as_tuple=True)[0]
            num_valid = valid_indices.numel()
            if num_valid > 0:
                output_features[b, :num_valid] = voxel_features[b, valid_indices]
                output_mask[b, :num_valid] = True
        
        return output_features, output_mask


class SimplePointTransformer(nn.Module):
    """
    Simplified Point Transformer implementation.
    
    This is a lightweight version that can be used when PTv3 is not available.
    For full PTv3, use the pointcept library integration.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input embedding
        self.input_proj = nn.Linear(in_channels, out_channels)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dim_feedforward=out_channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output norm
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(
        self,
        points: torch.Tensor,  # [B, N, 3]
        mask: Optional[torch.Tensor] = None  # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            points: Point positions [B, N, 3]
            mask: Valid point mask [B, N], True for valid points
            
        Returns:
            features: Point features [B, N, C]
        """
        # Project input points
        x = self.input_proj(points)  # [B, N, C]
        
        # Create attention mask if needed
        attn_mask = None
        if mask is not None:
            # Transformer expects mask where True means ignore
            attn_mask = ~mask
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Final norm
        x = self.norm(x)
        
        return x


class PointCloudBackbone(nn.Module):
    """
    Point Cloud Backbone for Molmo.
    
    Pipeline:
    1. Point Transformer V3 (or simplified version) encodes points
    2. Voxelize points into spatial grid
    3. Max pool features per voxel
    4. Linear projection to LLM dimension
    """
    
    def __init__(self, config: PointCloudBackboneConfig, llm_dim: int, device=None):
        super().__init__()
        self.config = config
        self.llm_dim = llm_dim
        
        # Try to load PTv3 from pointcept, fall back to simplified version
        self.ptv3 = None
        if config.use_pretrained_ptv3:
            try:
                self.ptv3 = self._load_ptv3(config, device)
                log.info("Loaded Point Transformer V3 from pointcept")
            except Exception as e:
                log.warning(f"Failed to load PTv3: {e}. Using simplified point transformer.")
        
        if self.ptv3 is None:
            self.ptv3 = SimplePointTransformer(
                in_channels=3,  # xyz
                out_channels=config.ptv3_channels,
                num_layers=config.ptv3_num_layers,
                num_heads=config.ptv3_num_heads,
                dropout=config.dropout
            )
            if device is not None:
                self.ptv3 = self.ptv3.to(device)
        
        # Voxel pooling
        self.voxel_pooling = VoxelPooling(
            voxel_size=config.voxel_size,
            grid_range=config.grid_range
        )
        
        # Linear projection to LLM dimension
        self.projector = nn.Linear(config.ptv3_channels, llm_dim, bias=False, device=device)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def _load_ptv3(self, config: PointCloudBackboneConfig, device=None):
        """
        Load Point Transformer V3 from pointcept library.
        """
        try:
            from pointcept.models import build_model
            from pointcept.utils.config import Config
            
            # Build PTv3 model
            ptv3_config = Config({
                'type': 'PTv3',
                'in_channels': 3,
                'out_channels': config.ptv3_channels,
                'num_layers': config.ptv3_num_layers,
            })
            ptv3 = build_model(ptv3_config)
            
            if config.ptv3_checkpoint is not None:
                state_dict = torch.load(config.ptv3_checkpoint, map_location='cpu')
                ptv3.load_state_dict(state_dict, strict=False)
                log.info(f"Loaded PTv3 checkpoint from {config.ptv3_checkpoint}")
            
            if device is not None:
                ptv3 = ptv3.to(device)
                
            return ptv3
            
        except ImportError:
            log.warning("pointcept not installed. Install with: pip install pointcept")
            raise
    
    def reset_parameters(self):
        """Initialize parameters."""
        if hasattr(self.ptv3, 'reset_parameters'):
            self.ptv3.reset_parameters()
        nn.init.xavier_uniform_(self.projector.weight)
        
    def forward(
        self,
        points: torch.Tensor,  # [B, N, 3] - xyz coordinates
        point_mask: Optional[torch.Tensor] = None,  # [B, N] - valid point mask
        pooled_idx: Optional[torch.Tensor] = None,  # Optional: indices for token positions
    ) -> torch.Tensor:
        """
        Process point cloud and return patch tokens for LLM.
        
        Args:
            points: Point positions [B, N, 3]
            point_mask: Valid point mask [B, N], True for valid points
            pooled_idx: Optional indices for mapping to token positions
            
        Returns:
            patch_tokens: Point cloud patch tokens [B, num_patches, llm_dim]
        """
        B = points.shape[0]
        
        # 1. Encode points with PTv3
        point_features = self.ptv3(points, point_mask)  # [B, N, C]
        
        # 2. Voxelize and max pool
        voxel_features, voxel_mask = self.voxel_pooling(
            points, point_features, point_mask
        )  # [B, num_voxels, C], [B, num_voxels]
        
        # 3. Project to LLM dimension
        patch_tokens = self.projector(voxel_features)  # [B, num_voxels, llm_dim]
        
        # 4. Apply dropout
        patch_tokens = self.dropout(patch_tokens)
        
        return patch_tokens, voxel_mask


# Vision Backbone

This document details the vision encoding components used in Molmo.

## Overview

The vision backbone is responsible for converting images into feature representations that can be processed by the language model. Molmo supports multiple vision encoder architectures with different trade-offs.

## Supported Vision Encoders

### OpenAI CLIP

**Architecture:**
- Vision Transformer (ViT) with patch size 14
- 24 layers, 1024 hidden dimension
- Trained on 400M image-text pairs
- Output: 577 tokens (24×24 patches + CLS token)

**Strengths:**
- Strong zero-shot transfer
- Good general-purpose features
- Wide adoption and compatibility

**Weaknesses:**
- Moderate resolution (224×224 pretrained)
- Trained with contrastive loss only

**Configuration:**
```python
vision_backbone = MolmoVisionBackboneConfig(
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    ...
)
```

### SigLIP

**Architecture:**
- Vision Transformer with sigmoid loss
- Similar architecture to CLIP
- Trained with improved loss function

**Strengths:**
- Better than CLIP on many benchmarks
- Efficient training
- Good multilingual capabilities

**Configuration:**
```python
vision_backbone = MolmoVisionBackboneConfig(
    image_model_type="siglip",
    ...
)
```

### DINOv2

**Architecture:**
- Self-supervised vision transformer
- No text supervision during pretraining
- Multiple model sizes (S/B/L/G)

**Strengths:**
- Excellent dense prediction features
- Strong spatial understanding
- Self-supervised (no text needed)

**Weaknesses:**
- No natural alignment with text
- May need more fine-tuning

**Configuration:**
```python
vision_backbone = MolmoVisionBackboneConfig(
    image_model_type="dino",
    ...
)
```

### MetaCLIP

**Architecture:**
- CLIP trained on curated data
- Improved data filtering
- Various model sizes

**Strengths:**
- Improved over original CLIP
- Better data quality
- Open and reproducible

## Vision Backbone Pipeline

### 1. Image Preprocessing

```python
class ImagePreprocessor:
    def __call__(
        self,
        images: np.ndarray,  # [H, W, C] or [B, H, W, C]
    ) -> torch.Tensor:
        """
        1. Resize to target resolution
        2. Normalize (mean/std)
        3. Convert to tensor
        """
```

**Normalization:**
- CLIP/SigLIP: ImageNet normalization
- DINO: Different normalization constants

### 2. Patch Extraction

Images are divided into patches:
- Patch size: 14×14 or 16×16
- Example: 336×336 image → 24×24 = 576 patches
- Plus CLS token → 577 tokens

### 3. Vision Transformer Encoding

```python
class VisionTransformer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # 1. Patch embedding
        x = self.patch_embedding(x)  # [B, N, D]
        # 2. Add position embeddings
        x = x + self.pos_embedding
        # 3. Transformer layers
        for layer in self.layers:
            x = layer(x)
        # 4. Layer norm
        x = self.ln_post(x)
        return x  # [B, N, D]
```

### 4. Feature Projection

Project vision features to LLM dimension:

```python
class ImageProjectorMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_vision]
        # Project to LLM dimension
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x  # [B, N, D_llm]
```

### 5. Optional Pooling

Reduce number of tokens for efficiency:

**Average Pooling:**
```python
# 2×2 pooling: 576 → 144 tokens
features = features.reshape(B, H//2, 2, W//2, 2, D)
features = features.mean(dim=(2, 4))
```

**Attention Pooling:**
```python
# Learnable queries aggregate features
queries = nn.Parameter(torch.randn(N_queries, D))
features = attention(queries, image_features)
```

**C-Abstractor:**
```python
# Cross-attention with learnable queries
# Reduces tokens while preserving important information
```

## MolmoVisionBackbone

Main vision processing module:

```python
class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        # Vision encoder (CLIP/SigLIP/DINO)
        self.image_vit = build_vit(config)
        # Projection to LLM space
        self.image_projector = ImageProjectorMLP(...)
        # Optional pooling
        self.image_pooling = ...
        
    def forward(
        self,
        images: torch.Tensor,  # [B, n_images, C, H, W]
        image_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_features: [B, n_images, n_tokens, D]
            image_masks: [B, n_images, n_tokens]
        """
```

## Configuration Options

### Resolution

Trade-off between quality and compute:

```python
# Low resolution: Fast but less detail
image_default_input_size = (224, 224)  # 256 tokens

# Medium resolution: Balanced
image_default_input_size = (336, 336)  # 576 tokens

# High resolution: Best quality
image_default_input_size = (768, 768)  # 3072 tokens
```

### Pooling Strategy

```python
# No pooling: All tokens
image_pooling_2d = None

# Average pooling: Reduce tokens
image_pooling_2d = "2"  # 2×2 pooling

# Attention pooling: Learnable reduction
image_pooling_2d = "attention"
```

### Projection Type

```python
# Linear projection: Simple and fast
image_project_type = "linear"

# MLP projection: More capacity
image_project_type = "mlp"

# MLP with residual: Best quality
image_project_type = "mlpx"
```

### Freezing

```python
# Freeze vision encoder
vit_load_path = "path/to/checkpoint"
vit_freeze = True

# Unfreeze after N steps
vit_unfreeze_step = 10000
```

## Multi-Resolution Support

For high-resolution understanding:

1. **Adaptive Resolution:**
   - Resize based on aspect ratio
   - Maintain aspect ratio when possible
   - Pad to fixed dimensions

2. **Tiling:**
   - Split large image into tiles
   - Process each tile independently
   - Combine features spatially

3. **Hierarchical:**
   - Low-res global view
   - High-res crops for details
   - Multi-scale fusion

## Position Encodings

### 2D Position Embeddings

```python
# Learnable 2D positions
pos_embed = nn.Parameter(torch.randn(H, W, D))

# Apply to patches
patch_features = patch_features + pos_embed[y, x]
```

### Rotary Position Embeddings (RoPE)

```python
# Apply rotary embeddings
# Better for variable resolutions
```

## Best Practices

### For Image Captioning
- Use CLIP or SigLIP
- Medium resolution (336×336)
- MLP projection
- No pooling or light pooling

### For Visual Question Answering
- SigLIP or CLIP
- High resolution if possible
- Full token preservation
- Task-specific fine-tuning

### For Dense Prediction (Pointing/Segmentation)
- DINOv2 or high-res CLIP
- No pooling
- Preserve spatial structure
- 2D position embeddings

### For Efficiency
- Lower resolution
- Aggressive pooling (2×2 or 4×4)
- Linear projection
- Freeze vision encoder

### For Video
- Lower per-frame resolution
- Shared vision encoder across frames
- Temporal position encoding
- Efficient attention

## Vision Encoder Comparison

| Encoder | Parameters | Best For | Speed | Quality |
|---------|-----------|----------|-------|---------|
| CLIP ViT-L/14 | 307M | General purpose | Medium | Good |
| SigLIP | ~300M | Balanced performance | Medium | Very Good |
| DINOv2-L | 304M | Dense tasks | Medium | Excellent |
| CLIP ViT-H/14 | 632M | Highest quality | Slow | Excellent |

## Implementation Notes

### Memory Considerations

- Image features are typically cached during generation
- High resolution requires significant memory
- Consider gradient checkpointing for training
- Use FP16/BF16 for efficiency

### Training Strategies

1. **Frozen Vision Encoder:**
   - Fast training
   - Good for data-limited scenarios
   - May underperform on novel domains

2. **Full Fine-tuning:**
   - Best quality
   - Requires more data
   - Longer training time

3. **Gradual Unfreezing:**
   - Start frozen, unfreeze later
   - Good balance
   - Stable training

### Inference Optimization

- **Batch Processing:** Process multiple images together
- **Feature Caching:** Cache image features for multiple generations
- **Precision:** FP16/BF16 inference
- **Pruning:** Remove unused tokens early

## Next Steps

- [LLM Components](llm_components.md) - Language model details
- [Data Pipeline](data_pipeline.md) - How images are loaded and processed
- [Training Guide](../guides/training_guide.md) - Training with different backbones


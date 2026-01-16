# Models API Reference

API documentation for Molmo model classes.

## ModelBase

Base class for all Molmo models.

```python
class ModelBase(torch.nn.Module):
    """Base class for all models."""
    
    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> OLMoOutput:
        """Forward pass for training."""
        
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ) -> OLMoGenerateOutput:
        """Generate text autoregressively."""
        
    def load_checkpoint(self, path: str, **kwargs):
        """Load model weights from checkpoint."""
        
    def save_checkpoint(self, path: str, **kwargs):
        """Save model weights to checkpoint."""
```

## Molmo

Main Molmo model for image-text understanding.

### MolmoConfig

```python
@dataclass
class MolmoConfig(BaseModelConfig):
    """Configuration for Molmo model."""
    
    llm: LlmConfig
    # Language model configuration
    
    vision_backbone: Optional[MolmoVisionBackboneConfig]
    # Vision encoder configuration
    
    data_formatter: DataFormatter
    # Task formatting configuration
    
    mm_preprocessor: MolmoPreprocessorConfig
    # Multimodal preprocessing configuration
    
    bi_directional_attn: Optional[str] = None
    # Bidirectional attention mode for image tokens
```

**Methods:**

```python
@classmethod
def load(cls, path: str, **kwargs) -> MolmoConfig:
    """Load configuration from YAML file."""
    
def build_tokenizer(self) -> Tokenizer:
    """Build tokenizer for this model."""
    
def build_preprocessor(
    self,
    for_inference: bool,
    is_training: bool = True,
    include_image: bool = False,
    max_seq_len: Optional[int] = None,
) -> Preprocessor:
    """Build preprocessor."""
    
def build_collator(
    self,
    sequence_length: int,
    pad_mode: str,
    include_metadata: bool = True
) -> MMCollator:
    """Build collator for batching."""
```

### Molmo Class

```python
class Molmo(ModelBase):
    """Molmo multimodal model."""
    
    def __init__(self, config: MolmoConfig):
        """Initialize model from config."""
        
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        images: Optional[torch.Tensor] = None,  # [B, N, C, H, W]
        image_masks: Optional[torch.Tensor] = None,  # [B, N, P]
        image_input_idx: Optional[torch.Tensor] = None,  # [B, N]
        labels: Optional[torch.Tensor] = None,  # [B, L]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]
        **kwargs
    ) -> OLMoOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            images: Input images
            image_masks: Valid patches in images
            image_input_idx: Positions of images in sequence
            labels: Target tokens for loss
            attention_mask: Attention mask
            
        Returns:
            OLMoOutput with logits and loss
        """
        
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> OLMoGenerateOutput:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs
            images: Input images
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            num_beams: Number of beams for beam search
            eos_token_id: End-of-sequence token ID(s)
            pad_token_id: Padding token ID
            
        Returns:
            OLMoGenerateOutput with generated tokens and text
        """
```

### Example Usage

```python
from olmo.models.molmo.molmo import MolmoConfig, Molmo

# Load config
config = MolmoConfig.load("config.yaml")

# Create model
model = Molmo(config)

# Load checkpoint
model.load_checkpoint("checkpoint_dir")
model.eval()

# Prepare inputs
preprocessor = config.build_preprocessor(for_inference=True)
inputs = preprocessor({"image": image, "messages": messages})

# Forward pass
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        images=inputs["images"]
    )
    
# Generation
with torch.no_grad():
    generated = model.generate(
        input_ids=inputs["input_ids"],
        images=inputs["images"],
        max_new_tokens=200
    )
    print(generated.text[0])
```

## VideoOlmo

Video understanding variant.

### VideoOlmoConfig

```python
@dataclass
class VideoOlmoConfig(MolmoConfig):
    """Configuration for VideoOlmo."""
    
    max_frames: int = 16
    # Maximum number of frames per video
    
    frame_sampling: str = "uniform"
    # Frame sampling strategy: "uniform", "random", "adaptive"
```

### VideoOlmo Class

```python
class VideoOlmo(ModelBase):
    """VideoOlmo for video understanding."""
    
    def __init__(self, config: VideoOlmoConfig):
        """Initialize video model."""
        
    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,  # [B, T, C, H, W]
        **kwargs
    ) -> OLMoOutput:
        """
        Forward pass with video frames.
        
        Args:
            input_ids: Token IDs
            images: Video frames [batch, time, channels, height, width]
            
        Returns:
            OLMoOutput
        """
```

## HeMolmo

Hierarchical encoding variant.

### HeMolmoConfig

```python
@dataclass
class HeMolmoConfig(BaseModelConfig):
    """Configuration for HeMolmo."""
    
    llm: LlmConfig
    vision_backbone: Optional[MolmoVisionBackboneConfig]
    token_scorer: TokenScorerConfig
    # Token selection network configuration
    
    selection_ratio: float = 0.25
    # Fraction of tokens to keep after selection
```

### HeMolmo Class

```python
class HeMolmo(ModelBase):
    """HeMolmo with hierarchical token selection."""
    
    def __init__(self, config: HeMolmoConfig):
        """Initialize hierarchical model."""
        
    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> OLMoOutput:
        """
        Forward pass with token selection.
        
        Token selection reduces computational cost by
        selecting only important image tokens.
        """
```

## Output Classes

### OLMoOutput

```python
class OLMoOutput(NamedTuple):
    """Output from forward pass."""
    
    logits: torch.Tensor
    # [batch, seq_len, vocab_size] - Output logits
    
    loss: Optional[torch.Tensor] = None
    # Scalar loss if labels provided
    
    past_key_values: Optional[Tuple] = None
    # Cached keys/values for generation
    
    hidden_states: Optional[torch.Tensor] = None
    # Hidden states if requested
    
    attentions: Optional[Tuple] = None
    # Attention weights if requested
```

### OLMoGenerateOutput

```python
class OLMoGenerateOutput(NamedTuple):
    """Output from generation."""
    
    token_ids: torch.Tensor
    # [batch, seq_len] - Generated token IDs
    
    text: List[str]
    # Decoded text for each example
    
    scores: Optional[torch.Tensor] = None
    # Generation scores if requested
```

## Configuration Loading

### From YAML

```python
config = MolmoConfig.load("path/to/config.yaml")
```

### From Dictionary

```python
config = MolmoConfig.new(
    llm=LlmConfig(
        d_model=4096,
        n_heads=32,
        n_layers=32,
    ),
    vision_backbone=MolmoVisionBackboneConfig(
        image_model_type="openai",
    ),
)
```

### With Overrides

```python
config = MolmoConfig.load(
    "config.yaml",
    overrides=["llm.n_layers=16", "vision_backbone.image_model_type=siglip"]
)
```

## Checkpointing

### Save Checkpoint

```python
model.save_checkpoint(
    path="/path/to/checkpoint",
    sharded=True,  # Save as sharded checkpoint
)
```

### Load Checkpoint

```python
model.load_checkpoint(
    path="/path/to/checkpoint",
    strict=True,  # Require exact match
)
```

### Load Pretrained

```python
# Load from HuggingFace Hub
model = Molmo.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)
```

## Device Placement

```python
# Single GPU
model = model.cuda()
model = model.to("cuda:0")

# CPU
model = model.cpu()

# Mixed precision
model = model.to(torch.bfloat16)
model = model.to(torch.float16)
```

## Distributed Training

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap with FSDP
model = FSDP(
    model,
    auto_wrap_policy=model.get_fsdp_wrap_policy(),
    mixed_precision=mixed_precision_policy,
)
```

## See Also

- [Training API](training.md)
- [Preprocessing API](preprocessing.md)
- [Datasets API](datasets.md)
- [Training Guide](../guides/training_guide.md)


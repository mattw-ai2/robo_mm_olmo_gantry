# Data Pipeline

This document describes how data flows through the Molmo training and inference pipeline.

## Overview

The data pipeline transforms raw data (images, videos, text) into model inputs through several stages:

```
Raw Data → Dataset → Preprocessor → Collator → Model
```

## Data Flow Stages

### 1. Dataset Loading

**Base Dataset Interface:**

```python
class Dataset:
    def __len__(self) -> int:
        """Return dataset size."""
        
    def get(self, item: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """
        Get a single example with deterministic randomness.
        
        Returns:
            {
                "image": np.ndarray or List[np.ndarray],
                "messages": List[Dict],  # Conversation format
                "metadata": Dict,  # Optional metadata
            }
        """
```

**Dataset Types:**

1. **DatasetBase:** Load all data in memory
   ```python
   class MyDataset(DatasetBase):
       def load(self):
           # Load entire dataset
           return list_of_examples
   ```

2. **HfDataset:** Load from HuggingFace
   ```python
   class MyDataset(HfDataset):
       PATH = "hf_dataset_name"
   ```

3. **Custom Dataset:** Any implementation
   ```python
   class CustomDataset(Dataset):
       def __len__(self):
           return self.size
           
       def get(self, item, rng):
           # Load and return example
           return example
   ```

### 2. Data Formatting

**DataFormatter converts task-specific data to prompts:**

```python
class DataFormatter:
    """Format different tasks into consistent prompt format."""
    
    def format_qa(
        self,
        question: str,
        answer: str,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Format QA into messages."""
        return {
            "image": image,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
    
    def format_caption(
        self,
        image: np.ndarray,
        caption: str,
    ) -> Dict[str, Any]:
        """Format captioning task."""
        return {
            "image": image,
            "messages": [
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": caption},
            ]
        }
```

**Task-Specific Formatting:**

- **VQA:** Question → Answer
- **Captioning:** Image → Caption
- **Pointing:** Question → "Point: <x, y>"
- **Counting:** Image → "Count: N"
- **Instruction Following:** Instruction → Response

### 3. Preprocessing

**Image Preprocessing:**

```python
class ImagePreprocessor:
    def __call__(
        self,
        image: np.ndarray,  # [H, W, C]
        rng: np.random.RandomState,
    ) -> torch.Tensor:  # [C, H', W']
        """
        1. Resize image
        2. Apply data augmentation (if training)
        3. Normalize
        4. Convert to tensor
        """
```

**Operations:**
- **Resize:** To target resolution (e.g., 336×336)
- **Crop:** Random crops during training
- **Padding:** Maintain aspect ratio
- **Normalization:** ImageNet mean/std
- **Augmentation:** Random flips, color jitter (optional)

**Text Preprocessing:**

```python
class InterleavedTextPreprocessor:
    def __call__(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Tokenizer,
    ) -> List[int]:
        """
        Convert messages to token IDs.
        
        Returns list of token IDs with special tokens for images.
        """
```

**Operations:**
- **Tokenization:** Text → Token IDs
- **Special Tokens:** Add image markers
- **Format:** Apply chat template
- **Truncation:** Handle max length

**Multimodal Preprocessing:**

```python
class MolmoPreprocessor:
    def __call__(
        self,
        example: Dict[str, Any],
        rng: np.random.RandomState,
    ) -> Dict[str, Any]:
        """
        Process both image and text.
        
        Returns:
            {
                "images": torch.Tensor,  # [n_images, C, H, W]
                "input_ids": List[int],
                "labels": List[int],
                "image_input_idx": List[int],  # Image positions
                "metadata": Dict,
            }
        """
```

### 4. Collation

**Collator batches examples with padding:**

```python
class MMCollator:
    def __call__(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch examples with padding.
        
        Returns:
            {
                "input_ids": [B, max_len],
                "labels": [B, max_len],
                "images": [B, max_images, C, H, W],
                "image_masks": [B, max_images, n_patches],
                "image_input_idx": [B, max_images],
                "attention_mask": [B, max_len],
            }
        """
```

**Padding Strategies:**

1. **Fixed Length:** Pad to max_seq_len
   ```python
   pad_mode = "max_len"
   ```

2. **Batch Max:** Pad to max in batch
   ```python
   pad_mode = "batch_max"
   ```

3. **Multiple of N:** Pad to multiple (e.g., 128)
   ```python
   pad_mode = "to_128"
   ```

### 5. Data Loading

**PyTorch DataLoader:**

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
)

for batch in dataloader:
    # batch ready for model
    outputs = model(**batch)
```

**Distributed Loading:**

```python
# Each GPU gets different data
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=per_device_batch_size,
    ...
)
```

## Dataset Mixtures

**Combine multiple datasets:**

```python
class IterableDatasetMixture:
    """Mix multiple datasets with configurable weights."""
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        seed: int,
    ):
        # Sample from datasets according to weights
        ...
```

**Example:**

```python
mixture = IterableDatasetMixture(
    datasets=[
        PixMoCap(),      # Captioning
        VQA2(),          # VQA
        PixMoPoints(),   # Pointing
    ],
    weights=[0.5, 0.3, 0.2],  # Sample proportions
    seed=42,
)
```

## Deterministic Data Iteration

**For reproducibility:**

```python
class DeterministicDataset:
    """Wrapper that ensures deterministic iteration."""
    
    def get(self, idx: int, epoch: int = 0) -> Dict:
        # Seed based on idx and epoch
        seed = (self.base_seed * 195172 + idx + len(self) * epoch) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        
        # Get example with deterministic randomness
        example = self.dataset.get(idx, rng)
        return example
```

**Benefits:**
- Reproducible training
- Deterministic augmentation
- Exact resume after preemption

## Special Data Handling

### Video Data

```python
class VideoPreprocessor:
    def __call__(
        self,
        video: np.ndarray,  # [T, H, W, C]
        max_frames: int = 16,
    ) -> torch.Tensor:  # [T', C, H, W]
        """
        1. Sample frames (uniform or adaptive)
        2. Resize each frame
        3. Normalize
        4. Stack into tensor
        """
```

**Frame Sampling:**
- **Uniform:** Evenly spaced frames
- **Random:** Random frames during training
- **Adaptive:** Based on motion/content

### Robot Data

```python
class RobotDataset(Dataset):
    """Specialized for robot navigation data."""
    
    def get(self, idx: int, rng) -> Dict:
        """
        Returns navigation episode with:
        - Multi-view images
        - Task description
        - Action sequence
        - Success labels
        """
```

**Unique Features:**
- Multi-view images (egocentric, overhead)
- Temporal sequences
- Structured outputs (actions)

### Document Data

```python
class DocumentDataset(Dataset):
    """For document understanding tasks."""
    
    def get(self, idx: int, rng) -> Dict:
        """
        Returns:
        - High-resolution document image
        - OCR annotations (optional)
        - Questions about document
        """
```

**Considerations:**
- High resolution needed
- Layout preservation
- Multi-page handling

## Data Augmentation

### Training Augmentations

```python
# Image augmentations
- Random crops
- Resize variations
- Color jitter
- Random flips (horizontal)

# Text augmentations
- Prompt variations
- Synonym replacement
- Paraphrasing
```

### Inference (No Augmentation)

```python
# Fixed processing
- Center crop or resize
- No color changes
- No flips
- Deterministic
```

## Caching and Efficiency

### Dataset Caching

```python
# Cache processed datasets
@functools.lru_cache(maxsize=1)
def get_dataset(name: str) -> Dataset:
    return load_dataset(name)
```

### Feature Caching

```python
# Cache image features during generation
image_features = vision_backbone(images)
# Reuse for multiple text generations
```

### Prefetching

```python
# DataLoader prefetching
dataloader = DataLoader(
    ...,
    num_workers=4,      # Parallel data loading
    prefetch_factor=2,  # Prefetch 2 batches
    pin_memory=True,    # Fast GPU transfer
)
```

## Data Storage

### Local Storage

```python
# Standard file system
dataset_dir = "/path/to/data"
images = glob.glob(f"{dataset_dir}/*.jpg")
```

### HuggingFace Cache

```python
# Automatic caching
export HF_HOME=/path/to/cache
dataset = load_dataset("dataset_name")
```

### Remote Storage

```python
# Google Cloud Storage
gs://bucket/path/to/data

# S3
s3://bucket/path/to/data

# Weka (AI2 internal)
weka://oe-training-default/path/to/data
```

## Performance Optimization

### Bottleneck Analysis

Common bottlenecks:
1. **Disk I/O:** Reading images from disk
2. **Preprocessing:** Image resizing, augmentation
3. **CPU-GPU Transfer:** Moving data to GPU

### Solutions

**1. More Workers:**
```python
num_workers = 8  # Increase parallel loading
```

**2. Larger Prefetch:**
```python
prefetch_factor = 4
```

**3. Pin Memory:**
```python
pin_memory = True  # Faster GPU transfer
```

**4. Larger Batches:**
```python
# Process more examples together
batch_size = 64
```

**5. In-Memory Datasets:**
```python
# Load entire dataset in RAM
dataset.preload()
```

## Best Practices

### For Training
- Use multiple workers (4-8)
- Enable pin_memory
- Use appropriate padding strategy
- Monitor data loading time
- Balance dataset mixture weights

### For Evaluation
- Disable augmentation
- Use deterministic preprocessing
- Cache when possible
- Smaller batch sizes for memory

### For Large Scale
- Use remote storage efficiently
- Shard datasets across nodes
- Monitor network bandwidth
- Use compressed formats when possible

### For Debugging
- Start with small datasets
- num_workers=0 for debugging
- Add extensive logging
- Visualize preprocessed examples

## Example: Complete Pipeline

```python
# 1. Define dataset
dataset = PixMoCap(split="train")

# 2. Build preprocessor
preprocessor = config.build_preprocessor(
    for_inference=False,
    is_training=True,
)

# 3. Wrap with deterministic iteration
dataset = DeterministicDataset(
    dataset=dataset,
    preprocessor=preprocessor,
    seed=42,
)

# 4. Build collator
collator = config.build_collator(
    sequence_length=2048,
    pad_mode="to_128",
)

# 5. Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True,
)

# 6. Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

## Next Steps

- [Training Guide](../guides/training_guide.md) - Training procedures
- [Datasets Documentation](../datasets/pixmo.md) - Specific datasets
- [Custom Datasets](../datasets/custom_datasets.md) - Adding new datasets


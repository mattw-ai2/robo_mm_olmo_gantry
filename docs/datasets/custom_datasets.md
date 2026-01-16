# Creating Custom Datasets

Guide to adding your own datasets to Molmo.

## Dataset Interface

All datasets inherit from the `Dataset` base class:

```python
from olmo.data.dataset import Dataset
import numpy as np

class Dataset:
    def __len__(self) -> int:
        """Return total number of examples."""
        raise NotImplementedError()
    
    def get(self, item: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """
        Get a single example with deterministic randomness.
        
        Args:
            item: Index of example to retrieve
            rng: Random state for deterministic augmentation
            
        Returns:
            Dictionary with 'image', 'messages', and optional 'metadata'
        """
        raise NotImplementedError()
```

## Example Format

Return dictionaries in this format:

```python
{
    "image": np.ndarray,  # [H, W, 3] RGB image
    "messages": [
        {"role": "user", "content": "Question or prompt"},
        {"role": "assistant", "content": "Answer or response"},
    ],
    "metadata": {  # Optional
        "id": str,
        "source": str,
        # ... any other metadata
    }
}
```

## Simple Dataset Example

### Loading from JSON

```python
from olmo.data.dataset import DatasetBase
import json
from PIL import Image
import numpy as np
from pathlib import Path

class MySimpleDataset(DatasetBase):
    """Load dataset from JSON annotations."""
    
    def __init__(self, split: str = "train", data_dir: str = "/path/to/data"):
        self.split = split
        self.data_dir = Path(data_dir)
        super().__init__(split)
    
    def load(self):
        """Load all annotations into memory."""
        annotation_file = self.data_dir / f"{self.split}.json"
        with open(annotation_file) as f:
            return json.load(f)
    
    def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Get a single example."""
        item = self.data[idx]
        
        # Load image
        image_path = self.data_dir / "images" / item["image_filename"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # Format as conversation
        return {
            "image": image,
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ],
            "metadata": {
                "id": item.get("id", idx),
                "source": "my_dataset",
            }
        }
```

### Usage

```python
# Create dataset
dataset = MySimpleDataset(split="train", data_dir="/path/to/data")

# Get example
example = dataset[0]
print(f"Question: {example['messages'][0]['content']}")
print(f"Answer: {example['messages'][1]['content']}")
print(f"Image shape: {example['image'].shape}")
```

## HuggingFace Dataset

### Loading from HF Hub

```python
from olmo.data.dataset import HfDataset
import numpy as np

class MyHfDataset(HfDataset):
    """Load from HuggingFace Datasets."""
    
    PATH = "organization/dataset-name"  # HF Hub path
    
    def __init__(self, split: str = "train"):
        self.split = split
        super().__init__()
    
    @classmethod
    def download(cls, n_procs=None):
        """Download dataset from HF Hub."""
        from datasets import load_dataset
        load_dataset(cls.PATH, num_proc=n_procs)
    
    def load(self):
        """Load dataset from cache."""
        from datasets import load_dataset
        return load_dataset(self.PATH, split=self.split)
    
    def __len__(self):
        return len(self.data)
    
    def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Get a single example."""
        item = self.data[idx]
        
        # Convert HF format to Molmo format
        image = np.array(item["image"])
        
        return {
            "image": image,
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ],
            "metadata": {"id": item["id"]}
        }
```

## Advanced Examples

### Multiple Images per Example

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Example with multiple images."""
    item = self.data[idx]
    
    # Load multiple images
    images = [
        np.array(Image.open(img_path))
        for img_path in item["image_paths"]
    ]
    
    return {
        "image": images,  # List of images
        "messages": [
            {"role": "user", "content": "Compare these images."},
            {"role": "assistant", "content": item["comparison"]},
        ],
    }
```

### Video Dataset

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Example with video frames."""
    item = self.data[idx]
    
    # Load video and extract frames
    frames = self.extract_frames(item["video_path"], max_frames=16)
    
    return {
        "image": frames,  # List of frames
        "messages": [
            {"role": "user", "content": "What happens in this video?"},
            {"role": "assistant", "content": item["caption"]},
        ],
    }
```

### With Data Augmentation

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Example with random augmentation."""
    item = self.data[idx]
    
    # Load image
    image = np.array(Image.open(item["image_path"]))
    
    # Apply random augmentation using provided RNG
    if rng.rand() > 0.5:
        image = np.fliplr(image)  # Random horizontal flip
    
    # Random crop
    h, w = image.shape[:2]
    crop_size = int(min(h, w) * (0.8 + 0.2 * rng.rand()))
    y = rng.randint(0, h - crop_size + 1)
    x = rng.randint(0, w - crop_size + 1)
    image = image[y:y+crop_size, x:x+crop_size]
    
    return {
        "image": image,
        "messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ],
    }
```

### Pointing/Grounding Dataset

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Example with object pointing."""
    item = self.data[idx]
    
    # Format point as "Point: x=X y=Y"
    x, y = item["point"]
    answer = f"Point: x={x} y={y}"
    
    return {
        "image": np.array(Image.open(item["image_path"])),
        "messages": [
            {"role": "user", "content": f"Where is the {item['object']}?"},
            {"role": "assistant", "content": answer},
        ],
    }
```

### Counting Dataset

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Example with counting."""
    item = self.data[idx]
    
    return {
        "image": np.array(Image.open(item["image_path"])),
        "messages": [
            {"role": "user", "content": f"Count the {item['object']}"},
            {"role": "assistant", "content": f"Count: {item['count']}"},
        ],
    }
```

## Registering Dataset

### Add to get_dataset.py

```python
# In olmo/data/get_dataset.py

from my_dataset import MySimpleDataset

DATASET_REGISTRY = {
    "my_dataset": MySimpleDataset,
    # ... other datasets
}

def get_dataset(name: str, split: str = "train", **kwargs):
    """Get dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    
    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(split=split, **kwargs)
```

### Use in Training

```python
from olmo.data.get_dataset import get_dataset

# Load your dataset
dataset = get_dataset("my_dataset", split="train")

# Use in training script
datasets = [
    {"name": "my_dataset", "weight": 0.5},
    {"name": "pixmo_cap", "weight": 0.5},
]
```

## Best Practices

### Efficient Loading

```python
import functools

class EfficientDataset(DatasetBase):
    @functools.lru_cache(maxsize=1000)
    def _load_image(self, path: str) -> np.ndarray:
        """Cache frequently accessed images."""
        return np.array(Image.open(path))
    
    def get(self, idx: int, rng: np.random.RandomState):
        item = self.data[idx]
        image = self._load_image(item["image_path"])
        # ... rest of implementation
```

### Error Handling

```python
def get(self, idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """Robust example loading with error handling."""
    try:
        item = self.data[idx]
        image = np.array(Image.open(item["image_path"]))
    except Exception as e:
        # Log error and return a valid fallback
        print(f"Error loading example {idx}: {e}")
        # Return first example as fallback
        return self.get(0, rng)
    
    return {
        "image": image,
        "messages": item["messages"],
    }
```

### Lazy Loading

```python
class LazyDataset(Dataset):
    """Don't load all data into memory at once."""
    
    def __init__(self, split: str):
        self.split = split
        self.index_file = f"index_{split}.json"
        # Only load index, not full data
        with open(self.index_file) as f:
            self.index = json.load(f)
    
    def __len__(self):
        return len(self.index)
    
    def get(self, idx: int, rng: np.random.RandomState):
        # Load data on-demand
        item_path = self.index[idx]["path"]
        with open(item_path) as f:
            item = json.load(f)
        # ... process item
```

## Testing Your Dataset

```python
def test_dataset():
    """Test your dataset implementation."""
    dataset = MySimpleDataset(split="train")
    
    # Test length
    assert len(dataset) > 0, "Dataset is empty"
    
    # Test example loading
    example = dataset[0]
    assert "image" in example
    assert "messages" in example
    assert isinstance(example["image"], np.ndarray)
    assert len(example["messages"]) >= 2
    
    # Test all examples load
    rng = np.random.RandomState(42)
    for i in range(min(100, len(dataset))):
        try:
            example = dataset.get(i, rng)
            assert example is not None
        except Exception as e:
            print(f"Failed to load example {i}: {e}")
            raise
    
    print("✓ Dataset tests passed")

if __name__ == "__main__":
    test_dataset()
```

## Common Patterns

### Multi-turn Conversations

```python
{
    "image": image,
    "messages": [
        {"role": "user", "content": "What's in this image?"},
        {"role": "assistant", "content": "A cat and a dog."},
        {"role": "user", "content": "What color is the cat?"},
        {"role": "assistant", "content": "The cat is orange."},
    ],
}
```

### Instruction Following

```python
{
    "image": image,
    "messages": [
        {"role": "user", "content": "List all objects in JSON format"},
        {"role": "assistant", "content": '{"objects": ["car", "tree", "person"]}'},
    ],
}
```

### Chain-of-Thought

```python
{
    "image": image,
    "messages": [
        {"role": "user", "content": "How many people are there?"},
        {"role": "assistant", "content": "Let me count: I see one person on the left, two in the center, and one on the right. Total: 4 people."},
    ],
}
```

## Next Steps

- [PixMo Datasets](pixmo.md) - Example datasets
- [Training Guide](../guides/training_guide.md) - Train with your dataset
- [Data Pipeline](../architecture/data_pipeline.md) - How data flows through system


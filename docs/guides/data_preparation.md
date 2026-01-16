# Data Preparation Guide

Guide to downloading, preparing, and managing datasets for Molmo.

## Environment Setup

### Set Data Directory

```bash
export MOLMO_DATA_DIR=/path/to/data
export HF_HOME=/path/to/huggingface/cache
```

Add to `~/.bashrc` for persistence:

```bash
echo 'export MOLMO_DATA_DIR=/path/to/data' >> ~/.bashrc
echo 'export HF_HOME=/path/to/huggingface/cache' >> ~/.bashrc
```

### Storage Requirements

Typical storage needs:
- **PixMo Datasets:** ~500GB
- **Academic Benchmarks:** ~100GB
- **Video Datasets:** ~1TB
- **Robot Datasets:** ~200GB
- **Total (all data):** ~2TB

Plan accordingly and ensure fast storage (SSD preferred).

## Downloading Datasets

### Download All Datasets

```bash
python scripts/download_data.py all --n_proc 12
```

**Options:**
- `--n_proc`: Number of parallel processes (default: 1)
- More processes = faster download (but risk rate limiting)

**Time:** 4-24 hours depending on connection and processes

### Download Specific Dataset

```bash
python scripts/download_data.py ChartQa --n_proc 12
```

### Available Datasets

**PixMo (Proprietary):**
- `pixmo_cap` - Dense captions
- `pixmo_points` - Pointing annotations
- `pixmo_count` - Counting annotations
- `pixmo_docs` - Document understanding
- `pixmo_cap_qa` - Caption-based QA

**Academic Benchmarks:**
- `vqa2` - VQA v2 dataset
- `text_vqa` - Text-based VQA
- `doc_qa` - Document QA
- `chart_qa` - Chart understanding
- `ai2d` - Science diagrams
- `math_vista` - Mathematical reasoning
- `mmmu` - Multidisciplinary understanding
- `real_world_qa` - Real-world scenarios
- `ok_vqa` - Knowledge-based VQA
- `science_qa` - Science QA
- `tally_qa` - Counting QA

**Video Datasets:**
- `video_mme` - Video understanding
- `mlvu` - Long video understanding
- `perception_test` - Perception tasks
- `ego_schema` - Egocentric video
- `next_qa` - Video QA

### Manual Download Required

Some datasets require manual steps:

**InfoQA:**
1. Visit dataset website
2. Download images manually
3. Place in `$MOLMO_DATA_DIR/info_qa/`
4. Run: `python scripts/download_data.py info_qa`

**Scene-Text:**
1. Download from official source
2. Extract to `$MOLMO_DATA_DIR/scene_text/`
3. Run: `python scripts/download_data.py scene_text`

## Dataset Structure

After downloading, data is organized as:

```
$MOLMO_DATA_DIR/
├── torch_datasets/
│   ├── pixmo_cap/
│   ├── pixmo_points/
│   ├── vqa2/
│   └── ...
├── video_datasets/
│   ├── video_mme/
│   ├── mlvu/
│   └── ...
├── robot_datasets/
│   └── vida/
└── huggingface/  (if HF_HOME points here)
    └── datasets/
```

## HuggingFace Datasets

Many datasets download via HuggingFace:

### Configuration

```bash
# Set cache location
export HF_HOME=/path/to/huggingface/cache

# Optional: HF token for gated datasets
export HF_ACCESS_TOKEN=your_token
```

### Offline Mode

After downloading, use offline mode:

```bash
export HF_DATASETS_OFFLINE=1
```

This prevents unnecessary network requests during training.

### Managing Cache

```bash
# Check cache size
du -sh $HF_HOME

# Clear cache (careful!)
rm -rf $HF_HOME/datasets

# Clean old versions
huggingface-cli delete-cache
```

## Visualizing Datasets

Before training, visualize to verify data:

```bash
python scripts/dataset_visualize.py chart_qa /path/to/output/dir
```

This generates HTML visualizations of examples.

### Example Visualization

```bash
# Visualize 100 examples from PixMo Cap
python scripts/dataset_visualize.py pixmo_cap ./viz \
    --split=train \
    --num_examples=100
```

Open `./viz/index.html` in browser to view.

## Dataset Verification

### Check Dataset Loading

```python
from olmo.data.get_dataset import get_dataset

# Load dataset
dataset = get_dataset("pixmo_cap", split="train")

print(f"Dataset size: {len(dataset)}")

# Get example
example = dataset[0]
print(f"Keys: {example.keys()}")
print(f"Image shape: {example['image'].shape}")
print(f"Messages: {example['messages']}")
```

### Verify Data Integrity

```bash
# Test data loading for all datasets
python scripts/verify_datasets.py
```

## Custom Datasets

### Adding Local Dataset

1. **Organize data:**
   ```
   my_dataset/
   ├── images/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   └── annotations.json
   ```

2. **Create dataset class:**
   ```python
   from olmo.data.dataset import DatasetBase
   import json
   from PIL import Image
   
   class MyDataset(DatasetBase):
       def load(self):
           with open(f"{self.data_dir}/annotations.json") as f:
               return json.load(f)
       
       def get(self, idx, rng):
           item = self.data[idx]
           image = Image.open(f"{self.data_dir}/images/{item['image']}")
           
           return {
               "image": np.array(image),
               "messages": [
                   {"role": "user", "content": item["question"]},
                   {"role": "assistant", "content": item["answer"]},
               ],
               "metadata": {"id": item["id"]}
           }
   ```

3. **Register dataset:**
   ```python
   from my_dataset import MyDataset
   
   dataset = MyDataset(split="train")
   ```

See [Custom Datasets Guide](../datasets/custom_datasets.md) for details.

### Converting from Other Formats

**From COCO:**
```python
from pycocotools.coco import COCO

coco = COCO("annotations.json")
# Convert to Molmo format
```

**From LVIS:**
```python
# Similar to COCO conversion
```

**From Custom JSON:**
```python
import json

with open("data.json") as f:
    data = json.load(f)
    
# Transform to Molmo format
```

## Data Preprocessing

### Image Preprocessing

Images are preprocessed automatically during training:

1. **Resize** to target resolution
2. **Normalize** with ImageNet stats
3. **Augment** (if training)
4. **Convert** to tensors

Configuration in model config:

```python
mm_preprocessor = MolmoPreprocessorConfig(
    max_crops=1,
    overlap_margins=[0, 0],
    base_image_input_size=[336, 336],
    image_token_length_w=24,
    image_token_length_h=24,
)
```

### Text Preprocessing

Text is tokenized and formatted:

1. **Apply chat template**
2. **Tokenize** with model tokenizer
3. **Add special tokens** (image markers)
4. **Truncate** to max length

### Video Preprocessing

Videos are sampled to frames:

```python
# Uniform sampling
frames = sample_frames(video, n_frames=16, method="uniform")

# Each frame preprocessed as image
```

## Data Augmentation

### Training Augmentations

Controlled via preprocessor:

```python
# Random crops
use_random_crops = True

# Color jitter
use_color_jitter = False  # Usually disabled for Molmo

# Horizontal flips
use_horizontal_flip = False  # Usually disabled
```

### No Augmentation for Eval

Evaluation uses deterministic preprocessing:
- Center crop or resize
- No random operations
- Reproducible results

## Data Mixtures

### Creating Dataset Mixtures

```python
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture

mixture = IterableDatasetMixture(
    datasets=[
        get_dataset("pixmo_cap"),
        get_dataset("vqa2"),
        get_dataset("text_vqa"),
    ],
    weights=[0.5, 0.25, 0.25],
    seed=42
)
```

### Mixture Strategies

**Proportional:** Sample proportionally to weights
```python
weights=[0.5, 0.3, 0.2]  # 50%, 30%, 20%
```

**Uniform:** Sample equally from all datasets
```python
weights=[1.0, 1.0, 1.0]  # Equal sampling
```

**Upsampling Small Datasets:**
```python
# Oversample small dataset
weights=[0.3, 0.7]  # Even if dataset 2 is smaller
```

## Remote Storage

### Google Cloud Storage

```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Or use JSON directly
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type": "service_account", ...}'

# Access data
--data_path=gs://bucket/path/to/data
```

### AWS S3

```bash
# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Access data
--data_path=s3://bucket/path/to/data
```

### Weka (AI2 Internal)

```bash
# On Cirrascale machines
export MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo

# On other machines with credentials
export WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000"
export WEKA_PROFILE="weka"
export AWS_CREDENTIALS=credentials_json

# Access data
--data_path=weka://oe-training-default/mm-olmo
```

## Data Loading Performance

### Optimization Tips

1. **More Workers:**
   ```python
   num_workers=4  # or more
   ```

2. **Prefetching:**
   ```python
   prefetch_factor=2
   ```

3. **Pin Memory:**
   ```python
   pin_memory=True
   ```

4. **Fast Storage:**
   - Local SSD > Network storage
   - Cache frequently used data

5. **Parallel Download:**
   ```bash
   --n_proc=12  # When downloading
   ```

### Monitoring

```python
import time

start = time.time()
for i, batch in enumerate(dataloader):
    if i == 100:
        break
elapsed = time.time() - start
print(f"Time per batch: {elapsed/100:.3f}s")
```

## Troubleshooting

### Download Failures

**Problem:** Download interrupted or fails

**Solution:**
- Downloads are resumable - rerun the same command
- Reduce `n_proc` if getting rate limited
- Check internet connection
- Verify storage space

### Missing Files

**Problem:** "File not found" errors

**Solution:**
- Verify `MOLMO_DATA_DIR` is set correctly
- Check dataset downloaded completely
- Look for manual download requirements
- Verify file permissions

### Slow Loading

**Problem:** Training bottlenecked by data loading

**Solution:**
- Increase `num_workers`
- Use faster storage (SSD)
- Enable `HF_DATASETS_OFFLINE=1`
- Preload datasets into memory
- Check CPU usage

### Corrupted Data

**Problem:** Invalid images or parsing errors

**Solution:**
- Delete and redownload dataset
- Check for disk errors
- Verify download checksums
- Report issue if persistent

## Best Practices

1. **Download Once:** Store in persistent location
2. **Verify First:** Visualize before training
3. **Use Offline Mode:** After download
4. **Monitor Space:** Datasets are large
5. **Fast Storage:** Use SSD for frequently accessed data
6. **Backup:** Important custom datasets
7. **Document:** Keep notes on data versions

## Next Steps

- **[Training Guide](training_guide.md)** - Start training
- **[Custom Datasets](../datasets/custom_datasets.md)** - Add your own data
- **[Dataset Documentation](../datasets/pixmo.md)** - Learn about specific datasets
- **[Configuration](configuration.md)** - Configure data loading


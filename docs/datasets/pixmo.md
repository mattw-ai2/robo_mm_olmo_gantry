# PixMo Dataset Family

PixMo is a family of proprietary datasets developed by AI2 for training multimodal models.

## Overview

PixMo consists of several complementary datasets:
- **PixMo-Cap:** Dense image captions
- **PixMo-Points:** Object pointing annotations
- **PixMo-Count:** Object counting
- **PixMo-Docs:** Document understanding
- **PixMo-CapQA:** Caption-based QA
- **PixMo-AskModelAnything:** Diverse questions
- **PixMo-Clocks:** Clock reading

## PixMo-Cap (Dense Captions)

### Description

High-quality, detailed image descriptions suitable for training captioning models.

### Statistics

- **Images:** ~1.2M
- **Splits:** train, validation
- **Average caption length:** 100-200 tokens
- **Domains:** General images, documents, charts, scenes

### Data Format

```python
{
    "image": np.ndarray,  # [H, W, 3]
    "caption": str,  # Detailed description
    "metadata": {
        "source": str,
        "id": str,
    }
}
```

### Example

```json
{
    "caption": "A bustling city street at sunset with tall buildings lining both sides. "
               "People walk along the sidewalks while cars and buses navigate the traffic. "
               "The warm orange and pink hues of the sky create a dramatic backdrop.",
    "metadata": {"source": "cc_images", "id": "img_12345"}
}
```

### Usage

```python
from olmo.data.pixmo_datasets import PixMoCap

dataset = PixMoCap(split="train")
example = dataset[0]

print(f"Caption: {example['caption']}")
print(f"Image shape: {example['image'].shape}")
```

### Training

Primary dataset for dense caption pretraining:

```bash
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
    --task=pixmo_cap
```

## PixMo-Points (Pointing)

### Description

Annotations for object pointing tasks. Each example has an object name and pixel coordinates.

### Statistics

- **Images:** ~500K
- **Points per image:** 1-5
- **Object categories:** Diverse (animals, objects, people, etc.)

### Data Format

```python
{
    "image": np.ndarray,
    "points": List[Tuple[int, int]],  # [(x, y), ...]
    "object_names": List[str],  # Object being pointed to
    "query": str,  # "Where is the cat?"
}
```

### Example

```json
{
    "query": "Point to the red car",
    "points": [[245, 378]],
    "object_names": ["red car"]
}
```

### Usage

```python
from olmo.data.pixmo_datasets import PixMoPoints

dataset = PixMoPoints(split="train")
example = dataset[0]

print(f"Query: {example['query']}")
print(f"Points: {example['points']}")
```

### Output Format

Model generates: `"Point: x=245 y=378"`

## PixMo-Count (Counting)

### Description

Object counting annotations with precise counts.

### Statistics

- **Images:** ~200K
- **Count range:** 0-30 objects
- **Object types:** Common objects, people, animals

### Data Format

```python
{
    "image": np.ndarray,
    "count": int,
    "object_name": str,  # "cars", "people", etc.
    "query": str,  # "How many cars are in the image?"
}
```

### Example

```json
{
    "query": "Count the number of people in this image",
    "count": 7,
    "object_name": "people"
}
```

### Usage

```python
from olmo.data.pixmo_datasets import PixMoCount

dataset = PixMoCount(split="train")
example = dataset[0]

print(f"Query: {example['query']}")
print(f"Count: {example['count']}")
```

## PixMo-Docs (Document Understanding)

### Description

Document images with questions and answers about content.

### Statistics

- **Documents:** ~300K
- **Questions per doc:** 1-10
- **Document types:** Forms, invoices, reports, presentations

### Data Format

```python
{
    "image": np.ndarray,  # High-res document
    "questions": List[str],
    "answers": List[str],
    "document_type": str,
}
```

### Example

```json
{
    "questions": ["What is the invoice number?", "What is the total amount?"],
    "answers": ["INV-12345", "$1,250.00"],
    "document_type": "invoice"
}
```

### Usage

```python
from olmo.data.pixmo_datasets import PixMoDocs

dataset = PixMoDocs(split="train")
example = dataset[0]

print(f"Questions: {example['questions']}")
print(f"Answers: {example['answers']}")
```

### Best Practices

- Use high resolution (768x768 or higher)
- Enable OCR preprocessing if available
- Fine-tune on domain-specific documents

## PixMo-CapQA (Caption-based QA)

### Description

Questions generated from image captions with answers derivable from the caption.

### Statistics

- **Examples:** ~800K
- **QA pairs per image:** 2-5

### Data Format

```python
{
    "image": np.ndarray,
    "question": str,
    "answer": str,
    "caption": str,  # Source caption
}
```

## PixMo-AskModelAnything

### Description

Diverse, open-ended questions covering reasoning, knowledge, and perception.

### Statistics

- **Examples:** ~400K
- **Question types:** What, where, why, how, count, etc.

### Usage

Good for instruction tuning and diverse task coverage.

## PixMo-Clocks

### Description

Clock images with time reading annotations.

### Statistics

- **Images:** ~50K
- **Time formats:** Analog and digital clocks

### Data Format

```python
{
    "image": np.ndarray,
    "time": str,  # "3:45 PM"
    "query": str,  # "What time is shown on the clock?"
}
```

## Downloading PixMo

### All PixMo Datasets

```bash
python scripts/download_data.py all --n_proc 12
```

### Individual Datasets

```bash
python scripts/download_data.py pixmo_cap --n_proc 12
python scripts/download_data.py pixmo_points --n_proc 12
python scripts/download_data.py pixmo_count --n_proc 12
python scripts/download_data.py pixmo_docs --n_proc 12
```

### Storage Requirements

- PixMo-Cap: ~200GB
- PixMo-Points: ~100GB
- PixMo-Count: ~50GB
- PixMo-Docs: ~150GB
- Total: ~500GB

## Data Quality

### Caption Quality

PixMo-Cap captions are:
- Detailed (100-200 tokens)
- Descriptive (objects, colors, spatial relationships)
- Diverse (various domains and styles)
- High-quality (human-verified samples)

### Annotation Quality

PixMo-Points and PixMo-Count annotations:
- Human-verified
- Precise coordinates
- Accurate counts
- Diverse object categories

## Using PixMo in Training

### Caption Pretraining

```bash
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
    --data.dataset=pixmo_cap \
    --max_steps=15000
```

### Multitask with PixMo

```python
# In training script
datasets = [
    {"name": "pixmo_cap", "weight": 0.4},
    {"name": "pixmo_points", "weight": 0.2},
    {"name": "pixmo_count", "weight": 0.15},
    {"name": "pixmo_docs", "weight": 0.15},
    {"name": "pixmo_cap_qa", "weight": 0.1},
]
```

## Evaluation Sets

### PixMo-Points Eval

```bash
torchrun --nproc-per-node=8 launch_scripts/eval_downstream.py \
    Molmo-7B-D-0924 \
    pixmo_points:validation \
    --high_res --fsdp
```

### Dense Caption Eval

```bash
torchrun --nproc-per-node=8 launch_scripts/eval.py \
    --task dense_caption_eval \
    /path/to/checkpoint

# GPT-4 evaluation
python scripts/gpt_dense_caption_eval.py predictions.json --metrics all
```

## Citation

If you use PixMo datasets, please cite the Molmo paper:

```bibtex
@article{molmo2024,
  title={Molmo: Multimodal Open Language Model},
  author={Deitke, Matt and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```

## License

PixMo datasets are released under [appropriate license]. See dataset documentation for details.

## Related Datasets

- [Academic Benchmarks](academic_benchmarks.md)
- [Video Datasets](video_datasets.md)
- [Robot Datasets](robot_datasets.md)
- [Custom Datasets](custom_datasets.md)


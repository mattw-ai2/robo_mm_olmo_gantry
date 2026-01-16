# Molmo Documentation

Welcome to the Molmo (Multimodal Open Language Model) documentation! Molmo is a family of state-of-the-art open multimodal models developed by the Allen Institute for AI.

## Quick Links

- **[Installation Guide](guides/installation.md)** - Get started with Molmo
- **[Quick Start](guides/quickstart.md)** - Your first Molmo model
- **[Training Guide](guides/training_guide.md)** - Train your own models
- **[Evaluation Guide](guides/evaluation_guide.md)** - Evaluate model performance
- **[API Reference](api/models.md)** - Detailed API documentation

## What is Molmo?

Molmo is a comprehensive framework for training and deploying multimodal language models that can:
- **Understand and generate** descriptions of images and videos
- **Answer questions** about visual content
- **Point to objects** and regions in images
- **Count objects** and perform visual reasoning
- **Navigate** robotic environments
- **Handle multiple tasks** with a single model

## Architecture Overview

Molmo combines:
- **Vision Encoders:** CLIP, SigLIP, DINOv2 for visual understanding
- **Language Models:** OLMo, Qwen2, Llama-based architectures
- **Multimodal Fusion:** Advanced attention mechanisms for vision-language integration

See [Architecture Overview](architecture/overview.md) for details.

## Key Features

### Multiple Model Variants
- **Molmo:** Standard image-text model
- **VideoOlmo:** Video understanding variant
- **HeMolmo:** Hierarchical encoding for efficiency
- **MolmoE:** Mixture-of-experts variant

### Extensive Dataset Support
- **PixMo:** Proprietary dataset family (captions, points, counts, documents)
- **Robot Datasets:** Navigation and manipulation tasks
- **Video Datasets:** Video understanding benchmarks
- **Academic Benchmarks:** VQA2, TextVQA, MMMU, MathVista, and more

### Production-Ready Training
- **Distributed Training:** FSDP support for multi-node training
- **Mixed Precision:** Efficient training with automatic mixed precision
- **Checkpointing:** Robust checkpoint management with remote storage
- **Monitoring:** Integration with Weights & Biases

### Flexible Evaluation
- **Downstream Tasks:** 11+ evaluation benchmarks
- **Custom Metrics:** Extensible evaluation framework
- **Batch Processing:** Efficient inference with FSDP

## Documentation Structure

### 📚 Guides
Step-by-step tutorials for common tasks:
- [Installation](guides/installation.md)
- [Quick Start](guides/quickstart.md)
- [Training Guide](guides/training_guide.md)
- [Evaluation Guide](guides/evaluation_guide.md)
- [Deployment Guide](guides/deployment_guide.md)
- [Data Preparation](guides/data_preparation.md)
- [Distributed Training](guides/distributed_training.md)
- [Configuration System](guides/configuration.md)

### 🏗️ Architecture
Technical documentation on model architecture:
- [Overview](architecture/overview.md)
- [Model Architectures](architecture/model_architectures.md)
- [Vision Backbone](architecture/vision_backbone.md)
- [LLM Components](architecture/llm_components.md)
- [Data Pipeline](architecture/data_pipeline.md)

### 🔧 API Reference
Detailed API documentation:
- [Models](api/models.md)
- [Datasets](api/datasets.md)
- [Training](api/training.md)
- [Evaluation](api/evaluation.md)
- [Preprocessing](api/preprocessing.md)
- [Utilities](api/utilities.md)

### 📊 Datasets
Information about datasets:
- [PixMo](datasets/pixmo.md)
- [Robot Datasets](datasets/robot_datasets.md)
- [Video Datasets](datasets/video_datasets.md)
- [Academic Benchmarks](datasets/academic_benchmarks.md)
- [Custom Datasets](datasets/custom_datasets.md)

### 📓 Tutorials
Interactive examples:
- [Basic Inference](tutorials/01_basic_inference.ipynb)
- [Training a Captioner](tutorials/02_training_captioner.ipynb)
- [Custom Dataset](tutorials/03_custom_dataset.ipynb)
- [Running Evaluations](tutorials/04_evaluation.ipynb)
- [Fine-tuning Models](tutorials/05_finetuning.ipynb)

### 📖 Reference
Additional resources:
- [Configuration Reference](reference/configuration_reference.md)
- [CLI Reference](reference/cli_reference.md)
- [Environment Variables](reference/environment_variables.md)
- [Troubleshooting](reference/troubleshooting.md)
- [FAQ](faq.md)
- [Changelog](changelog.md)

## Getting Started

### Installation

```bash
git clone https://github.com/allenai/molmo.git
cd molmo
pip install -e .[all]
```

### Quick Example

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Process image and text
image = Image.open("example.jpg")
inputs = processor(text="Describe this image", images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate response
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

See [Quick Start Guide](guides/quickstart.md) for more examples.

## Model Zoo

Available models on HuggingFace:
- [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) - Base 7B model
- [Molmo-7B-O-0924](https://huggingface.co/allenai/Molmo-7B-O-0924) - Optimized 7B
- [Molmo-72B-0924](https://huggingface.co/allenai/Molmo-72B-0924) - Large 72B model
- [MolmoE-1B-0924](https://huggingface.co/allenai/MolmoE-1B-0924) - Efficient 1B MoE

See the [model collection](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19) for all available models.

## Community and Support

- **GitHub Issues:** [Report bugs and request features](https://github.com/allenai/molmo/issues)
- **GitHub Discussions:** [Ask questions and share ideas](https://github.com/allenai/molmo/discussions)
- **Paper:** [Read the Molmo paper](https://arxiv.org/abs/2409.17146)
- **Blog:** [Molmo blog post](https://molmo.allenai.org/blog)

## Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details on:
- Setting up development environment
- Code style and standards
- Testing guidelines
- Pull request process

## Citation

If you use Molmo in your research, please cite:

```bibtex
@article{molmo2024,
  title={Molmo: Multimodal Open Language Model},
  author={Deitke, Matt and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```

## License

Molmo is released under the Apache License 2.0. See [LICENSE](../LICENSE) for details.


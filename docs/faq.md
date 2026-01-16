# Frequently Asked Questions

## General Questions

### What is Molmo?

Molmo (Multimodal Open Language Model) is a family of open-source multimodal models that can understand images, videos, and text. It's developed by the Allen Institute for AI and supports tasks like image captioning, visual question answering, object pointing, and more.

### What makes Molmo different?

- **Fully Open:** Training data, code, and model weights are all released
- **Strong Performance:** Competitive with proprietary models
- **Multimodal:** Understands images, videos, documents, and text
- **Flexible:** Multiple model variants for different use cases
- **Production-Ready:** Includes serving, evaluation, and deployment tools

### Which model should I use?

- **Research:** Molmo-7B (best balance of quality and speed)
- **Production:** MolmoE-1B (efficient inference)
- **Best Quality:** Molmo-72B (highest performance)
- **Limited Resources:** Molmo-1B (smallest footprint)
- **Video:** VideoOlmo variants

## Installation & Setup

### How do I install Molmo?

```bash
git clone https://github.com/allenai/molmo.git
cd molmo
pip install -e .[all]
```

See the [Installation Guide](guides/installation.md) for details.

### What are the system requirements?

- Python 3.10+
- CUDA 11.8+ for GPU support
- 16GB+ GPU RAM for inference
- 32GB+ system RAM recommended
- 100GB+ storage for datasets

### Do I need a GPU?

For inference: A GPU is highly recommended but not strictly required. CPU inference is very slow.

For training: A GPU is required. Training 7B models needs 8x A100 40GB GPUs or equivalent.

### Can I use AMD GPUs?

Currently, Molmo is optimized for NVIDIA GPUs with CUDA. AMD ROCm support is not officially tested but may work with some modifications.

## Usage

### How do I run inference?

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Process image and generate
inputs = processor(text="Describe this image", images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
```

See the [Quick Start Guide](guides/quickstart.md) for more examples.

### How do I train a model?

```bash
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
    --save_folder=./checkpoints
```

See the [Training Guide](guides/training_guide.md) for complete instructions.

### How do I evaluate a model?

```bash
torchrun --nproc-per-node=8 launch_scripts/eval_downstream.py \
    Molmo-7B-D-0924 text_vqa
```

See the [Evaluation Guide](guides/evaluation_guide.md) for details.

## Technical Questions

### What vision encoders are supported?

- CLIP (OpenAI)
- SigLIP (Google)
- DINOv2 (Meta)
- MetaCLIP

### What LLM backbones are supported?

- OLMo (AI2)
- Qwen2 (Alibaba)
- Llama (Meta)

### How does multimodal fusion work?

Images are encoded by the vision backbone, projected to the LLM dimension, and inserted as special tokens in the text sequence. The LLM processes the combined image-text sequence with attention mechanisms.

### Can I use custom images/videos?

Yes! You can pass any image or video to the model for inference. For training, you can create custom datasets by implementing the Dataset interface.

### What output formats are supported?

- Text (captions, answers)
- Coordinates (pointing tasks): `"Point: x=123 y=456"`
- Counts (counting tasks): `"Count: 5"`
- JSON (structured outputs)
- Any text-based format

## Training

### How long does training take?

- **Caption pretraining:** 2-3 days on 8x A100 GPUs
- **Multitask finetuning:** 1-2 days on 8x A100 GPUs
- **Total:** 3-5 days for full training pipeline

### How much does training cost?

At typical cloud GPU rates:
- **8x A100 80GB:** ~$25-30/hour
- **Caption pretraining:** ~$1,500-2,000
- **Multitask finetuning:** ~$750-1,000
- **Total:** ~$2,500-3,000

Academic users may have access to free compute resources.

### Can I train on a single GPU?

For research purposes, you can train on a single GPU by:
- Using smaller models (1B)
- Reducing batch size
- Using gradient accumulation
- Enabling gradient checkpointing

Production training requires multiple GPUs.

### How do I resume training?

Training automatically resumes from the latest checkpoint if you rerun the same command:

```bash
# Same command as original training
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
    --save_folder=./checkpoints
```

### Can I finetune on my own data?

Yes! Create a custom dataset class and add it to your training mixture. See [Custom Datasets](datasets/custom_datasets.md) for details.

## Performance

### How fast is inference?

On A100 40GB GPU:
- **Molmo-1B:** ~150 tokens/sec (batch=1)
- **Molmo-7B:** ~40 tokens/sec (batch=1)
- **Molmo-72B:** ~5 tokens/sec (batch=1)

Use vLLM for faster production serving.

### How can I speed up inference?

1. Use smaller models (MolmoE-1B)
2. Enable quantization (INT8/INT4)
3. Use vLLM for serving
4. Batch multiple requests
5. Cache image features
6. Use lower resolution images

### Why is training slow?

Common causes:
- Data loading bottleneck (increase num_workers)
- Slow storage (use local SSD)
- Small batch size (increase if memory allows)
- Too much activation checkpointing
- Network latency (use local or fast remote storage)

### Out of memory errors?

Solutions:
- Reduce batch size
- Enable/increase gradient checkpointing
- Use FSDP with FULL_SHARD
- Lower image resolution
- Use gradient accumulation
- Use smaller model

## Datasets

### Where can I download datasets?

```bash
python scripts/download_data.py all --n_proc 12
```

See [Data Preparation](guides/data_preparation.md) for details.

### How much storage do I need?

- All datasets: ~2TB
- PixMo only: ~500GB
- Academic benchmarks only: ~100GB

### Are the datasets free to use?

PixMo datasets are released by AI2. Academic benchmarks have their own licenses. Check individual dataset documentation for license details.

### Can I use my own datasets?

Yes! Implement the Dataset interface and add your data to the training mixture. See [Custom Datasets](datasets/custom_datasets.md).

## Deployment

### How do I deploy Molmo to production?

Options:
1. **vLLM:** High-performance serving (`vllm serve allenai/Molmo-7B-D-0924`)
2. **Modal:** Serverless deployment (see `scripts/serving/`)
3. **Custom:** Build your own API with FastAPI/Flask

See [Deployment Guide](guides/deployment_guide.md) for details.

### Can I use HuggingFace Inference API?

Yes, Molmo models on HuggingFace Hub support the Inference API.

### What about quantization?

Molmo supports:
- INT8 quantization (minimal quality loss)
- INT4 quantization (some quality loss, 4x smaller)
- GPTQ
- AWQ

### Can I run Molmo on edge devices?

Smaller models (1B) can run on edge devices with:
- Quantization (INT4/INT8)
- Optimized inference engines (TFLite, ONNX)
- Reduced resolution
- Careful memory management

## Troubleshooting

### Import errors?

```bash
pip install -e .[all] --force-reinstall
```

### CUDA out of memory?

Use smaller batch size, enable gradient checkpointing, or use a smaller model.

### Training loss is NaN?

- Reduce learning rate
- Increase warmup steps
- Use BF16 instead of FP16
- Enable gradient clipping

### Poor model quality?

- Train longer
- Use larger model
- Increase batch size
- Check data quality
- Try different hyperparameters

### Slow data loading?

- Increase num_workers
- Use faster storage (SSD)
- Enable HF_DATASETS_OFFLINE=1
- Preprocess data in advance

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Adding documentation

### Where do I report bugs?

Open an issue on [GitHub Issues](https://github.com/allenai/molmo/issues).

### How do I cite Molmo?

```bibtex
@article{molmo2024,
  title={Molmo: Multimodal Open Language Model},
  author={Deitke, Matt and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```

## Getting Help

### Where can I ask questions?

- **GitHub Discussions:** [Community Q&A](https://github.com/allenai/molmo/discussions)
- **GitHub Issues:** [Bug reports](https://github.com/allenai/molmo/issues)
- **Documentation:** Comprehensive guides and API docs

### Is there a Discord/Slack?

Check the [GitHub repository](https://github.com/allenai/molmo) for community links.

### Can I get commercial support?

Contact AI2 for commercial inquiries and support options.

## License

### What is the license?

Molmo is released under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

### Can I use Molmo commercially?

Yes, the Apache 2.0 license permits commercial use.

### What about the datasets?

PixMo datasets have their own licenses. Check individual dataset documentation.

## See Also

- [Documentation Home](index.md)
- [Installation Guide](guides/installation.md)
- [Quick Start](guides/quickstart.md)
- [Training Guide](guides/training_guide.md)
- [Troubleshooting](reference/troubleshooting.md)


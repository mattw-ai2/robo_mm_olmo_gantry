# Changelog

All notable changes to Molmo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Documentation
- Comprehensive documentation added
  - Architecture documentation
  - User guides (installation, training, evaluation, deployment)
  - API reference
  - Dataset documentation
  - FAQ and troubleshooting

## [0.1.0] - 2024-09-XX

### Added
- Initial public release of Molmo
- Multiple model variants: Molmo-1B, Molmo-7B, Molmo-72B, MolmoE-1B
- Support for multiple vision encoders (CLIP, SigLIP, DINOv2)
- Support for multiple LLM backbones (OLMo, Qwen2, Llama)
- PixMo dataset family release
- Training scripts for caption pretraining and multitask finetuning
- Evaluation framework for 20+ benchmarks
- HuggingFace integration
- vLLM support
- Remote storage support (GCS, S3, Weka)
- Distributed training with FSDP
- Mixed precision training (BF16/FP16)
- Gradient checkpointing
- Comprehensive evaluation suite

### Vision Encoders
- OpenAI CLIP
- SigLIP
- DINOv2
- MetaCLIP

### LLM Backbones
- OLMo (1B, 7B)
- Qwen2 (0.5B-72B)
- Llama (7B, 13B, 70B)

### Datasets
- PixMo-Cap: Dense captions
- PixMo-Points: Pointing annotations
- PixMo-Count: Counting annotations
- PixMo-Docs: Document understanding
- PixMo-CapQA: Caption-based QA
- Support for 20+ academic benchmarks

### Training Features
- Dense caption pretraining pipeline
- Multitask finetuning pipeline
- Distributed training with FSDP
- Mixed precision (BF16/FP16)
- Gradient checkpointing
- Learning rate scheduling
- Gradient clipping
- Checkpoint management
- WandB integration
- Remote storage support

### Evaluation Features
- VQA tasks (VQA2, TextVQA, DocQA, etc.)
- Chart/figure understanding (ChartQA, AI2D)
- Mathematical reasoning (MathVista)
- Document understanding
- Counting tasks
- Pointing/grounding tasks
- Video understanding (VideoMME, MLVU)
- Automated metric computation
- HTML visualizations
- Test set support

### Deployment
- HuggingFace Hub integration
- vLLM support for fast serving
- Modal deployment scripts
- Docker support
- API examples

### Developer Tools
- Comprehensive test suite
- Data visualization tools
- Dataset download scripts
- Model conversion utilities
- Checkpoint utilities

## Version History

### Model Releases

**Molmo-7B-D-0924** (September 2024)
- 7B parameter dense model
- CLIP vision encoder
- Strong performance across benchmarks

**Molmo-7B-O-0924** (September 2024)
- 7B parameter optimized variant
- Improved efficiency

**Molmo-72B-0924** (September 2024)
- 72B parameter large model
- Best performance on all benchmarks

**MolmoE-1B-0924** (September 2024)
- 1B active parameters (7B total with MoE)
- Efficient inference
- Strong performance for size

### Dataset Releases

**PixMo-v1** (September 2024)
- PixMo-Cap: 1.2M dense captions
- PixMo-Points: 500K pointing annotations
- PixMo-Count: 200K counting annotations
- PixMo-Docs: 300K document QA pairs
- PixMo-CapQA: 800K caption-based QA
- PixMo-AskModelAnything: 400K diverse questions
- PixMo-Clocks: 50K clock reading examples

## Future Plans

### Upcoming Features
- [ ] INT4/INT8 quantization support
- [ ] ONNX export
- [ ] Mobile deployment
- [ ] Multi-turn conversations
- [ ] Video captioning
- [ ] Real-time inference optimizations
- [ ] More vision encoders
- [ ] More LLM backbones
- [ ] Additional datasets

### Research Directions
- [ ] Improved video understanding
- [ ] Better instruction following
- [ ] Multilingual support
- [ ] 3D understanding
- [ ] Embodied AI integration
- [ ] Tool use and agent capabilities

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Links

- **Repository:** https://github.com/allenai/molmo
- **Paper:** https://arxiv.org/abs/2409.17146
- **Models:** https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19
- **Datasets:** https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b
- **Blog:** https://molmo.allenai.org/blog


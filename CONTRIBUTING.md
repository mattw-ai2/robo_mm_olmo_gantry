# Contributing to Molmo

Thank you for your interest in contributing to Molmo! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (for training and inference)
- Git

### Setting Up Your Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/allenai/molmo.git
   cd molmo
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install PyTorch first (adjust for your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   
   # Install Molmo with all dependencies including dev tools
   pip install -e .[all]
   ```

4. **Set up environment variables:**
   ```bash
   export MOLMO_DATA_DIR=/path/to/data
   export HF_HOME=/path/to/huggingface/cache
   ```

### Running Tests

Run the test suite to ensure your environment is set up correctly:

```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/data/test_preprocessor.py
```

## Code Style and Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines with some modifications:

- **Line length:** Maximum 100 characters (soft limit), 120 (hard limit)
- **Imports:** Group imports as standard library, third-party, local
- **Naming conventions:**
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Code Formatting

We use `ruff` for linting and formatting:

```bash
# Check code style
ruff check olmo/

# Auto-fix issues
ruff check --fix olmo/
```

### Type Hints

- Use type hints for function signatures
- Use `Optional[Type]` for nullable parameters
- Use `List[Type]`, `Dict[Type, Type]` for container types
- Example:
  ```python
  def process_batch(
      data: List[Dict[str, Any]],
      batch_size: int = 32,
      device: Optional[torch.device] = None
  ) -> torch.Tensor:
      ...
  ```

### Docstrings

Use Google-style docstrings:

```python
def train_model(config: TrainConfig, model: Molmo) -> None:
    """Train a Molmo model with the given configuration.
    
    Args:
        config: Training configuration including hyperparameters and data settings.
        model: The model to train.
        
    Returns:
        None. Checkpoints are saved to config.save_folder.
        
    Raises:
        OLMoConfigurationError: If configuration is invalid.
        
    Example:
        >>> config = TrainConfig.load("configs/train.yaml")
        >>> model = Molmo(MolmoConfig())
        >>> train_model(config, model)
    """
    ...
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory mirroring the source structure
- Name test files with `test_` prefix: `test_preprocessor.py`
- Name test functions with `test_` prefix: `test_image_preprocessing()`
- Use descriptive test names that explain what is being tested

### Test Structure

```python
import pytest
from olmo.data.dataset import Dataset

class TestDataset:
    """Test suite for Dataset class."""
    
    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        dataset = Dataset()
        assert len(dataset) > 0
    
    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_dataset_splits(self, split):
        """Test dataset loading for different splits."""
        dataset = Dataset(split=split)
        assert dataset.split == split
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=olmo tests/

# Run specific test
pytest tests/data/test_preprocessor.py::TestPreprocessor::test_image_crop
```

## Pull Request Process

### Branch Naming

Use descriptive branch names:
- `feature/add-video-support` - New features
- `fix/memory-leak-in-dataloader` - Bug fixes
- `docs/update-training-guide` - Documentation updates
- `refactor/simplify-config-loading` - Code refactoring

### Commit Messages

Write clear, concise commit messages:
- Use present tense: "Add feature" not "Added feature"
- First line: Brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Related to #456"

Example:
```
Add support for video datasets

- Implement VideoDataset class
- Add video preprocessing pipeline
- Update documentation
- Add tests for video loading

Fixes #789
```

### Creating a Pull Request

1. **Fork and clone** the repository
2. **Create a branch** from `main`
3. **Make your changes** following the code style
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests** to ensure everything passes
7. **Commit your changes** with clear messages
8. **Push to your fork** and create a pull request
9. **Describe your changes** in the PR description:
   - What problem does it solve?
   - What is the approach?
   - Are there any breaking changes?
   - Screenshots/examples if applicable

### Pull Request Checklist

Before submitting, ensure:
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main
- [ ] Code has been reviewed by yourself

## Adding New Features

### Adding a New Dataset

1. **Create dataset class** in `olmo/data/`:
   ```python
   from olmo.data.dataset import DatasetBase
   
   class MyNewDataset(DatasetBase):
       def __init__(self, split: str):
           super().__init__(split)
           
       def load(self):
           # Load your data
           ...
           
       def get(self, item: int, rng):
           # Return a single example
           ...
   ```

2. **Add download method** if applicable:
   ```python
   @classmethod
   def download(cls, n_procs=1):
       # Download dataset files
       ...
   ```

3. **Register in `get_dataset.py`** if using the dataset registry

4. **Add tests** in `tests/data/`

5. **Update documentation** in `docs/datasets/`

### Adding a New Model Variant

1. **Create model config** extending `BaseModelConfig`
2. **Implement model class** extending `ModelBase`
3. **Add preprocessor** if needed
4. **Add launch script** in `launch_scripts/`
5. **Add tests**
6. **Update documentation**

### Adding a New Evaluation Task

1. **Create evaluator class** in `olmo/eval/evaluators.py`:
   ```python
   class MyTaskEval(Evaluator):
       def evaluate(self, predictions, ground_truth):
           # Compute metrics
           ...
   ```

2. **Add dataset** if needed in `olmo/data/`

3. **Update eval scripts** to support new task

4. **Add tests**

5. **Document the task** in `docs/guides/evaluation_guide.md`

## Documentation Standards

### Code Comments

- Use comments to explain **why**, not **what**
- Keep comments up-to-date with code changes
- Use TODO comments for temporary code:
  ```python
  # TODO(username): Implement caching for better performance
  ```

### API Documentation

- All public classes and functions must have docstrings
- Include type hints in signatures
- Provide usage examples for complex functions
- Document all parameters and return values

### Tutorial Notebooks

When adding Jupyter notebooks:
- Include clear explanations between code cells
- Use markdown cells for documentation
- Ensure all cells run in order
- Include output cells for reference
- Add notebook to `docs/tutorials/`

## Getting Help

- **Documentation:** Check the [documentation](docs/index.md)
- **Issues:** Search [existing issues](https://github.com/allenai/molmo/issues)
- **Discussions:** Use [GitHub Discussions](https://github.com/allenai/molmo/discussions)
- **Contact:** Reach out to the maintainers

## License

By contributing to Molmo, you agree that your contributions will be licensed under the Apache License 2.0.

## Acknowledgments

Thank you for contributing to Molmo! Your contributions help make this project better for everyone.


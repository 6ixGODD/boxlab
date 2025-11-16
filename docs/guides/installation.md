# Installation Guide

This guide covers all methods of installing BoxLab, from simple pip installation to development setup.

## Prerequisites

BoxLab requires Python 3.10 or higher. Check your Python version:

```bash
python --version
# or
python3 --version
```

## Installation Methods

### Option 1: Install from PyPI (Recommended)

The simplest way to install BoxLab for end users:

```bash
pip install boxlab
```

Verify installation:

```bash
boxlab --version
```

### Option 2: Install from Source

For the latest development version or to contribute:

#### Using Poetry (Recommended for Development)

1. **Install Poetry** (if not already installed):

```bash
pip install pipx
pipx install poetry
```

2. **Clone the repository**:

```bash
git clone https://github.com/6ixGODD/boxlab.git
cd boxlab
```

3. **Install dependencies**:

```bash
# Basic installation
poetry install

# With development tools
poetry install --extras dev

# With testing tools
poetry install --extras test
```

4. **Activate the virtual environment**:

```bash
poetry shell  # Need to install shell plugin. See https://python-poetry.org/docs/managing-environments/#activating-the-environment
```

5. **Verify installation**:

```bash
poetry run python -m boxlab --version
```

#### Using pip

1. **Clone the repository**:

```bash
git clone https://github.com/6ixGODD/boxlab.git
cd boxlab
```

2. **Install in editable mode**:

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With testing tools
pip install -e ".[test]"

# With all extras
pip install -e ".[dev,test]"
```

3. **Verify installation**:

```bash
boxlab --version
```

## Optional Dependencies

### PyTorch Integration

For PyTorch integration and training workflows:

```bash
# Install PyTorch (CPU)
pip install torch torchvision

# Install PyTorch (GPU - CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch (GPU - CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

After installing PyTorch, you can use the PyTorch adapter:

```python
from boxlab.dataset.torchadapter import build_torchdataset
```

### GUI Dependencies

The annotator GUI requires additional packages (usually installed automatically):

```bash
pip install pillow matplotlib
```

## Development Setup

For contributors and developers:

### 1. Install Development Tools

```bash
# Using Poetry
poetry install --extras dev

# Using pip
pip install -e ".[dev]"
```

This installs:

- **mkdocs** - Documentation generation
- **mkdocs-material** - Documentation theme
- **mkdocstrings** - API documentation
- **pre-commit** - Git hooks
- **ruff** - Linting and formatting

### 2. Install Testing Tools

```bash
# Using Poetry
poetry install --extras test

# Using pip
pip install -e ".[test]"
```

This installs:

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **mypy** - Type checking
- **pylint** - Code analysis

### 3. Setup Pre-commit Hooks

```bash
pre-commit install
```

### 4. Run Tests

```bash
# Using Poetry
poetry run pytest

# Using pip
pytest

# With coverage
pytest --cov=boxlab --cov-report=html
```

### 5. Build Documentation

```bash
# Using Poetry
poetry run mkdocs serve

# Using pip
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view documentation.

## Running BoxLab

After installation, you can run BoxLab in multiple ways:

### As a Command

```bash
boxlab --help
```

### As a Python Module

```bash
python -m boxlab --help
```

### In Python Scripts

```python
from boxlab.dataset.io import load_dataset

dataset = load_dataset("path/to/dataset", format="coco")
```

## Troubleshooting

### Command Not Found

If `boxlab` command is not found after installation:

1. **Check if it's in PATH**:

```bash
which boxlab
```

2. **Use full path** (if installed with pip):

```bash
python -m boxlab --help
```

3. **Reinstall with pip**:

```bash
pip uninstall boxlab
pip install boxlab
```

### Import Errors

If you get import errors:

1. **Verify installation**:

```bash
pip show boxlab
```

2. **Check Python version**:

```bash
python --version
```

3. **Reinstall dependencies**:

```bash
pip install --force-reinstall boxlab
```

### Poetry Issues

If Poetry gives errors:

1. **Update Poetry**:

```bash
poetry self update
```

2. **Clear cache**:

```bash
poetry cache clear pypi --all
```

3. **Remove lock file and reinstall**:

```bash
rm poetry.lock
poetry install
```

## Upgrading

### Upgrade from PyPI

```bash
pip install --upgrade boxlab
```

### Upgrade from Source

```bash
cd boxlab
git pull
poetry install
# or
pip install -e .
```

## Uninstallation

### Uninstall with pip

```bash
pip uninstall boxlab
```

### Uninstall with Poetry

```bash
# Remove virtual environment
poetry env remove python

# Or just uninstall
pip uninstall boxlab
```

## System Requirements

### Minimum Requirements

- Python 3.10+
- 2GB RAM
- 500MB disk space

### Recommended Requirements

- Python 3.11+
- 4GB+ RAM
- 1GB+ disk space
- GPU (for PyTorch training)

### Supported Platforms

- Linux (Ubuntu 20.04+, Debian 10+, CentOS 7+)
- macOS (10.15+)
- Windows (10+)

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started with BoxLab
- [CLI Overview](cli-overview.md) - Learn the command-line interface

## Getting Help

- **Documentation**: [Full documentation](../index.md)
- **GitHub Issues**: [Report problems](https://github.com/6ixGODD/boxlab/issues)
- **Discussions**: [Ask questions](https://github.com/6ixGODD/boxlab/discussions)

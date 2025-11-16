# BoxLab

A comprehensive Python toolkit for managing, converting, and annotating object detection datasets with support for COCO and YOLO formats.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-boxlab-orange.svg)](https://pypi.org/project/boxlab/)

## Features

- **Dataset Management** - Load, merge, split, and analyze datasets with multi-source support
- **Format Conversion** - Seamlessly convert between COCO and YOLO formats
- **GUI Annotator** - Interactive desktop application for viewing and editing annotations
- **CLI Tools** - Powerful command-line interface for batch operations
- **PyTorch Integration** - Direct integration with PyTorch training pipelines
- **Plugin System** - Extensible architecture for custom format support

## Installation

### From PyPI (Recommended)

```bash
pip install boxlab
```

### From Source

```bash
# Clone repository
git clone https://github.com/6ixGODD/boxlab.git
cd boxlab

# Install with Poetry (recommended)
poetry install

# Or install with pip
pip install -e .
```

### Optional Dependencies

For PyTorch integration:

```bash
pip install torch torchvision
```

## Quick Start

### View Dataset Information

```bash
boxlab dataset info data/coco/annotations.json --format coco
```

### Convert Between Formats

```bash
# COCO to YOLO
boxlab dataset convert input.json -if coco output -of yolo

# YOLO to COCO
boxlab dataset convert data/yolo -if yolo output -of coco
```

### Merge Multiple Datasets

```bash
boxlab dataset merge \
  -i dataset1/ann.json coco source1 \
  -i dataset2/ann.json coco source2 \
  -o merged_dataset
```

### Launch GUI Annotator

```bash
boxlab annotator
```

### Python API

```python
from boxlab.dataset.io import load_dataset, export_dataset

# Load dataset
dataset = load_dataset("annotations.json", format="coco")

# Get statistics
stats = dataset.get_statistics()
print(f"Images: {stats['num_images']}")
print(f"Annotations: {stats['num_annotations']}")

# Export to different format
export_dataset(dataset, "output/yolo", format="yolo")
```

## CLI Commands

### Dataset Operations

```bash
# View information
boxlab dataset info <path> --format <coco|yolo>

# Convert formats
boxlab dataset convert <input> -if <format> <output> -of <format>

# Merge datasets
boxlab dataset merge -i <path> <format> [name] -o <output>

# Visualize dataset
boxlab dataset visualize <path> --format <format> -o <output>
```

### Annotator

```bash
# Launch GUI application
boxlab annotator
```

## PyTorch Integration

```python
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader

# Load and prepare dataset
dataset = load_dataset("train.json", format="coco")

# Create PyTorch dataset with augmentation
torch_dataset = build_torchdataset(
    dataset,
    image_size=640,
    augment=True,
    normalize=True,
    return_format="xyxy"
)

# Create DataLoader
train_loader = DataLoader(
    torch_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=torch_dataset.collate
)

# Training loop
for images, targets in train_loader:
    # Your training code here
    pass
```

## Custom Plugin Development

### Custom Loader

```python
from boxlab.dataset.plugins import LoaderPlugin
from boxlab.dataset import Dataset

class CustomLoader(LoaderPlugin):
    @property
    def name(self) -> str:
        return "custom_format"

    @property
    def description(self) -> str:
        return "Custom format loader"

    def load(self, path, **kwargs):
        dataset = Dataset(name="custom")
        # Your loading logic here
        return dataset

# Register and use
from boxlab.dataset.plugins.registry import register_loader
register_loader("custom_format", CustomLoader)
```

## Documentation

Full documentation is available at: [https://6ixgodd.github.io/boxlab](https://6ixgodd.github.io/boxlab)

- [Installation Guide](https://6ixgodd.github.io/boxlab/guides/installation/)
- [Quick Start Tutorial](https://6ixgodd.github.io/boxlab/guides/quick-start/)
- [CLI Reference](https://6ixgodd.github.io/boxlab/guides/cli-overview/)
- [API Reference](https://6ixgodd.github.io/boxlab/reference/)
- [Advanced Usage](https://6ixgodd.github.io/boxlab/guides/advanced-usage/)

## Examples

### Convert with Custom Split Ratio

```bash
boxlab dataset convert annotations.json \
  -if coco \
  output/yolo \
  -of yolo \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

### Merge with Source Tracking

```bash
boxlab dataset merge \
  -i manual_annotations.json coco manual \
  -i automatic_annotations.json coco automatic \
  -o combined_dataset \
  --preserve-sources
```

### Visualize with Heatmap

```bash
boxlab dataset visualize data/yolo \
  --format yolo \
  -o visualizations \
  --samples 10 \
  --show-heatmap \
  --show-source-dist
```

## Requirements

- Python 3.10 or higher
- NumPy
- Pillow
- Matplotlib
- PyYAML
- Pandas

### Optional

- PyTorch (for training integration)
- torchvision (for training integration)

## Project Structure

```
boxlab/
├── boxlab/
│   ├── dataset/            # Core dataset management
│   │   ├── plugins/        # Format plugins (COCO, YOLO)
│   │   ├── io.py           # I/O operations
│   │   ├── types.py        # Data structures
│   │   └── torchadapter.py # PyTorch integration
│   ├── annotator/          # GUI application
│   ├── cli/                # Command-line interface
│   └── exceptions.py       # Error handling
├── docs/                   # Documentation
├── tests/                  # Test suite
└── pyproject.toml          # Project configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/6ixGODD/boxlab.git
cd boxlab

# Install with development and test dependencies
poetry install --all-extras

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Build documentation
poetry run mkdocs serve
```

### Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Support

- **Documentation**: [https://6ixgodd.github.io/boxlab](https://6ixgodd.github.io/boxlab)
- **Issues**: [GitHub Issues](https://github.com/6ixGODD/boxlab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/6ixGODD/boxlab/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

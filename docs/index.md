# BoxLab Documentation

Welcome to BoxLab - A Python toolkit for managing, converting, and annotating object detection datasets with support for COCO and YOLO formats.

## What is BoxLab?

BoxLab is a comprehensive solution for working with object detection datasets. It provides:

- **Dataset Management** - Load, merge, split, and analyze datasets
- **Format Conversion** - Seamlessly convert between COCO and YOLO formats
- **GUI Annotator** - Interactive desktop application for viewing and editing annotations
- **CLI Tools** - Powerful command-line interface for batch operations
- **PyTorch Integration** - Direct integration with PyTorch training pipelines
- **Plugin System** - Extensible architecture for custom formats

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Getting Started**

    ---

    New to BoxLab? Start here!

    [:octicons-arrow-right-24: Installation Guide](guides/installation.md)

    [:octicons-arrow-right-24: Quick Start](guides/quick-start.md)

-   :material-console: **CLI Reference**

    ---

    Command-line interface documentation

    [:octicons-arrow-right-24: CLI Overview](guides/cli-overview.md)

    [:octicons-arrow-right-24: Dataset Commands](guides/cli-dataset.md)

-   :material-code-tags: **API Reference**

    ---

    Complete Python API documentation

    [:octicons-arrow-right-24: API Reference](reference/index.md)

    [:octicons-arrow-right-24: Dataset API](reference/dataset/index.md)

-   :material-application-outline: **GUI Annotator**

    ---

    Interactive annotation application

    [:octicons-arrow-right-24: Annotator Guide](guides/annotator-guide.md)

    [:octicons-arrow-right-24: CLI Command](guides/cli-annotator.md)

</div>

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

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

See the [Installation Guide](guides/installation.md) for detailed instructions.

## Quick Start

### View Dataset Info

```bash
boxlab dataset info data/coco/annotations.json --format coco
```

### Convert Format

```bash
# COCO to YOLO
boxlab dataset convert input.json -if coco output -of yolo

# YOLO to COCO
boxlab dataset convert data/yolo -if yolo output -of coco
```

### Launch Annotator

```bash
boxlab annotator
```

### Python API

```python
from boxlab.dataset.io import load_dataset, export_dataset

# Load dataset
dataset = load_dataset("annotations.json", format="coco")

# Export to different format
export_dataset(dataset, "output/yolo", format="yolo")
```

See the [Quick Start Guide](guides/quick-start.md) for more examples.

## Features

### Dataset Management

- **Multi-format Support** - COCO JSON and YOLO formats
- **Source Tracking** - Track dataset origins in merged datasets
- **Statistics** - Comprehensive dataset analysis
- **Visualization** - Generate distribution plots and sample images

### Format Conversion

- **Bidirectional** - Convert between COCO and YOLO
- **Flexible Splitting** - Custom train/val/test ratios
- **Naming Strategies** - Multiple file naming options
- **Validation** - Automatic format validation

### Annotation Tools

- **Visual Editor** - Interactive bounding box editing
- **Audit Workflow** - Approve/reject images systematically
- **Tagging System** - Organize images with custom tags
- **Workspace Persistence** - Save and restore work sessions

### Command Line Interface

- **Intuitive Commands** - Easy-to-use CLI structure
- **Rich Output** - Formatted tables and progress indicators
- **Batch Operations** - Process multiple datasets
- **Scriptable** - Integration with automation workflows

### PyTorch Integration

- **Dataset Adapter** - Direct PyTorch Dataset compatibility
- **Transform Support** - Built-in augmentation pipelines
- **DataLoader Ready** - Custom collate functions
- **Training Workflows** - Seamless integration with training loops

## Use Cases

### Format Conversion

Convert your existing datasets to the format required by your training framework:

```bash
boxlab dataset convert coco_annotations.json -if coco yolo_output -of yolo
```

### Dataset Merging

Combine multiple annotation sources into a single unified dataset:

```bash
boxlab dataset merge \
  -i manual_labels.json coco manual \
  -i auto_labels.json coco automatic \
  -o merged_dataset
```

### Quality Assurance

Use the annotator to review and audit dataset quality:

```bash
boxlab annotator
# Enable Audit Mode → Review images → Export report
```

### Training Preparation

Prepare datasets for model training with PyTorch:

```python
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader

dataset = load_dataset("train.json", format="coco")
torch_dataset = build_torchdataset(dataset, image_size=640, augment=True)
loader = DataLoader(torch_dataset, batch_size=16, collate_fn=torch_dataset.collate)
```

## Documentation Structure

### Guides

Step-by-step tutorials and conceptual guides:

- [Installation](guides/installation.md) - Setup and installation
- [Quick Start](guides/quick-start.md) - Get started in 5 minutes
- [CLI Overview](guides/cli-overview.md) - Command-line interface basics
- [Dataset Commands](guides/cli-dataset.md) - Dataset operation reference
- [Annotator Command](guides/cli-annotator.md) - GUI application usage

### API Reference

Complete technical documentation:

- [API Overview](reference/index.md) - API documentation index
- [Dataset Core](reference/dataset/index.md) - Dataset management
- [I/O Operations](reference/dataset/io.md) - Loading and exporting
- [Types](reference/dataset/types.md) - Data structures
- [Plugin System](reference/dataset/plugins/index.md) - Extensibility
- [PyTorch Adapter](reference/dataset/torchadapter.md) - Training integration
- [Exceptions](reference/exceptions.md) - Error handling

## Examples

### Convert COCO to YOLO with Split

```bash
boxlab dataset convert \
  annotations.json \
  -if coco \
  output/yolo \
  -of yolo \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

### Merge Three Datasets

```bash
boxlab dataset merge \
  -i dataset1/ann.json coco source1 \
  -i dataset2/ann.json coco source2 \
  -i dataset3 yolo source3 \
  -o merged_output \
  --output-format coco
```

### Visualize Dataset

```bash
boxlab dataset visualize \
  data/yolo \
  --format yolo \
  -o visualizations \
  --samples 10 \
  --show-heatmap
```

### PyTorch Training Loop

```python
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader
import torch

# Prepare dataset
dataset = load_dataset("train_annotations.json", format="coco")
train_dataset = build_torchdataset(
    dataset,
    image_size=640,
    augment=True,
    normalize=True,
    return_format="xyxy"
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate
)

# Training loop
model = YourDetectionModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
```

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, Windows
- **RAM**: 2GB minimum, 4GB recommended
- **Disk Space**: 500MB for installation

### Optional Dependencies

- **PyTorch**: For training integration (`pip install torch torchvision`)
- **CUDA**: For GPU acceleration (with PyTorch GPU version)

## Project Information

- **GitHub**: [github.com/6ixGODD/boxlab](https://github.com/6ixGODD/boxlab)
- **PyPI**: [pypi.org/project/boxlab](https://pypi.org/project/boxlab)
- **License**: MIT
- **Author**: BoChenSHEN (6ixGODD)

## Getting Help

### Documentation

- Browse the [Guides](guides/index.md) for tutorials
- Check the [API Reference](reference/index.md) for detailed documentation
- Review [Common Issues](guides/troubleshooting.md) for solutions

### Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/6ixGODD/boxlab/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/6ixGODD/boxlab/discussions)

### Contributing

Contributions are welcome! Please see:

- [Contributing Guidelines](contributing.md)
- [Code of Conduct](code-of-conduct.md)
- [Development Setup](guides/installation.md#development-setup)

## What's Next?

<div class="grid cards" markdown>

-   :material-clock-fast: **New Users**

    ---

    Follow the installation and quick start guides

    [:octicons-arrow-right-24: Installation](guides/installation.md)

    [:octicons-arrow-right-24: Quick Start](guides/quick-start.md)

-   :material-console-line: **CLI Users**

    ---

    Learn the command-line interface

    [:octicons-arrow-right-24: CLI Overview](guides/cli-overview.md)

    [:octicons-arrow-right-24: Dataset Commands](guides/cli-dataset.md)

-   :material-language-python: **Python Developers**

    ---

    Explore the Python API

    [:octicons-arrow-right-24: API Reference](reference/index.md)

    [:octicons-arrow-right-24: PyTorch Integration](reference/dataset/torchadapter.md)

-   :material-pencil-ruler: **Annotators**

    ---

    Use the GUI application

    [:octicons-arrow-right-24: Annotator Command](guides/cli-annotator.md)

</div>

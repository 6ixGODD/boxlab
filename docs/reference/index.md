# API Reference

Welcome to the BoxLab API Reference documentation. This section provides comprehensive technical documentation for all
public APIs, classes, and functions in the BoxLab library.

## Overview

BoxLab is a Python library for managing, processing, and annotating object detection datasets. It provides a unified
interface for working with multiple dataset formats (COCO, YOLO) and includes tools for dataset manipulation,
visualization, and PyTorch integration.

## Quick Navigation

### Core Components

#### [Dataset Management](dataset/index.md)

The core `Dataset` class for managing images, annotations, and categories with multi-source support.

- **Key Classes**: `Dataset`
- **Use Cases**: Loading datasets, managing annotations, merging datasets, statistics
- **Go to**: [Dataset API →](dataset/index.md)

#### [I/O Operations](dataset/io.md)

High-level functions for loading, exporting, and merging datasets with automatic format detection.

- **Key Functions**: `load_dataset()`, `export_dataset()`, `merge()`
- **Use Cases**: Format conversion, batch processing, dataset operations
- **Go to**: [I/O API →](dataset/io.md)

#### [Types](dataset/types.md)

Core data structures including bounding boxes, annotations, and image metadata.

- **Key Types**: `BBox`, `Annotation`, `ImageInfo`, `SplitRatio`
- **Use Cases**: Type annotations, data validation, coordinate conversions
- **Go to**: [Types API →](dataset/types.md)

### Plugin System

#### [Plugin Architecture](dataset/plugins/index.md)

Extensible plugin system for custom loaders and exporters.

- **Key Classes**: `LoaderPlugin`, `ExporterPlugin`, `NamingStrategy`
- **Use Cases**: Custom format support, extending functionality
- **Go to**: [Plugins API →](dataset/plugins/index.md)

#### [Plugin Registry](dataset/plugins/registry.md)

Registration and discovery system for dataset format plugins.

- **Key Functions**: `register_loader()`, `get_loader()`, `list_loaders()`
- **Use Cases**: Plugin management, format discovery
- **Go to**: [Registry API →](dataset/plugins/registry.md)

#### Format Plugins

- **[COCO Plugin](dataset/plugins/coco.md)** - COCO format support with JSON I/O
- **[YOLO Plugin](dataset/plugins/yolo.md)** - YOLO format support with YAML configuration

### Integration

#### [PyTorch Adapter](dataset/torchadapter.md)

Seamless integration with PyTorch for training object detection models.

- **Key Classes**: `TorchDatasetAdapter`
- **Key Functions**: `build_torchdataset()`
- **Use Cases**: Model training, DataLoader integration, augmentation
- **Go to**: [PyTorch API →](dataset/torchadapter.md)

### GUI Application

#### [Annotator](annotator.md)

Desktop application for viewing, editing, and auditing datasets.

- **Key Class**: `AnnotatorApp`
- **Use Cases**: Visual annotation, dataset review, audit workflows
- **Go to**: [Annotator API →](annotator.md)

### Error Handling

#### [Exceptions](exceptions.md)

Comprehensive exception hierarchy for error handling.

- **Base Exception**: `BoxlabError`
- **Use Cases**: Error handling, debugging, validation
- **Go to**: [Exceptions API →](exceptions.md)

## Common Usage Patterns

### Loading Datasets

```python
from boxlab.dataset.io import load_dataset

# Auto-detect format
dataset = load_dataset("path/to/dataset")

# Explicit format
dataset = load_dataset("annotations.json", format="coco")
```

### Converting Formats

```python
from boxlab.dataset.io import load_dataset, export_dataset

# Load COCO
dataset = load_dataset("coco_annotations.json", format="coco")

# Export to YOLO
export_dataset(dataset, "output/yolo", format="yolo")
```

### PyTorch Training

```python
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("dataset/")

# Create PyTorch dataset
torch_dataset = build_torchdataset(
    dataset,
    image_size=640,
    augment=True,
    normalize=True
)

# Create DataLoader
loader = DataLoader(
    torch_dataset,
    batch_size=16,
    collate_fn=torch_dataset.collate
)
```

### Dataset Manipulation

```python
from boxlab.dataset import Dataset
from boxlab.dataset.io import merge

# Create datasets
ds1 = Dataset(name="dataset1")
ds2 = Dataset(name="dataset2")

# Merge
merged = merge(ds1, ds2, resolve_conflicts="skip")

# Split
from boxlab.dataset.types import SplitRatio

splits = merged.split(SplitRatio(train=0.7, val=0.2, test=0.1), seed=42)
```

## API Organization

### By Module

```
boxlab/
├── dataset/
│   ├── __init__.py          → Dataset class
│   ├── io.py                → I/O operations
│   ├── types.py             → Data structures
│   ├── torchadapter.py      → PyTorch integration
│   └── plugins/
│       ├── __init__.py      → Plugin base classes
│       ├── registry.py      → Plugin registry
│       ├── coco.py          → COCO plugin
│       └── yolo.py          → YOLO plugin
├── annotator/
│   └── __init__.py          → Annotator app
└── exceptions.py            → Exception classes
```

### By Use Case

#### Dataset Loading

- [load_dataset()](dataset/io.md#load_dataset)
- [LoaderPlugin](dataset/plugins/index.md#loaderplugin)
- [get_loader()](dataset/plugins/registry.md#get_loader)

#### Dataset Export

- [export_dataset()](dataset/io.md#export_dataset)
- [ExporterPlugin](dataset/plugins/index.md#exporterplugin)
- [get_exporter()](dataset/plugins/registry.md#get_exporter)

#### Dataset Manipulation

- [Dataset.merge()](dataset/index.md#merge)
- [Dataset.split()](dataset/index.md#split)
- [merge()](dataset/io.md#merge)

#### Statistics & Analysis

- [Dataset.get_statistics()](dataset/index.md#get_statistics)
- [Dataset.print_statistics()](dataset/index.md#print_statistics)
- [Dataset.visualize_category_distribution()](dataset/index.md#visualize_category_distribution)

#### PyTorch Integration

- [TorchDatasetAdapter](dataset/torchadapter.md#torchdatasetadapter)
- [build_torchdataset()](dataset/torchadapter.md#build_torchdataset)

## Type Hints

BoxLab provides comprehensive type hints for better IDE support and type checking:

```python
from boxlab.dataset import Dataset
from boxlab.dataset.types import BBox, Annotation, ImageInfo
from boxlab.dataset.io import load_dataset


def process_dataset(path: str) -> Dataset:
    dataset: Dataset = load_dataset(path)
    return dataset


def create_annotation(
    bbox: BBox,
    category_id: int,
    category_name: str,
    image_id: str
) -> Annotation:
    return Annotation(
        bbox=bbox,
        category_id=category_id,
        category_name=category_name,
        image_id=image_id
    )
```

## Error Codes

BoxLab uses numeric error codes for programmatic error handling:

| Code | Exception                   | Category        |
|------|-----------------------------|-----------------|
| 1    | BoxlabError                 | Base            |
| 2    | RequiredModuleNotFoundError | Dependencies    |
| 10   | DatasetError                | Dataset (base)  |
| 11   | DatasetNotFoundError        | Dataset I/O     |
| 12   | DatasetFormatError          | Dataset Format  |
| 13   | DatasetLoadError            | Dataset Loading |
| 14   | DatasetExportError          | Dataset Export  |
| 15   | DatasetMergeError           | Dataset Merge   |
| 16   | CategoryConflictError       | Dataset Merge   |
| 20   | ValidationError             | Validation      |

See [Exceptions](exceptions.md) for detailed error handling guide.

## Version Information

This documentation is for **BoxLab version 0.1.0** (as of 2025-01-16).

For the latest updates, see the [CHANGELOG](../changelog.md) and [GitHub repository](https://github.com/6ixGODD/boxlab).

## License

BoxLab is released under the MIT License. See [LICENSE](../license.md) for details.

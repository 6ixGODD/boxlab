# Dataset

::: boxlab.dataset.Dataset
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The `Dataset` class is the core component of BoxLab's dataset management system. It provides comprehensive functionality
for managing object detection datasets, including loading, exporting, merging, and analyzing data from multiple sources.

## Key Features

- **Multi-format Support**: Load and export datasets in COCO, YOLO, and other formats
- **Multi-source Management**: Track and manage data from multiple sources
- **Category Management**: Handle categories with conflict resolution
- **Dataset Operations**: Split, merge, and transform datasets
- **Statistics & Visualization**: Comprehensive dataset analysis and visualization tools
- **Flexible Architecture**: Plugin-based system for extending functionality

## Quick Start

```python
import pathlib

from boxlab.dataset import Dataset
from boxlab.dataset.types import ImageInfo, Annotation, BBox

# Create a new dataset
dataset = Dataset(name="my_dataset")

# Add categories
dataset.add_category(1, "person")
dataset.add_category(2, "car")

# Add an image
img_info = ImageInfo(
    image_id="001",
    file_name="image1.jpg",
    width=640,
    height=480,
    path=pathlib.Path("/path/to/image1.jpg"),
)
dataset.add_image(img_info, source_name="camera1")

# Add an annotation
annotation = Annotation(
    bbox=BBox(x_min=10, y_min=20, x_max=100, y_max=150),
    category_id=1,
    category_name="person",
    image_id="001",
    annotation_id="ann_001",
)
dataset.add_annotation(annotation)

# Get statistics
stats = dataset.get_statistics()
print(f"Total images: {stats['num_images']}")
print(f"Total annotations: {stats['num_annotations']}")
```

## Category Management

Manage object categories in your dataset:

- `add_category()`: Add a new category
- `get_category_name()`: Retrieve category name by ID
- `get_category_id()`: Retrieve category ID by name
- `fix_duplicate_categories()`: Resolve duplicate category names

## Image Management

Handle image metadata and sources:

- `add_image()`: Add image with optional source tracking
- `get_image()`: Retrieve image information
- `get_image_source()`: Get source name for an image
- `get_sources()`: List all unique data sources

## Annotation Management

Work with object annotations:

- `add_annotation()`: Add bounding box annotation
- `get_annotations()`: Get all annotations for an image

## Statistics & Analysis

Analyze your dataset:

- `get_statistics()`: Compute comprehensive statistics
- `print_statistics()`: Display statistics in console
- `num_images()`: Get image count
- `num_annotations()`: Get annotation count
- `num_categories()`: Get category count

## Dataset Operations

Transform and combine datasets:

- `split()`: Split dataset into train/val/test sets
- `merge()`: Combine multiple datasets
- `__add__()`: Merge using `+` operator

## Visualization

Visualize dataset content:

- `visualize_sample()`: Display image with annotations
- `visualize_category_distribution()`: Show category balance

## See Also

- [Plugin System](plugins/index.md): Extend dataset functionality
- [Types](types.md): Data structures and type definitions
- [I/O Operations](io.md) - Loading and exporting datasets
- [PyTorch Adapter](torchadapter.md) - Training integration

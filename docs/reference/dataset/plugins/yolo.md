# YOLO Plugin

::: boxlab.dataset.plugins.yolo
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The YOLO plugin provides support for loading and exporting datasets in YOLOv5/YOLOv8 format. It handles YAML
configuration files, normalized bounding box coordinates, and the standard YOLO directory structure.

## Format Specification

### Directory Structure

```
dataset/
├── data.yaml              # Configuration file
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images
└── labels/
    ├── train/            # Training labels
    ├── val/              # Validation labels
    └── test/             # Test labels
```

### YAML Configuration

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # Number of classes

names:
  0: person
  1: car
  2: bicycle
```

### Label Format

Each label file (`.txt`) contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1] range:

- `x_center`: Center X coordinate / image width
- `y_center`: Center Y coordinate / image height
- `width`: Bounding box width / image width
- `height`: Bounding box height / image height

## YOLOLoader

Load datasets from YOLO format.

### Basic Usage

```python
from boxlab.dataset.plugins.registry import get_loader

loader = get_loader("yolo")
dataset = loader.load("path/to/yolo_dataset")
```

### Load Specific Splits

```python
# Load only training data
dataset = loader.load("path/to/yolo_dataset", splits="train")

# Load multiple splits
dataset = loader.load("path/to/yolo_dataset", splits=["train", "val"])

# Load all splits (default)
dataset = loader.load("path/to/yolo_dataset", splits=None)
```

### Custom YAML File

```python
# Use custom YAML filename
dataset = loader.load(
    "path/to/yolo_dataset",
    yaml_file="custom.yaml"
)
```

### Features

- Supports both YOLOv5 and YOLOv8 formats
- Handles dict or list category definitions in YAML
- Converts normalized coordinates to absolute pixels
- Validates label file format
- Logs warnings for invalid annotations
- Supports multiple image formats (jpg, png, bmp, tiff, webp)

## YOLOExporter

Export datasets to YOLO format.

### Basic Usage

```python
from boxlab.dataset.plugins.registry import get_exporter

exporter = get_exporter("yolo")
exporter.export(dataset, output_dir="output/yolo_format")
```

### Export with Splits

```python
from boxlab.dataset.types import SplitRatio

# Define split ratios
split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

exporter.export(
    dataset,
    output_dir="output/yolo_format",
    split_ratio=split_ratio,
    seed=42  # For reproducibility
)
```

### Export Options

```python
from boxlab.dataset.plugins.naming import SequentialNaming

# Custom naming strategy
strategy = SequentialNaming(prefix="img", start=1, digits=6)

# Export with options
exporter.export(
    dataset,
    output_dir="output/yolo_format",
    split_ratio=split_ratio,
    seed=42,
    naming_strategy=strategy,
    copy_images=True,  # Copy image files
    unified_structure=False  # Use standard structure
)
```

### Unified Structure

Use unified directory structure (annotations instead of labels):

```python
exporter.export(
    dataset,
    output_dir="output/yolo_format",
    unified_structure=True  # Uses 'annotations' directory
)
```

Output structure:

```
output/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/          # Instead of 'labels'
    ├── train/
    ├── val/
    └── test/
```

### Features

- Generates compliant YAML configuration
- Converts absolute coordinates to normalized format
- Handles filename conflicts automatically
- Supports custom naming strategies
- Optional image copying
- 0-indexed class IDs in output (YOLO convention)
- Preserves annotation precision with 6 decimal places

## Coordinate Conversion

### Loading (Normalized → Absolute)

```python
# YOLO label: 0 0.5 0.5 0.3 0.2
# Image size: 640x480

cx_norm, cy_norm = 0.5, 0.5
w_norm, h_norm = 0.3, 0.2

cx = cx_norm * 640  # 320.0
cy = cy_norm * 480  # 240.0
w = w_norm * 640    # 192.0
h = h_norm * 480    # 96.0
```

### Exporting (Absolute → Normalized)

```python
# BBox: x_min=224, y_min=144, x_max=416, y_max=336
# Image size: 640x480

cx = (x_min + x_max) / 2  # 320.0
cy = (y_min + y_max) / 2  # 240.0
w = x_max - x_min          # 192.0
h = y_max - y_min          # 192.0

cx_norm = cx / 640  # 0.5
cy_norm = cy / 480  # 0.5
w_norm = w / 640    # 0.3
h_norm = h / 480    # 0.4
```

## Category ID Handling

YOLO uses 0-indexed category IDs, while BoxLab's Dataset uses 1-indexed IDs internally.

### During Loading

```python
# YOLO label: class_id = 0
# Internal: category_id = 1
category_id = yolo_class_id + 1
```

### During Export

```python
# Internal: category_id = 1
# YOLO label: class_id = 0
yolo_class_id = category_id - 1
```

## Error Handling

The YOLO plugin handles various error conditions:

- **Missing YAML**: Raises `FileNotFoundError`
- **Invalid YAML**: Raises `ValueError` if 'names' field is missing
- **Missing directories**: Logs warning and skips
- **Invalid label format**: Logs warning and skips line
- **Unknown category**: Logs warning and skips annotation
- **Image read errors**: Logs error and continues

## See Also

- [Plugin System](index.md): Plugin architecture overview
- [Registry](registry.md): Plugin registration and discovery
- [COCO Plugin](coco.md): COCO format plugin
- [Dataset](../index.md): Core dataset management

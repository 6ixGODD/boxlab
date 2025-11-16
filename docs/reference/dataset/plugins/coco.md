# COCO Plugin

::: boxlab.dataset.plugins.coco
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The COCO plugin provides support for loading and exporting datasets in COCO (Common Objects in Context) JSON format. It
handles the standard COCO annotation structure with support for images, categories, and bounding box annotations.

## Format Specification

### JSON Structure

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person"
    }
  ]
}
```

### Bounding Box Format

COCO uses `[x, y, width, height]` format where:

- `x`: Left coordinate (x_min)
- `y`: Top coordinate (y_min)
- `width`: Box width
- `height`: Box height

## COCOLoader

Load datasets from COCO format.

### Basic Usage

```python
from boxlab.dataset.plugins.registry import get_loader

loader = get_loader("coco")
dataset = loader.load("annotations/instances.json")
```

### Specify Image Root

```python
# Images located in different directory
dataset = loader.load(
    "annotations/instances.json",
    image_root="/path/to/images"
)
```

### Custom Dataset Name

```python
# Set custom dataset name
dataset = loader.load(
    "annotations/instances.json",
    name="my_coco_dataset"
)
```

### Features

- Parses standard COCO JSON format
- Converts COCO bbox format to internal representation
- Handles missing or invalid annotations gracefully
- Supports both relative and absolute image paths
- Validates image file existence
- Logs warnings for missing images or invalid data

## COCOExporter

Export datasets to COCO format.

### Basic Usage

```python
from boxlab.dataset.plugins.registry import get_exporter

exporter = get_exporter("coco")
exporter.export(dataset, output_dir="output/coco_format")
```

### Export with Splits

```python
from boxlab.dataset.types import SplitRatio

# Define split ratios
split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

exporter.export(
    dataset,
    output_dir="output/coco_format",
    split_ratio=split_ratio,
    seed=42  # For reproducibility
)
```

Output structure:

```
output/coco_format/
├── train.json
├── val.json
├── test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

### Export Options

```python
from boxlab.dataset.plugins.naming import SequentialNaming

# Custom naming strategy
strategy = SequentialNaming(prefix="img", start=1, digits=6)

# Export with options
exporter.export(
    dataset,
    output_dir="output/coco_format",
    split_ratio=split_ratio,
    seed=42,
    naming_strategy=strategy,
    copy_images=True,  # Copy image files
    indent=2  # JSON formatting
)
```

### Export Without Image Copying

```python
# Only export annotations
exporter.export(
    dataset,
    output_dir="output/coco_format",
    copy_images=False
)
```

### Features

- Generates compliant COCO JSON format
- Converts internal bbox format to COCO format
- Handles filename conflicts automatically
- Supports custom naming strategies
- Optional image copying
- Configurable JSON indentation
- Preserves annotation metadata (area, iscrowd)

## Coordinate Conversion

### Loading (COCO → Internal)

```python
# COCO bbox: [100, 50, 200, 150]
x, y, w, h = 100, 50, 200, 150

# Convert to internal format
x_min = x  # 100
y_min = y  # 50
x_max = x + w  # 300
y_max = y + h  # 200
```

### Exporting (Internal → COCO)

```python
# Internal bbox: x_min=100, y_min=50, x_max=300, y_max=200

x = x_min  # 100
y = y_min  # 50
w = x_max - x_min  # 200
h = y_max - y_min  # 150

# COCO bbox: [100, 50, 200, 150]
```

## Image Path Resolution

### During Loading

```python
# Annotation specifies: "file_name": "image1.jpg"
# image_root: "/data/images"

# Resolved path: /data/images/image1.jpg
image_path = Path(image_root) / file_name
```

### During Export

```python
# Original path: /data/source/subfolder/image1.jpg
# Naming strategy: OriginalNaming()

# Exported filename: image1.jpg
# Exported path: output_dir/images/train/image1.jpg
```

## Metadata Handling

### Annotation Metadata

```python
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 50, 200, 150],
  "area": 30000.0,     # Computed if not provided
  "iscrowd": 0         # Default: 0 (not crowd)
}
```

### Image Metadata

```python
{
  "id": 1,
  "file_name": "image1.jpg",
  "width": 640,
  "height": 480,
  "date_captured": "",  # Optional
  "license": 0,        # Optional
  "coco_url": "",      # Optional
  "flickr_url": ""     # Optional
}
```

### Category Metadata

```python
{
  "id": 1,
  "name": "person",
  "supercategory": ""  # Optional
}
```

## Error Handling

The COCO plugin handles various error conditions:

- **Missing JSON file**: Raises `FileNotFoundError`
- **Invalid JSON**: Raises `ValueError`
- **Missing images**: Logs warning and continues
- **Missing categories**: Logs warning and skips annotation
- **Invalid bbox format**: Logs warning and skips annotation
- **Duplicate IDs**: Handled automatically with ID remapping

## Advanced Usage

### Custom JSON Configuration

```python
# Export with custom JSON formatting
exporter.export(
    dataset,
    output_dir="output/coco_format",
    indent=4,  # Pretty print with 4 spaces
    ensure_ascii=False  # Allow Unicode characters
)
```

### Batch Export

```python
# Export multiple datasets to COCO format
datasets = [dataset1, dataset2, dataset3]

for i, ds in enumerate(datasets):
    exporter.export(
        ds,
        output_dir=f"output/coco_batch_{i}",
        copy_images=True
    )
```

## See Also

- [Plugin System](index.md): Plugin architecture overview
- [Registry](registry.md): Plugin registration and discovery
- [YOLO Plugin](yolo.md): YOLO format plugin
- [Dataset](../index.md): Core dataset management
- [COCO Format Specification](https://cocodataset.org/#format-data): Official COCO documentation

# Types

::: boxlab.dataset.types
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The types module defines core data structures used throughout BoxLab. These types provide type-safe, immutable
representations of bounding boxes, annotations, images, and dataset metadata.

## Bounding Box Formats

BoxLab supports three common bounding box coordinate formats:

### XYXY Format (Internal)

Top-left and bottom-right corners:

```
(x_min, y_min, x_max, y_max)
```

- Most common in detection frameworks
- Used internally by BoxLab
- Direct representation of box corners

### XYWH Format (COCO)

Top-left corner and dimensions:

```
(x, y, width, height)
```

- Used by COCO dataset
- Convenient for drawing operations
- Common in visualization tools

### CXCYWH Format (YOLO)

Center point and dimensions:

```
(center_x, center_y, width, height)
```

- Used by YOLO formats
- Natural for certain augmentations
- Often used with normalized coordinates

## Format Conversions

### BBox Conversions

```python
from boxlab.dataset.types import BBox

# Create in XYXY (internal format)
bbox = BBox(x_min=10, y_min=20, x_max=100, y_max=150)

# Convert to other formats
xyxy = bbox.xyxy  # (10, 20, 100, 150)
xywh = bbox.xywh  # (10, 20, 90, 130)
cxcywh = bbox.cxcywh  # (55.0, 85.0, 90, 130)

# Create from other formats
bbox1 = BBox.from_xywh(x=10, y=20, w=90, h=130)
bbox2 = BBox.from_cxcywh(cx=55, cy=85, w=90, h=130)

# All represent the same box
assert bbox == bbox1 == bbox2
```

## Usage Patterns

### Creating Annotations

```python
from boxlab.dataset.types import BBox, Annotation

# Complete annotation
annotation = Annotation(
    bbox=BBox(10, 20, 100, 150),
    category_id=1,
    category_name="person",
    image_id="img_001",
    annotation_id="ann_001",
    area=11700.0,
    iscrowd=0
)

# Minimal annotation (area computed automatically)
annotation = Annotation(
    bbox=BBox(10, 20, 100, 150),
    category_id=1,
    category_name="person",
    image_id="img_001"
)
```

### Image Metadata

```python
from pathlib import Path
from boxlab.dataset.types import ImageInfo

# With file path
img_info = ImageInfo(
    image_id="001",
    file_name="photo.jpg",
    width=1920,
    height=1080,
    path=Path("/data/images/photo.jpg")
)

# Without path (for annotations-only export)
img_info = ImageInfo(
    image_id="001",
    file_name="photo.jpg",
    width=1920,
    height=1080
)
```

### Dataset Splitting

```python
from boxlab.dataset.types import SplitRatio

# Standard split
split = SplitRatio(train=0.7, val=0.2, test=0.1)
split.validate()

# No test set
split = SplitRatio(train=0.8, val=0.2, test=0.0)
split.validate()

# Use with dataset
splits = dataset.split(split, seed=42)
```

### Analyzing Statistics

```python
from boxlab.dataset import Dataset

dataset = Dataset(name="my_dataset")
# ... populate dataset ...

stats = dataset.get_statistics()

# Access statistics
print(f"Total images: {stats['num_images']}")
print(f"Total annotations: {stats['num_annotations']}")

# Category analysis
dist = stats['category_distribution']
most_common = max(dist.items(), key=lambda x: x[1])
print(f"Most common: {most_common[0]} ({most_common[1]} instances)")

# Annotation density
avg = stats['avg_annotations_per_image']
std = stats['std_annotations_per_image']
print(f"Objects per image: {avg:.2f} ± {std:.2f}")

# Box size analysis
print(f"Average box area: {stats['avg_bbox_area']:.2f}")
print(f"Median box area: {stats['median_bbox_area']:.2f}")
```

## Coordinate Systems

### Pixel Coordinates

All bounding box coordinates in BoxLab are in **absolute pixel coordinates**:

```python
# For a 1920x1080 image
bbox = BBox(x_min=100, y_min=200, x_max=300, y_max=400)

# Width and height in pixels
width = bbox.x_max - bbox.x_min  # 200 pixels
height = bbox.y_max - bbox.y_min  # 200 pixels
```

### Normalized Coordinates

For normalized coordinates (e.g., YOLO format), convert manually:

```python
# Image dimensions
img_width, img_height = 1920, 1080

# Absolute bbox
bbox = BBox(100, 200, 300, 400)

# Normalize
cx, cy, w, h = bbox.cxcywh
cx_norm = cx / img_width  # 0.104
cy_norm = cy / img_height  # 0.278
w_norm = w / img_width  # 0.104
h_norm = h / img_height  # 0.185
```

### Coordinate Origin

BoxLab uses the **top-left corner** as origin (0, 0):

```
(0,0) ──────→ x
  │
  │
  ↓
  y
```

## Immutability

All types are immutable (NamedTuple or TypedDict):

```python
bbox = BBox(10, 20, 100, 150)

# This raises an error
# bbox.x_min = 15  # AttributeError

# Create a new bbox instead
new_bbox = BBox(15, 20, 100, 150)
```

## Type Safety

Use type hints for better IDE support:

```python
from boxlab.dataset.types import BBox, Annotation, ImageInfo


def process_bbox(bbox: BBox) -> float:
    return bbox.area


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

## Validation

### SplitRatio Validation

```python
from boxlab.dataset.types import SplitRatio
from boxlab.exceptions import ValidationError

# Valid split
split = SplitRatio(0.7, 0.2, 0.1)
split.validate()  # OK

# Invalid split
try:
    split = SplitRatio(0.5, 0.3, 0.1)  # Sums to 0.9
    split.validate()
except ValidationError as e:
    print(f"Error: {e}")
```

### BBox Validation

BBox doesn't validate coordinates by default. Add custom validation if needed:

```python
def validate_bbox(bbox: BBox, img_width: int, img_height: int) -> bool:
    """Check if bbox is within image bounds."""
    return (
        0 <= bbox.x_min < bbox.x_max <= img_width and
        0 <= bbox.y_min < bbox.y_max <= img_height
    )


bbox = BBox(10, 20, 100, 150)
is_valid = validate_bbox(bbox, 1920, 1080)  # True
```

## See Also

- [Dataset Core](index.md) - Dataset management
- [Plugin System](plugins/index.md): Extend dataset functionality
- [I/O Operations](io.md) - Loading and exporting datasets
- [PyTorch Adapter](torchadapter.md) - Training integration

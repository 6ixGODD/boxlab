# PyTorch Adapter

::: boxlab.dataset.torchadapter
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The PyTorch adapter module provides seamless integration between BoxLab datasets and PyTorch. It converts BoxLab
datasets into PyTorch-compatible format, enabling direct use with DataLoader, torchvision transforms, and popular
detection models.

## Installation Requirements

This module requires additional dependencies:

```bash
pip install torch torchvision
```

If these packages are not installed, import errors will be raised with helpful installation instructions.

## Key Components

### TorchDatasetAdapter

Wraps BoxLab Dataset instances to provide PyTorch Dataset interface. Handles:

- Image loading and format conversion
- Annotation format conversion
- Transform pipeline application
- Batch collation for variable-sized objects

### Target Dictionary

The adapter returns targets in torchvision's standard object detection format:

```python
{
    'boxes': Tensor,  # Shape: (N, 4) - bounding boxes
    'labels': Tensor,  # Shape: (N,) - class labels
    'image_id': Tensor,  # Shape: (1,) - image identifier
    'area': Tensor,  # Shape: (N,) - box areas
    'iscrowd': Tensor  # Shape: (N,) - crowd flags
}
```

### Bounding Box Formats

Three formats are supported:

- **xyxy**: `[x_min, y_min, x_max, y_max]` - Top-left and bottom-right corners
- **xywh**: `[x_min, y_min, width, height]` - COCO format
- **cxcywh**: `[center_x, center_y, width, height]` - YOLO format

## Common Usage Patterns

### Basic Training Setup

```python
from boxlab.dataset import Dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader

# Load dataset
dataset = Dataset(name="my_dataset")
# ... populate dataset ...

# Create training dataset with augmentation
train_ds = build_torchdataset(
    dataset,
    image_size=640,
    augment=True,
    normalize=True
)

# Create DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=train_ds.collate
)

# Training loop
for images, targets in train_loader:
    # images: list of tensors
    # targets: list of dicts
    ...
```

### Train/Val Split

```python
from boxlab.dataset import Dataset
from boxlab.dataset.types import SplitRatio
from boxlab.dataset.torchadapter import build_torchdataset

# Split dataset
dataset = Dataset(name="full_dataset")
splits = dataset.split(SplitRatio(train=0.8, val=0.2, test=0.0), seed=42)

# Create separate Dataset instances
train_dataset = Dataset(name="train")
val_dataset = Dataset(name="val")

# Populate split datasets
for img_id in splits['train']:
    img_info = dataset.get_image(img_id)
    train_dataset.add_image(img_info)
    for ann in dataset.get_annotations(img_id):
        train_dataset.add_annotation(ann)

for img_id in splits['val']:
    img_info = dataset.get_image(img_id)
    val_dataset.add_image(img_info)
    for ann in dataset.get_annotations(img_id):
        val_dataset.add_annotation(ann)

# Create PyTorch datasets
train_torch = build_torchdataset(train_dataset, image_size=640, augment=True, normalize=True)
val_torch = build_torchdataset(val_dataset, image_size=640, augment=False, normalize=True)
```

### Custom Transforms

```python
from torchvision import transforms as T
from boxlab.dataset.torchadapter import TorchDatasetAdapter

# Define custom transform pipeline
transform = T.Compose(
    [
        T.Resize((640, 640)),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Create adapter with custom transforms
adapter = TorchDatasetAdapter(
    dataset,
    transform=transform,
    return_format="xyxy"
)
```

### Using with Detection Models

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from boxlab.dataset.torchadapter import build_torchdataset

# Prepare dataset
torch_dataset = build_torchdataset(
    dataset,
    image_size=800,
    augment=True,
    normalize=True,
    return_format="xyxy"  # Faster R-CNN expects xyxy
)

loader = DataLoader(
    torch_dataset,
    batch_size=4,
    collate_fn=torch_dataset.collate
)

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.train()

# Training
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for images, targets in loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
```

## Transform Pipeline Order

When using `build_torchdataset()`, transforms are applied in this order:

1. **Resize** (if `image_size` specified)
2. **Augmentation** (if `augment=True`):
    - Random horizontal flip
    - Color jitter
    - Random affine transformations
3. **ToTensor** (always applied)
4. **Normalization** (if `normalize=True`)
5. **Custom transforms** (additional args)

## Error Handling

### Missing Dependencies

```python
try:
    from boxlab.dataset.torchadapter import build_torchdataset
except RequiredModuleNotFoundError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch torchvision pillow")
```

### Missing Images

```python
from boxlab.exceptions import DatasetNotFoundError

try:
    image, target = torch_dataset[0]
except DatasetNotFoundError as e:
    print(f"Image file not found: {e}")
```

## See Also

- [Dataset Core](index.md) - Core dataset management
- [Plugin System](plugins/index.md): Extend dataset functionality
- [Types](types.md): Data structures and type definitions
- [I/O Operations](io.md) - Loading and exporting datasets
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) - Official docs
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html) - Transform reference

# Advanced Usage

This guide covers advanced BoxLab features including PyTorch training integration, custom plugin development, batch processing automation, and advanced dataset manipulation techniques.

## Table of Contents

- [PyTorch Training Integration](#pytorch-training-integration)
- [Custom Plugin Development](#custom-plugin-development)
- [Batch Processing & Automation](#batch-processing--automation)
- [Advanced Dataset Manipulation](#advanced-dataset-manipulation)
- [Performance Optimization](#performance-optimization)

## PyTorch Training Integration

### Complete Training Pipeline

Here's a complete example of using BoxLab with PyTorch for object detection training:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from boxlab.dataset.types import SplitRatio

# 1. Load and prepare dataset
print("Loading dataset...")
dataset = load_dataset("data/coco/annotations.json", format="coco")

# 2. Split dataset
print("Splitting dataset...")
split_ratio = SplitRatio(train=0.8, val=0.1, test=0.1)
splits = dataset.split(split_ratio, seed=42)

# 3. Create separate datasets for each split
from boxlab.dataset import Dataset

train_ds = Dataset(name="train")
val_ds = Dataset(name="val")

# Copy categories
for cat_id, cat_name in dataset.categories.items():
    train_ds.add_category(cat_id, cat_name)
    val_ds.add_category(cat_id, cat_name)

# Populate train split
for img_id in splits['train']:
    img_info = dataset.get_image(img_id)
    train_ds.add_image(img_info)
    for ann in dataset.get_annotations(img_id):
        train_ds.add_annotation(ann)

# Populate val split
for img_id in splits['val']:
    img_info = dataset.get_image(img_id)
    val_ds.add_image(img_info)
    for ann in dataset.get_annotations(img_id):
        val_ds.add_annotation(ann)

# 4. Create PyTorch datasets with augmentation
print("Creating PyTorch datasets...")
train_dataset = build_torchdataset(
    train_ds,
    image_size=(800, 800),
    augment=True,
    normalize=True,
    return_format="xyxy"  # Faster R-CNN expects xyxy format
)

val_dataset = build_torchdataset(
    val_ds,
    image_size=(800, 800),
    augment=False,
    normalize=True,
    return_format="xyxy"
)

# 5. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=val_dataset.collate,
    pin_memory=True
)

# 6. Initialize model
print("Initializing model...")
num_classes = dataset.num_categories() + 1  # +1 for background
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 7. Setup training
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# 8. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # Training phase
    model.train()
    train_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {losses.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"  Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")

    # Update learning rate
    lr_scheduler.step()

# 9. Save model
torch.save(model.state_dict(), 'trained_model.pth')
print("\nTraining completed! Model saved to 'trained_model.pth'")
```

### Custom Transforms

Create custom augmentation pipelines:

```python
from torchvision import transforms as T
from boxlab.dataset.torchadapter import TorchDatasetAdapter

# Define custom transform
class CustomTransform:
    def __init__(self, brightness=0.3, contrast=0.3):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast
        )

    def __call__(self, img):
        return self.color_jitter(img)

# Build dataset with custom transforms
train_transform = T.Compose([
    T.Resize((640, 640)),
    T.RandomHorizontalFlip(p=0.5),
    CustomTransform(brightness=0.3, contrast=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = TorchDatasetAdapter(
    dataset=train_ds,
    transform=train_transform,
    return_format="xyxy"
)
```

### Different Detection Frameworks

#### YOLO Format Training

```python
from boxlab.dataset.torchadapter import build_torchdataset

# YOLO expects cxcywh normalized format
yolo_dataset = build_torchdataset(
    dataset,
    image_size=640,
    augment=True,
    normalize=False,  # YOLO handles normalization internally
    return_format="cxcywh"
)

# Normalize coordinates manually
for images, targets in DataLoader(yolo_dataset, batch_size=16, collate_fn=yolo_dataset.collate):
    for target in targets:
        # Normalize boxes
        target['boxes'][:, 0] /= 640  # cx
        target['boxes'][:, 1] /= 640  # cy
        target['boxes'][:, 2] /= 640  # width
        target['boxes'][:, 3] /= 640  # height
```

#### DETR Training

```python
# DETR expects cxcywh normalized format
detr_dataset = build_torchdataset(
    dataset,
    image_size=(800, 800),
    augment=True,
    normalize=True,
    return_format="cxcywh"
)
```

## Custom Plugin Development

### Creating a Custom Loader

Create a loader for a custom annotation format:

```python
from boxlab.dataset.plugins import LoaderPlugin
from boxlab.dataset import Dataset
from boxlab.dataset.types import ImageInfo, Annotation, BBox
import json
import pathlib

class CustomJSONLoader(LoaderPlugin):
    """Loader for custom JSON format.

    Expected format:
    {
        "images": [
            {
                "id": "img_001",
                "filename": "image1.jpg",
                "width": 1920,
                "height": 1080
            }
        ],
        "annotations": [
            {
                "image_id": "img_001",
                "bbox": {"x": 100, "y": 200, "w": 300, "h": 400},
                "class": "person",
                "confidence": 0.95
            }
        ],
        "classes": ["person", "car", "bicycle"]
    }
    """

    @property
    def name(self) -> str:
        return "custom_json"

    @property
    def description(self) -> str:
        return "Custom JSON format loader"

    @property
    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load(self, path, **kwargs):
        path = pathlib.Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Load JSON
        with open(path, 'r') as f:
            data = json.load(f)

        # Create dataset
        dataset = Dataset(name=kwargs.get('name', path.stem))

        # Load classes
        for idx, class_name in enumerate(data.get('classes', [])):
            dataset.add_category(idx + 1, class_name)

        # Create class name to ID mapping
        class_to_id = {name: idx + 1 for idx, name in enumerate(data.get('classes', []))}

        # Load images
        image_root = kwargs.get('image_root', path.parent)
        for img_data in data.get('images', []):
            img_id = str(img_data['id'])
            img_path = pathlib.Path(image_root) / img_data['filename']

            img_info = ImageInfo(
                image_id=img_id,
                file_name=img_data['filename'],
                width=img_data['width'],
                height=img_data['height'],
                path=img_path if img_path.exists() else None
            )
            dataset.add_image(img_info)

        # Load annotations
        for ann_idx, ann_data in enumerate(data.get('annotations', [])):
            img_id = str(ann_data['image_id'])

            # Get category
            class_name = ann_data['class']
            if class_name not in class_to_id:
                print(f"Warning: Unknown class '{class_name}', skipping")
                continue

            cat_id = class_to_id[class_name]

            # Parse bbox
            bbox_data = ann_data['bbox']
            bbox = BBox.from_xywh(
                x=bbox_data['x'],
                y=bbox_data['y'],
                w=bbox_data['w'],
                h=bbox_data['h']
            )

            # Create annotation
            annotation = Annotation(
                bbox=bbox,
                category_id=cat_id,
                category_name=class_name,
                image_id=img_id,
                annotation_id=f"ann_{ann_idx}",
                area=bbox.area
            )
            dataset.add_annotation(annotation)

        return dataset

    def validate(self, path):
        """Validate that file is in custom JSON format."""
        path = pathlib.Path(path)

        if not super().validate(path):
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Check required fields
            return (
                'images' in data and
                'annotations' in data and
                'classes' in data
            )
        except Exception:
            return False

# Register the loader
from boxlab.dataset.plugins.registry import register_loader
register_loader("custom_json", CustomJSONLoader)

# Use the loader
from boxlab.dataset.io import load_dataset
dataset = load_dataset("data/custom.json", format="custom_json")
```

### Creating a Custom Exporter

Create an exporter for a custom format:

```python
from boxlab.dataset.plugins import ExporterPlugin
from boxlab.dataset import Dataset
import json
import pathlib
import shutil

class CustomJSONExporter(ExporterPlugin):
    """Exporter for custom JSON format."""

    @property
    def name(self) -> str:
        return "custom_json"

    @property
    def description(self) -> str:
        return "Custom JSON format exporter"

    @property
    def default_extension(self) -> str:
        return ".json"

    def export(
        self,
        dataset,
        output_dir,
        split_ratio=None,
        seed=None,
        naming_strategy=None,
        copy_images=True,
        **kwargs
    ):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle splits
        if split_ratio:
            splits = dataset.split(split_ratio, seed=seed)
        else:
            splits = {"all": list(dataset.images.keys())}

        # Export each split
        for split_name, image_ids in splits.items():
            split_data = {
                "images": [],
                "annotations": [],
                "classes": list(dataset.categories.values())
            }

            # Create images directory if copying
            if copy_images:
                images_dir = output_dir / "images" / split_name
                images_dir.mkdir(parents=True, exist_ok=True)

            # Process images
            for img_id in image_ids:
                img_info = dataset.get_image(img_id)
                if not img_info:
                    continue

                # Add image info
                split_data["images"].append({
                    "id": img_id,
                    "filename": img_info.file_name,
                    "width": img_info.width,
                    "height": img_info.height
                })

                # Copy image if requested
                if copy_images and img_info.path and img_info.path.exists():
                    dest_path = images_dir / img_info.file_name
                    shutil.copy2(img_info.path, dest_path)

                # Add annotations
                for ann in dataset.get_annotations(img_id):
                    x, y, w, h = ann.bbox.xywh
                    split_data["annotations"].append({
                        "image_id": img_id,
                        "bbox": {
                            "x": float(x),
                            "y": float(y),
                            "w": float(w),
                            "h": float(h)
                        },
                        "class": ann.category_name,
                        "confidence": 1.0
                    })

            # Write JSON
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)

    def get_default_config(self):
        return {
            "copy_images": True,
            "naming_strategy": "original",
            "indent": 2
        }

# Register the exporter
from boxlab.dataset.plugins.registry import register_exporter
register_exporter("custom_json", CustomJSONExporter)

# Use the exporter
from boxlab.dataset.io import export_dataset
export_dataset(dataset, "output/custom", format="custom_json")
```

### Creating a Custom Naming Strategy

```python
from boxlab.dataset.plugins import NamingStrategy
import hashlib
from datetime import datetime

class TimestampNaming:
    """Naming strategy using timestamps."""

    def gen_name(self, origin, source, image_id):
        """Generate name with timestamp prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract extension
        import pathlib
        ext = pathlib.Path(origin).suffix

        if source:
            return f"{timestamp}_{source}_{image_id}{ext}"
        return f"{timestamp}_{image_id}{ext}"

class HashNaming:
    """Naming strategy using content hash."""

    def gen_name(self, origin, source, image_id):
        """Generate name using MD5 hash."""
        import pathlib
        ext = pathlib.Path(origin).suffix

        # Create hash from origin + image_id
        content = f"{origin}_{image_id}".encode()
        hash_value = hashlib.md5(content).hexdigest()[:12]

        if source:
            return f"{source}_{hash_value}{ext}"
        return f"{hash_value}{ext}"

# Use custom naming
from boxlab.dataset.io import export_dataset

export_dataset(
    dataset,
    "output/",
    format="coco",
    naming_strategy=TimestampNaming()
)
```

## Advanced Dataset Manipulation

### Combining Multiple Sources with Weights

```python
from boxlab.dataset import Dataset
from boxlab.dataset.io import load_dataset, merge, export_dataset
from boxlab.dataset.types import SplitRatio
import random

def weighted_merge(datasets_with_weights, output_name="weighted_merge", seed=None):
    """Merge datasets with specified weights.

    Args:
        datasets_with_weights: List of (dataset, weight) tuples
        output_name: Name for merged dataset
        seed: Random seed

    Example:
        datasets_with_weights = [
            (manual_dataset, 0.7),   # 70% from manual
            (auto_dataset, 0.3)      # 30% from automatic
        ]
    """
    if seed:
        random.seed(seed)

    # Calculate total images needed
    total_images = sum(len(ds) for ds, _ in datasets_with_weights)

    # Sample from each dataset according to weight
    sampled_datasets = []

    for dataset, weight in datasets_with_weights:
        # Calculate number of images to sample
        n_samples = int(total_images * weight)
        n_samples = min(n_samples, len(dataset))  # Can't sample more than available

        # Random sample image IDs
        all_img_ids = list(dataset.images.keys())
        sampled_ids = random.sample(all_img_ids, n_samples)

        # Create subset dataset
        subset = Dataset(name=f"{dataset.name}_sampled")

        # Copy categories
        for cat_id, cat_name in dataset.categories.items():
            subset.add_category(cat_id, cat_name)

        # Add sampled images
        for img_id in sampled_ids:
            img_info = dataset.get_image(img_id)
            subset.add_image(img_info, source_name=dataset.name)

            for ann in dataset.get_annotations(img_id):
                subset.add_annotation(ann)

        sampled_datasets.append(subset)
        print(f"Sampled {len(subset)} images from {dataset.name} (weight: {weight})")

    # Merge sampled datasets
    merged = merge(*sampled_datasets, name=output_name, preserve_sources=True)

    print(f"\nMerged dataset: {len(merged)} images")

    # Show source distribution
    sources = merged.get_sources()
    for source in sources:
        source_imgs = [img_id for img_id in merged.images
                      if merged.get_image_source(img_id) == source]
        print(f"  {source}: {len(source_imgs)} images ({len(source_imgs)/len(merged)*100:.1f}%)")

    return merged

# Usage
manual_ds = load_dataset("data/manual.json", format="coco")
auto_ds = load_dataset("data/automatic.json", format="coco")

# 70% manual, 30% automatic
weighted = weighted_merge([
    (manual_ds, 0.7),
    (auto_ds, 0.3)
], seed=42)

export_dataset(weighted, "output/weighted", format="yolo")
```

### Category Remapping

```python
from boxlab.dataset import Dataset
from boxlab.dataset.types import Annotation

def remap_categories(dataset: Dataset, mapping: dict):
    """Remap category names/IDs.

    Args:
        dataset: Source dataset
        mapping: Dict mapping old category names to new names

    Example:
        mapping = {
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle"
        }
    """
    remapped = Dataset(name=f"{dataset.name}_remapped")

    # Build new category set
    new_categories = {}
    old_to_new_id = {}
    next_cat_id = 1

    for old_cat_id, old_cat_name in dataset.categories.items():
        new_cat_name = mapping.get(old_cat_name, old_cat_name)

        # Find or create new category ID
        if new_cat_name not in new_categories:
            new_categories[new_cat_name] = next_cat_id
            remapped.add_category(next_cat_id, new_cat_name)
            next_cat_id += 1

        old_to_new_id[old_cat_id] = new_categories[new_cat_name]

    # Copy images and remap annotations
    for img_id, img_info in dataset.images.items():
        remapped.add_image(img_info)

        for ann in dataset.get_annotations(img_id):
            new_cat_id = old_to_new_id[ann.category_id]
            new_cat_name = remapped.get_category_name(new_cat_id)

            new_ann = Annotation(
                bbox=ann.bbox,
                category_id=new_cat_id,
                category_name=new_cat_name,
                image_id=ann.image_id,
                annotation_id=ann.annotation_id,
                area=ann.area,
                iscrowd=ann.iscrowd
            )
            remapped.add_annotation(new_ann)

    print(f"Remapped categories:")
    print(f"  Original: {dataset.num_categories()} categories")
    print(f"  Remapped: {remapped.num_categories()} categories")

    return remapped

# Usage
dataset = load_dataset("data/annotations.json", format="coco")

# Merge similar categories
mapping = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "bike"
}

remapped = remap_categories(dataset, mapping)
export_dataset(remapped, "output/remapped", format="coco")
```

## Next Steps

- [CLI Overview](cli-overview.md) - Command-line interface
- [API Reference](../reference/index.md) - Complete API documentation

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [GitHub Repository](https://github.com/6ixGODD/boxlab)

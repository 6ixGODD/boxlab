# I/O Operations

::: boxlab.dataset.io
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The I/O module provides high-level convenience functions for common dataset operations. It simplifies loading,
exporting, and merging datasets with automatic format detection and sensible defaults.

## Key Concepts

### Automatic Format Detection

The `load_dataset()` function can automatically detect the format based on file structure:

- `.json` files → COCO format
- Directories with `.yaml`/`.yml` + `images/`/`labels/` → YOLO format

### Naming Strategies

When exporting datasets, you can control how output files are named:

| Strategy            | Pattern                 | Example                  |
|---------------------|-------------------------|--------------------------|
| `original`          | Keep original name      | `image001.jpg`           |
| `uuid`              | Random UUID             | `a1b2c3d4-e5f6-7890.jpg` |
| `uuid_prefix`       | UUID + source prefix    | `camera1_a1b2c3d4.jpg`   |
| `sequential`        | Sequential numbers      | `000001.jpg`             |
| `sequential_prefix` | Numbers + source prefix | `camera1_000001.jpg`     |

### Conflict Resolution

When merging datasets with duplicate category names:

- **skip**: Keep the first occurrence
- **rename**: Add `_other` suffix to duplicates
- **error**: Raise an exception

## Common Workflows

### Convert Between Formats

```python
from boxlab.dataset.io import load_dataset, export_dataset

# Load COCO dataset
dataset = load_dataset("coco/instances.json", format="coco")

# Export to YOLO format
export_dataset(dataset, "output/yolo", format="yolo")
```

### Split Dataset for Training

```python
from boxlab.dataset.io import load_dataset, export_dataset
from boxlab.dataset.types import SplitRatio

# Load dataset
dataset = load_dataset("my_dataset/")

# Export with 70/20/10 split
split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)
export_dataset(
    dataset,
    "output/split_data",
    format="yolo",
    split_ratio=split_ratio,
    seed=42  # Reproducible split
)
```

### Combine Multiple Datasets

```python
from boxlab.dataset.io import load_dataset, merge, export_dataset

# Load datasets from different sources
ds1 = load_dataset("source1/")
ds2 = load_dataset("source2/")
ds3 = load_dataset("source3/")

# Merge all datasets
merged = merge(ds1, ds2, ds3, name="combined_dataset")

# Export merged dataset
export_dataset(merged, "output/merged", format="coco")
```

## Format Support Discovery

### List Available Formats

```python
from boxlab.dataset.io import get_supported_formats

formats = get_supported_formats()

print("Available Loaders:")
for name, info in formats["loaders"].items():
    print(f"  • {name}: {info['description']}")

print("\nAvailable Exporters:")
for name, info in formats["exporters"].items():
    print(f"  • {name}: {info['description']}")
```

### Check Format Capabilities

```python
from boxlab.dataset.io import get_supported_formats

formats = get_supported_formats()

# Check COCO loader extensions
coco_info = formats["loaders"]["coco"]
print(f"COCO supports: {coco_info['supported_extensions']}")

# Check YOLO exporter defaults
yolo_info = formats["exporters"]["yolo"]
print(f"YOLO defaults: {yolo_info['default_config']}")
```

## Error Handling

### Handle Format Detection Failures

```python
from boxlab.dataset.io import load_dataset

try:
    # Try auto-detection
    dataset = load_dataset("unknown_structure/")
except ValueError as e:
    print(f"Auto-detection failed: {e}")
    # Fall back to explicit format
    dataset = load_dataset("unknown_structure/", format="yolo")
```

### Handle Merge Conflicts

```python
from boxlab.dataset.io import merge
from boxlab.exceptions import CategoryConflictError

try:
    merged = merge(
        ds1, ds2,
        resolve_conflicts="error"  # Strict mode
    )
except CategoryConflictError as e:
    print(f"Category conflict: {e}")
    # Retry with automatic resolution
    merged = merge(ds1, ds2, resolve_conflicts="rename")
```

## See Also

- [Dataset Core](index.md) - Core dataset management
- [Plugin System](plugins/index.md): Extend dataset functionality
- [Types](types.md): Data structures and type definitions
- [PyTorch Adapter](torchadapter.md) - Training integration

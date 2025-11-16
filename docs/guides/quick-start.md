# Quick Start

Get started with BoxLab in 5 minutes. This guide covers the essential operations to start working with your datasets
immediately.

## Prerequisites

BoxLab installed and working:

```bash
boxlab --version
```

If not installed, see [Installation Guide](installation.md).

## 5-Minute Walkthrough

### 1. View Dataset Information (30 seconds)

Check what's in your dataset:

```bash
# COCO format
boxlab dataset info data/coco/annotations.json --format coco

# YOLO format
boxlab dataset info data/yolo --format yolo
```

**Output:**

```
v Loading YOLO dataset
✓ Loaded 3 split(s)

Dataset: mydataset
-------------------------------------------------------

Overview
--------
Total Images      : 1371
Total Annotations : 967
Categories        : 5
Splits            : 3
Sources           : 1


Split Distribution
------------------
Split | Images | Annotations | Percentage
-----------------------------------------
Train | 959    | 670         | 69.9%
Val   | 205    | 148         | 15.0%
Test  | 207    | 149         | 15.1%

•
Showing combined statistics across all splits:

Combined Statistics
-------------------
  Images                : 1371
  Annotations           : 967
  Avg Annotations/Image : 0.71


Categories
----------
Category | Count | Percentage
-----------------------------
Cat_A    | 300   | 31.0%
Cat_B    | 250   | 25.9%
Cat_C    | 200   | 20.7%
Cat_D    | 150   | 15.5%
Cat_E    | 67    | 6.9%

✓ Dataset information displayed successfully
```

### 2. Convert Format (1 minute)

Convert between COCO and YOLO:

```bash
# COCO → YOLO
boxlab dataset convert \
  data/coco/annotations.json \
  -if coco \
  output/yolo \
  -of yolo

# YOLO → COCO
boxlab dataset convert \
  data/yolo \
  -if yolo \
  output/coco \
  -of coco
```

### 3. Visualize Dataset (1 minute)

Generate visual overview:

```bash
boxlab dataset visualize \
  data/coco/annotations.json \
  --format coco \
  --output viz
```

**Generated files:**

- Sample images with annotations
- Category distribution chart
- Split comparison
- Statistics summary

### 4. Launch Annotator (30 seconds)

Open the GUI application:

```bash
boxlab annotator
```

Then:

1. File → Import Dataset
2. Select format and path
3. Browse and edit annotations

### 5. Export Results (1 minute)

Export your work:

```bash
# In annotator:
File → Export Dataset...
# Select format and options

# Or via CLI after saving workspace:
boxlab dataset convert \
  workspace.cyw \
  -if coco \
  output/final \
  -of yolo
```

## Common Tasks

### Convert with Custom Split

```bash
boxlab dataset convert \
  input.json \
  -if coco \
  output \
  -of yolo \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

### Merge Multiple Datasets

```bash
boxlab dataset merge \
  -i data/set1.json coco dataset1 \
  -i data/set2.json coco dataset2 \
  -o output/merged
```

### Get Detailed Statistics

```bash
boxlab dataset info \
  data/coco/annotations.json \
  --format coco \
  --detailed \
  --by-source
```

### Visualize with All Features

```bash
boxlab dataset visualize \
  data/yolo \
  --format yolo \
  -o viz \
  --samples 10 \
  --show-heatmap \
  --show-source-dist
```

## Python API Quick Start

### Load and Explore

```python
from boxlab.dataset.io import load_dataset

# Load dataset
dataset = load_dataset("data/coco/annotations.json", format="coco")

# Get basic info
print(f"Images: {len(dataset)}")
print(f"Annotations: {dataset.num_annotations()}")
print(f"Categories: {dataset.num_categories()}")

# Get statistics
stats = dataset.get_statistics()
print(f"Avg annotations/image: {stats['avg_annotations_per_image']:.2f}")
```

### Convert Format

```python
from boxlab.dataset.io import load_dataset, export_dataset
from boxlab.dataset.types import SplitRatio

# Load
dataset = load_dataset("input.json", format="coco")

# Export with split
split_ratio = SplitRatio(train=0.8, val=0.1, test=0.1)
export_dataset(
    dataset,
    "output/yolo",
    format="yolo",
    split_ratio=split_ratio,
    seed=42
)
```

### Merge Datasets

```python
from boxlab.dataset.io import load_dataset, merge, export_dataset

# Load datasets
ds1 = load_dataset("data/set1.json", format="coco")
ds2 = load_dataset("data/set2.json", format="coco")

# Merge
merged = merge(ds1, ds2, name="combined")

# Export
export_dataset(merged, "output/merged", format="coco")
```

### PyTorch Integration

```python
from boxlab.dataset.io import load_dataset
from boxlab.dataset.torchadapter import build_torchdataset
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("data/train.json", format="coco")

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
    shuffle=True,
    collate_fn=torch_dataset.collate
)

# Training loop
for images, targets in loader:
    # Your training code
    pass
```

## Typical Workflows

### Workflow 1: Format Conversion

```bash
# 1. Check source format
boxlab dataset info input.json --format coco

# 2. Convert to YOLO
boxlab dataset convert input.json -if coco output -of yolo --seed 42

# 3. Verify output
boxlab dataset info output --format yolo

# 4. Visualize
boxlab dataset visualize output --format yolo -o viz
```

### Workflow 2: Dataset Merging

```bash
# 1. Check both datasets
boxlab dataset info set1.json --format coco
boxlab dataset info set2.json --format coco

# 2. Merge
boxlab dataset merge \
  -i set1.json coco \
  -i set2.json coco \
  -o merged \
  --seed 42

# 3. Verify merge
boxlab dataset info merged --format coco --show-source-dist

# 4. Export if needed
boxlab dataset convert merged/annotations.json -if coco final -of yolo
```

### Workflow 3: Annotation Review

```bash
# 1. Launch annotator
boxlab annotator

# 2. Import dataset
#    File → Import Dataset → Select format

# 3. Enable audit mode
#    Control Panel → Enable Audit Mode

# 4. Review images
#    F1 for approve, F2 for reject

# 5. Export audit report
#    Audit → Export Audit Report

# 6. Export dataset
#    File → Export Dataset
```

### Workflow 4: Training Preparation

```bash
# 1. Merge training sources
boxlab dataset merge \
  -i data/manual.json coco manual \
  -i data/auto.json coco auto \
  -o data/combined

# 2. Verify quality
boxlab annotator
# Import combined dataset, audit mode review

# 3. Convert to training format
boxlab dataset convert \
  data/combined/annotations.json \
  -if coco \
  data/training \
  -of yolo \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42

# 4. Visualize final splits
boxlab dataset visualize \
  data/training \
  --format yolo \
  -o viz \
  --show-heatmap
```

## Next Steps

Now that you've got the basics, explore more:

- **[CLI Dataset Commands](cli-dataset.md)** - Complete command reference

## Common Issues

### Command Not Found

```bash
# Try:
python -m boxlab --help

# Or check installation:
pip show boxlab
```

### Format Not Recognized

```bash
# Specify format explicitly:
boxlab dataset info data --format yolo

# Not:
boxlab dataset info data  # May fail auto-detection
```

### Permission Denied

```bash
# Check permissions:
ls -la data/

# Use sudo if needed (not recommended):
sudo boxlab dataset convert ...

# Better: Fix permissions
chmod -R u+rw data/
```

### Out of Memory

```bash
# Load specific split only:
boxlab dataset info data --format yolo --splits train

# Or use --no-copy for conversion:
boxlab dataset convert input.json -if coco output -of yolo --no-copy
```

## Getting Help

- **Documentation**: [Full guides](index.md)
- **API Reference**: [Complete API docs](../reference/index.md)
- **GitHub Issues**: [Report bugs](https://github.com/6ixGODD/boxlab/issues)
- **Discussions**: [Ask questions](https://github.com/6ixGODD/boxlab/discussions)

# Dataset Commands

Complete reference for all `boxlab dataset` subcommands. These commands provide powerful tools for managing, converting,
analyzing, and visualizing object detection datasets.

## Overview

The `dataset` command provides four subcommands:

```bash
boxlab dataset convert      # Convert between formats
boxlab dataset info         # Display dataset statistics
boxlab dataset merge        # Combine multiple datasets
boxlab dataset visualize    # Generate visualizations
```

## `dataset convert`

Convert datasets between COCO and YOLO formats with flexible splitting and naming options.

### Basic Usage

```bash
boxlab dataset convert INPUT -if FORMAT OUTPUT -of FORMAT
```

### Arguments

| Argument | Description        | Required |
|----------|--------------------|----------|
| `INPUT`  | Input dataset path | Yes      |
| `OUTPUT` | Output directory   | Yes      |

### Options

| Option            | Short | Type   | Default    | Description                                   |
|-------------------|-------|--------|------------|-----------------------------------------------|
| `--input-format`  | `-if` | choice | -          | Input format: `coco` or `yolo`                |
| `--output-format` | `-of` | choice | -          | Output format: `coco` or `yolo`               |
| `--split`         | `-s`  | choice | `train`    | YOLO split to load: `train`, `val`, or `test` |
| `--naming`        | `-n`  | choice | `original` | Naming strategy (see below)                   |
| `--train-ratio`   | -     | float  | 0.8        | Training set ratio                            |
| `--val-ratio`     | -     | float  | 0.1        | Validation set ratio                          |
| `--test-ratio`    | -     | float  | 0.1        | Test set ratio                                |
| `--no-split`      | -     | flag   | -          | Export as single split                        |
| `--seed`          | -     | int    | -          | Random seed for reproducibility               |
| `--no-copy`       | -     | flag   | -          | Don't copy images (annotations only)          |

### Naming Strategies

| Strategy            | Description             | Example Output         |
|---------------------|-------------------------|------------------------|
| `original`          | Keep original filename  | `image001.jpg`         |
| `prefix`            | Add prefix              | `PREFIX_image001.jpg`  |
| `uuid`              | Generate UUID           | `a1b2c3d4-e5f6.jpg`    |
| `uuid_prefix`       | UUID with source prefix | `camera1_a1b2c3d4.jpg` |
| `sequential`        | Sequential numbering    | `000001.jpg`           |
| `sequential_prefix` | Sequential with prefix  | `camera1_000001.jpg`   |

### Examples

#### COCO to YOLO

```bash
# Basic conversion
boxlab dataset convert annotations.json -if coco output/yolo -of yolo

# With custom split ratios
boxlab dataset convert annotations.json -if coco output/yolo -of yolo \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

# With UUID naming
boxlab dataset convert annotations.json -if coco output/yolo -of yolo \
  --naming uuid --seed 42

# Without splitting
boxlab dataset convert annotations.json -if coco output/yolo -of yolo \
  --no-split

# Annotations only (no image copy)
boxlab dataset convert annotations.json -if coco output/yolo -of yolo \
  --no-copy
```

#### YOLO to COCO

```bash
# Convert train split
boxlab dataset convert data/yolo -if yolo output/coco -of coco \
  --split train

# Convert with sequential naming
boxlab dataset convert data/yolo -if yolo output/coco -of coco \
  --split train --naming sequential
```

#### Advanced Usage

```bash
# Full control with all options
boxlab --verbose dataset convert annotations.json \
  -if coco \
  output/yolo \
  -of yolo \
  --naming sequential_prefix \
  --train-ratio 0.8 \
  --val-ratio 0.15 \
  --test-ratio 0.05 \
  --seed 42
```

### Output Structure

#### COCO Output

```
output/
├── annotations_train.json
├── annotations_val.json
├── annotations_test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

#### YOLO Output

```
output/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Common Errors

**Input not found:**

```bash
✗ Error: Dataset not found at path: annotations.json
```

Solution: Check file path and permissions

**Invalid split ratios:**

```bash
✗ Validation failed: Split ratios must sum to 1.0, got 0.9
```

Solution: Ensure ratios sum to 1.0

**Format detection failed:**

```bash
✗ Error: Could not auto-detect format
```

Solution: Specify format explicitly with `-if`

## `dataset info`

Display comprehensive information and statistics about a dataset.

### Basic Usage

```bash
boxlab dataset info INPUT --format FORMAT
```

### Arguments

| Argument | Description     | Required |
|----------|-----------------|----------|
| `INPUT`  | Path to dataset | Yes      |

### Options

| Option               | Short | Type   | Default | Description                                     |
|----------------------|-------|--------|---------|-------------------------------------------------|
| `--format`           | `-f`  | choice | -       | Dataset format: `coco` or `yolo`                |
| `--splits`           | `-s`  | list   | all     | Specific splits to load                         |
| `--by-source`        | -     | flag   | -       | Show statistics by source                       |
| `--detailed`         | `-d`  | flag   | -       | Show detailed statistics                        |
| `--show-source-dist` | -     | flag   | -       | Show source distribution from filename prefixes |

### Examples

#### Basic Information

```bash
# COCO dataset
boxlab dataset info data/coco --format coco

# YOLO dataset
boxlab dataset info data/yolo --format yolo

# Specific splits
boxlab dataset info data/yolo --format yolo --splits train val
```

#### Detailed Analysis

```bash
# Detailed statistics
boxlab dataset info data/coco --format coco --detailed

# Statistics by source
boxlab dataset info data/merged --format yolo --by-source

# Source distribution from filenames
boxlab dataset info data/merged --format yolo --show-source-dist
```

### Sample Output

```
╔═══════════════════════════════════════════════════════════╗
║                    Dataset: my_dataset                     ║
╚═══════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────────────────
Overview
──────────────────────────────────────────────────────────────
  Total Images         : 1000
  Total Annotations    : 5000
  Categories           : 10
  Splits               : 3
  Sources              : 2

──────────────────────────────────────────────────────────────
Split Distribution
──────────────────────────────────────────────────────────────
┌──────────┬────────┬──────────────┬────────────┐
│ Split    │ Images │ Annotations  │ Percentage │
├──────────┼────────┼──────────────┼────────────┤
│ Train    │ 800    │ 4000         │ 80.0%      │
│ Val      │ 100    │ 500          │ 10.0%      │
│ Test     │ 100    │ 500          │ 10.0%      │
└──────────┴────────┴──────────────┴────────────┘

──────────────────────────────────────────────────────────────
Statistics
──────────────────────────────────────────────────────────────
  Images                        : 1000
  Annotations                   : 5000
  Avg Annotations/Image         : 5.00

──────────────────────────────────────────────────────────────
Categories
──────────────────────────────────────────────────────────────
┌────────────────┬────────┬────────────┐
│ Category       │ Count  │ Percentage │
├────────────────┼────────┼────────────┤
│ person         │ 2000   │ 40.0%      │
│ car            │ 1500   │ 30.0%      │
│ bicycle        │ 1000   │ 20.0%      │
│ ...            │ ...    │ ...        │
└────────────────┴────────┴────────────┘

✓ Dataset information displayed successfully
```

### Use Cases

**Before conversion:**

```bash
# Check dataset before converting
boxlab dataset info input/coco --format coco
boxlab dataset convert input/coco/annotations.json -if coco output/yolo -of yolo
```

**Quality assurance:**

```bash
# Verify split distribution
boxlab dataset info data/yolo --format yolo --detailed

# Check source balance
boxlab dataset info data/merged --format yolo --show-source-dist
```

**Documentation:**

```bash
# Generate dataset report
boxlab dataset info data/dataset --format coco > dataset_report.txt
```

## `dataset merge`

Combine multiple datasets with conflict resolution and source tracking.

### Basic Usage

```bash
boxlab dataset merge \
  -i PATH FORMAT [NAME] \
  -i PATH FORMAT [NAME] \
  --output DIR
```

### Arguments

| Argument   | Description      | Required |
|------------|------------------|----------|
| `--output` | Output directory | Yes      |

### Options

| Option                  | Short | Type   | Default          | Description                                       |
|-------------------------|-------|--------|------------------|---------------------------------------------------|
| `--input`               | `-i`  | multi  | -                | Dataset input: `path format [name]`               |
| `--output-format`       | `-of` | choice | `coco`           | Output format                                     |
| `--name`                | `-n`  | string | `merged_dataset` | Merged dataset name                               |
| `--resolve-conflicts`   | `-r`  | choice | `skip`           | Conflict resolution: `skip`, `rename`, or `error` |
| `--no-preserve-sources` | -     | flag   | -                | Don't preserve source info                        |
| `--no-fix-duplicates`   | -     | flag   | -                | Don't fix duplicate categories                    |
| `--naming`              | -     | choice | `original`       | Naming strategy                                   |
| `--train-ratio`         | -     | float  | 0.8              | Training set ratio                                |
| `--val-ratio`           | -     | float  | 0.1              | Validation set ratio                              |
| `--test-ratio`          | -     | float  | 0.1              | Test set ratio                                    |
| `--no-split`            | -     | flag   | -                | Don't split output                                |
| `--seed`                | -     | int    | -                | Random seed                                       |
| `--no-copy`             | -     | flag   | -                | Don't copy images                                 |
| `--yolo-splits`         | -     | list   | all              | YOLO splits to load                               |
| `--unified-structure`   | -     | flag   | -                | Use unified directory structure                   |

### Input Format

Each `-i` flag specifies one dataset:

```bash
-i <path> <format> [name]
```

- `path`: Path to dataset (required)
- `format`: `coco` or `yolo` (required)
- `name`: Custom name for source tracking (optional)

### Conflict Resolution

| Strategy | Behavior                                |
|----------|-----------------------------------------|
| `skip`   | Use existing category (default)         |
| `rename` | Rename conflicting category with suffix |
| `error`  | Raise error and stop                    |

### Examples

#### Basic Merge

```bash
# Merge two COCO datasets
boxlab dataset merge \
  -i data/coco1/ann.json coco dataset1 \
  -i data/coco2/ann.json coco dataset2 \
  -o output/merged

# Merge without custom names
boxlab dataset merge \
  -i data/ann1.json coco \
  -i data/ann2.json coco \
  -o output/merged
```

#### YOLO Datasets

```bash
# Merge all splits
boxlab dataset merge \
  -i data/yolo1 yolo source1 \
  -i data/yolo2 yolo source2 \
  -o output/merged --output-format yolo

# Merge specific splits only
boxlab dataset merge \
  -i data/yolo1 yolo source1 \
  -i data/yolo2 yolo source2 \
  -o output/merged \
  --yolo-splits train val
```

#### Mixed Formats

```bash
# Merge COCO and YOLO
boxlab dataset merge \
  -i data/coco/ann.json coco coco_source \
  -i data/yolo yolo yolo_source \
  -o output/merged --output-format coco
```

#### Advanced Options

```bash
# Full control
boxlab dataset merge \
  -i data/ds1/ann.json coco dataset1 \
  -i data/ds2/ann.json coco dataset2 \
  -i data/ds3 yolo dataset3 \
  -o output/merged \
  --output-format yolo \
  --name combined_dataset \
  --resolve-conflicts rename \
  --naming sequential_prefix \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42 \
  --unified-structure
```

#### Without Source Preservation

```bash
# Merge without tracking sources
boxlab dataset merge \
  -i data/ds1.json coco \
  -i data/ds2.json coco \
  -o output/merged \
  --no-preserve-sources
```

### Sample Output

```
╔═══════════════════════════════════════════════════════════╗
║                  Dataset Merge Operation                   ║
╚═══════════════════════════════════════════════════════════╝

ℹ Loading 3 datasets...

[Step 1] Loading dataset 1/3: data/ds1.json
⠋ Loading COCO dataset...
✓ Loaded 'dataset1': 500 images, 2500 annotations, 10 categories

[Step 2] Loading dataset 2/3: data/ds2.json
⠋ Loading COCO dataset...
✓ Loaded 'dataset2': 300 images, 1500 annotations, 10 categories

[Step 3] Loading dataset 3/3: data/ds3
⠋ Loading YOLO dataset...
✓ Loaded 'dataset3': 200 images, 1000 annotations, 8 categories

──────────────────────────────────────────────────────────────

ℹ Merge Summary:
  • Total images: 1000
  • Total annotations: 5000
  • Conflict resolution: skip
  • Preserve sources: True
  • Fix duplicates: True
  • Output format: coco
  • Split ratio: train=0.80, val=0.10, test=0.10

──────────────────────────────────────────────────────────────

? Proceed with merge? (Y/n) y

⠋ Merging datasets...
✓ Merged successfully: 1000 images, 5000 annotations, 10 categories
ℹ Sources: dataset1, dataset2, dataset3

⠋ Exporting to: output/merged...

──────────────────────────────────────────────────────────────
✓ Dataset merge completed successfully!
ℹ Output: output/merged
```

### Use Cases

**Combine training data:**

```bash
# Merge multiple annotation sources
boxlab dataset merge \
  -i data/manual/ann.json coco manual \
  -i data/auto/ann.json coco automatic \
  -o data/combined
```

**Add new data:**

```bash
# Add new batch to existing dataset
boxlab dataset merge \
  -i data/existing/ann.json coco existing \
  -i data/new_batch/ann.json coco new_batch \
  -o data/updated
```

**Format unification:**

```bash
# Convert and merge different formats
boxlab dataset merge \
  -i data/coco/ann.json coco \
  -i data/yolo yolo \
  -o data/unified --output-format coco
```

## `dataset visualize`

Generate comprehensive visual representations of datasets including sample images, distribution plots, and heatmaps.

### Basic Usage

```bash
boxlab dataset visualize INPUT --format FORMAT --output DIR
```

### Arguments

| Argument | Description     | Required |
|----------|-----------------|----------|
| `INPUT`  | Path to dataset | Yes      |

### Options

| Option               | Short | Type   | Default | Description                          |
|----------------------|-------|--------|---------|--------------------------------------|
| `--format`           | `-f`  | choice | -       | Dataset format: `coco` or `yolo`     |
| `--output`           | `-o`  | string | -       | Output directory                     |
| `--splits`           | `-s`  | list   | all     | Specific splits to visualize         |
| `--samples`          | `-n`  | int    | 5       | Number of sample images per split    |
| `--no-category-dist` | -     | flag   | -       | Don't generate category distribution |
| `--show-source-dist` | -     | flag   | -       | Generate source distribution plot    |
| `--show-heatmap`     | -     | flag   | -       | Generate annotation density heatmap  |
| `--seed`             | -     | int    | -       | Random seed for sample selection     |

### Examples

#### Basic Visualization

```bash
# Visualize COCO dataset
boxlab dataset visualize data/coco --format coco -o viz

# Visualize YOLO dataset
boxlab dataset visualize data/yolo --format yolo -o viz

# Specific splits only
boxlab dataset visualize data/yolo --format yolo -o viz \
  --splits train val
```

#### Advanced Visualizations

```bash
# With all analysis features
boxlab dataset visualize data/merged --format yolo -o viz \
  --show-heatmap \
  --show-source-dist \
  --samples 10

# More samples, no category distribution
boxlab dataset visualize data/coco --format coco -o viz \
  --samples 20 \
  --no-category-dist

# Reproducible sample selection
boxlab dataset visualize data/coco --format coco -o viz \
  --samples 10 \
  --seed 42
```

### Generated Files

The command generates several visualization files:

```
viz/
├── samples_train_001.png          # Sample images with annotations
├── samples_train_002.png
├── samples_val_001.png
├── category_distribution.png      # Category counts bar chart
├── split_comparison.png           # Split size comparison
├── source_distribution.png        # Source distribution (if enabled)
└── annotation_heatmap.png         # Density heatmap (if enabled)
```

### Sample Output

```
╔═══════════════════════════════════════════════════════════╗
║                  Dataset Visualization                     ║
╚═══════════════════════════════════════════════════════════╝

⠋ Loading COCO dataset...
✓ Loaded 3 split(s): 1000 total images

ℹ Generating visualizations...

──────────────────────────────────────────────────────────────
✓ Visualization completed!
ℹ Output directory: viz

Generated files:
  • Sample Images (Train): samples_train_001.png
  • Sample Images (Val): samples_val_001.png
  • Category Distribution: category_distribution.png
  • Split Comparison: split_comparison.png
  • Source Distribution: source_distribution.png
  • Annotation Heatmap: annotation_heatmap.png
```

### Visualization Types

#### 1. Sample Images

Shows random images with bounding boxes and labels:

- Colored boxes per category
- Category labels
- Confidence scores (if available)
- Grid layout

#### 2. Category Distribution

Bar chart showing:

- Annotation count per category
- Percentage distribution
- Sorted by frequency

#### 3. Split Comparison

Comparison chart showing:

- Image counts per split
- Annotation counts per split
- Percentage breakdown

#### 4. Source Distribution

Pie chart showing (with `--show-source-dist`):

- Images per source
- Percentage per source
- Color-coded sources

#### 5. Annotation Heatmap

Density map showing (with `--show-heatmap`):

- Annotation spatial distribution
- Hot spots and cold spots
- Overlay on sample images

### Use Cases

**Dataset exploration:**

```bash
# Quick overview
boxlab dataset visualize data/new_dataset --format coco -o explore

# Detailed analysis
boxlab dataset visualize data/new_dataset --format coco -o explore \
  --samples 15 \
  --show-heatmap \
  --show-source-dist
```

**Quality assurance:**

```bash
# Check annotation quality
boxlab dataset visualize data/annotated --format coco -o qa \
  --samples 20 \
  --seed 42

# Verify split balance
boxlab dataset visualize data/split --format yolo -o qa \
  --splits train val test
```

**Documentation:**

```bash
# Generate dataset documentation
boxlab dataset visualize data/final --format coco -o docs \
  --samples 10 \
  --show-heatmap \
  --show-source-dist
```

**Presentation:**

```bash
# Create presentation materials
boxlab dataset visualize data/project --format coco -o presentation \
  --samples 6 \
  --seed 42
```

## Best Practices

### 1. Always Use Seeds

For reproducibility:

```bash
boxlab dataset convert input.json -if coco output -of yolo --seed 42
boxlab dataset merge -i ds1.json coco -i ds2.json coco -o merged --seed 42
boxlab dataset visualize data/coco -f coco -o viz --seed 42
```

### 2. Check Before Processing

Always inspect datasets first:

```bash
# Check source
boxlab dataset info data/source --format coco

# Then process
boxlab dataset convert data/source/ann.json -if coco output -of yolo
```

### 3. Use Verbose Mode for Debugging

```bash
boxlab --verbose dataset convert input.json -if coco output -of yolo
```

### 4. Validate Output

After processing:

```bash
# Convert
boxlab dataset convert input.json -if coco output/yolo -of yolo

# Verify
boxlab dataset info output/yolo --format yolo
```

### 5. Use Descriptive Names

```bash
# Bad
boxlab dataset merge -i ds1.json coco -i ds2.json coco -o out

# Good
boxlab dataset merge \
  -i manual_annotations.json coco manual_data \
  -i auto_annotations.json coco automatic_data \
  -o merged_training_data \
  --name training_v1
```

## Next Steps

- [Annotator Command](cli-annotator.md) - GUI annotation tool

## Reference

- [API Reference](../reference/index.md) - Python API documentation
- [CLI Overview](cli-overview.md) - General CLI guide

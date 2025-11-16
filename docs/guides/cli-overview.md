# CLI Overview

BoxLab provides a comprehensive command-line interface for managing, converting, and analyzing object detection
datasets. This guide covers the CLI structure, global options, and common usage patterns.

## Command Structure

BoxLab CLI follows a hierarchical command structure:

```
boxlab [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options

Available for all commands:

```bash
boxlab --help              # Show help message
boxlab --version           # Show version information
boxlab --verbose           # Enable verbose output with tracebacks
boxlab -v                  # Short form of --verbose
```

## Available Commands

### `annotator`

Launch the GUI annotation tool:

```bash
boxlab annotator
```

No arguments required. Opens the interactive annotation interface.

### `dataset`

Dataset management operations with subcommands:

```bash
boxlab dataset SUBCOMMAND [OPTIONS]
```

Subcommands:

- `convert` - Convert between formats
- `info` - Display dataset information
- `merge` - Merge multiple datasets
- `visualize` - Generate visualizations

## Common Patterns

### Getting Help

```bash
# General help
boxlab --help

# Command help
boxlab dataset --help

# Subcommand help
boxlab dataset convert --help
```

### Verbose Mode

Enable detailed output and full tracebacks:

```bash
boxlab --verbose dataset convert input.json -if coco output -of yolo
```

### Version Information

```bash
boxlab --version
```

## Running Methods

### As Installed Command

After `pip install boxlab`:

```bash
boxlab dataset info data/coco --format coco
```

### As Python Module

Useful when `boxlab` command is not in PATH:

```bash
python -m boxlab dataset info data/coco --format coco
```

### With Poetry

In development environment:

```bash
poetry run boxlab dataset info data/coco --format coco
```

## Error Handling

### Error Codes

BoxLab CLI uses exit codes to indicate status:

- `0` - Success
- `1` - General error
- `2` - Missing dependency error
- `10-16` - Dataset-related errors
- `20` - Validation error
- `130` - Interrupted by user (Ctrl+C)

### Error Messages

Without `--verbose`:

```
✗ Dataset not found at path: /path/to/dataset
```

With `--verbose`:

```
✗ Dataset not found at path: /path/to/dataset

Traceback (most recent call last):
  File "boxlab/cli/__init__.py", line 15, in main
    ...
boxlab.exceptions.DatasetNotFoundError: Dataset not found at path: /path/to/dataset
```

## Shell Completion

### Bash

Add to `~/.bashrc`:

```bash
eval "$(boxlab --completion bash)"
```

### Zsh

Add to `~/.zshrc`:

```bash
eval "$(boxlab --completion zsh)"
```

### Fish

Add to `~/.config/fish/completions/boxlab.fish`:

```fish
boxlab --completion fish | source
```

## Tips and Tricks

### 1. Use Absolute Paths

Always use absolute paths for clarity:

```bash
boxlab dataset convert /home/user/data/coco.json -if coco /home/user/output -of yolo
```

### 2. Check Before Converting

View dataset info before conversion:

```bash
# Check source
boxlab dataset info data/coco --format coco

# Then convert
boxlab dataset convert data/coco.json -if coco output/yolo -of yolo
```

### 3. Use `--seed` for Reproducibility

Always specify seed for consistent splits:

```bash
boxlab dataset convert input.json -if coco output -of yolo --seed 42
```

### 4. Dry Run with `--no-copy`

Test conversion without copying images:

```bash
boxlab dataset convert input.json -if coco output -of yolo --no-copy
```

## Next Steps

- [Dataset Commands](cli-dataset.md) - Detailed guide to dataset operations
- [Annotator Command](cli-annotator.md) - Using the GUI from CLI

## Reference

- [API Reference](../reference/index.md) - Python API documentation
- [Dataset CLI](cli-dataset.md) - Dataset command reference

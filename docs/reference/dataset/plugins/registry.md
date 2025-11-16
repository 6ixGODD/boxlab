# Registry

::: boxlab.dataset.plugins.registry
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The registry module provides centralized management for dataset loader and exporter plugins. It allows registration,
retrieval, and discovery of available plugins at runtime.

## Key Features

- **Plugin Registration**: Register custom loaders and exporters
- **Plugin Retrieval**: Get plugin instances by name
- **Plugin Discovery**: List and inspect available plugins
- **Plugin Information**: Access plugin metadata and capabilities

## Registration

Register custom plugins to make them available throughout the application:

```python
from boxlab.dataset.plugins.registry import register_loader, register_exporter
from boxlab.dataset.loaders import COCOLoader
from boxlab.dataset.exporters import COCOExporter

# Register loader
register_loader("coco", COCOLoader)

# Register exporter
register_exporter("coco", COCOExporter)
```

## Retrieval

Get plugin instances by name:

```python
from boxlab.dataset.plugins.registry import get_loader, get_exporter

# Get loader instance
loader = get_loader("coco")
dataset = loader.load("annotations.json")

# Get exporter instance
exporter = get_exporter("yolo")
exporter.export(dataset, "output/yolo_format")
```

## Discovery

List and inspect available plugins:

```python
from boxlab.dataset.plugins.registry import (
    list_loaders,
    list_exporters,
    get_loader_info,
    get_exporter_info,
)

# List available loaders
loaders = list_loaders()
print(f"Available loaders: {loaders}")  # ['coco', 'yolo']

# List available exporters
exporters = list_exporters()
print(f"Available exporters: {exporters}")  # ['coco', 'yolo']

# Get detailed loader information
loader_info = get_loader_info()
for name, info in loader_info.items():
    print(f"\nLoader: {name}")
    print(f"  Description: {info['description']}")
    print(f"  Extensions: {info['supported_extensions']}")

# Get detailed exporter information
exporter_info = get_exporter_info()
for name, info in exporter_info.items():
    print(f"\nExporter: {name}")
    print(f"  Description: {info['description']}")
    print(f"  Default Config: {info['default_config']}")
```

## Auto-Detection

Use registry information for format auto-detection:

```python
from pathlib import Path
from boxlab.dataset.plugins.registry import get_loader_info, get_loader

def auto_load_dataset(file_path: str):
    file_ext = Path(file_path).suffix.lower()

    # Find compatible loader
    loader_info = get_loader_info()
    for name, info in loader_info.items():
        if file_ext in info['supported_extensions']:
            print(f"Detected format: {name}")
            loader = get_loader(name)
            return loader.load(file_path)

    raise ValueError(f"No loader found for extension '{file_ext}'")

# Usage
dataset = auto_load_dataset("annotations.json")  # Auto-detects COCO
```

## Error Handling

Handle missing plugins gracefully:

```python
from boxlab.dataset.plugins.registry import get_loader, list_loaders

try:
    loader = get_loader("unknown_format")
    dataset = loader.load("data.txt")
except KeyError as e:
    print(f"Error: {e}")
    print(f"Available loaders: {list_loaders()}")
```

## API Reference

### Registration Functions

- `register_loader(name, loader_class)`: Register a loader plugin
- `register_exporter(name, exporter_class)`: Register an exporter plugin

### Retrieval Functions

- `get_loader(name)`: Get a loader instance
- `get_exporter(name)`: Get an exporter instance

### Discovery Functions

- `list_loaders()`: List all registered loader names
- `list_exporters()`: List all registered exporter names
- `get_loader_info()`: Get detailed information about all loaders
- `get_exporter_info()`: Get detailed information about all exporters

## See Also

- [Plugin System](index.md): Plugin architecture overview
- [COCO Plugin](coco.md): COCO format plugin
- [YOLO Plugin](yolo.md): YOLO format plugin
- [Dataset](../index.md): Core dataset management

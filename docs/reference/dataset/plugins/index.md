# Plugins

::: boxlab.dataset.plugins
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

The plugin system provides extensible interfaces for loading and exporting datasets in various formats. BoxLab comes
with built-in plugins for popular formats like COCO and YOLO, and allows custom plugin development.

## Architecture

The plugin system consists of three main components:

1. **NamingStrategy**: Protocol for generating file names during export
2. **LoaderPlugin**: Abstract base class for dataset loaders
3. **ExporterPlugin**: Abstract base class for dataset exporters

## NamingStrategy Protocol

Define custom file naming strategies when exporting datasets:

```python
class CustomNamingStrategy:
    def gen_name(self, origin: str, source: str | None, image_id: str) -> str:
        if source:
            return f"{source}_{image_id}_{origin}"
        return f"{image_id}_{origin}"

# Use with exporter
strategy = CustomNamingStrategy()
exporter.export(dataset, output_dir="output/", naming_strategy=strategy)
```

## LoaderPlugin

Create custom dataset loaders by implementing the `LoaderPlugin` abstract class:

```python
import json

from boxlab.dataset import Dataset
from boxlab.dataset.plugins import LoaderPlugin

class CustomLoader(LoaderPlugin):
    @property
    def name(self) -> str:
        return "custom"

    @property
    def description(self) -> str:
        return "Custom JSON format loader"

    @property
    def supported_extensions(self) -> list[str]:
        return [".json", ".jsonl"]

    def load(self, path, **kwargs):
        dataset = Dataset(name="custom_dataset")

        with open(path, "r") as f:
            data = json.load(f)

        # Parse and populate dataset
        for item in data["images"]:
            # Add images, annotations, categories
            pass

        return dataset

# Register and use the loader
from boxlab.dataset.plugins.registry import register_loader
register_loader("custom", CustomLoader)

loader = get_loader("custom")
dataset = loader.load("path/to/dataset.json")
```

## ExporterPlugin

Create custom dataset exporters by implementing the `ExporterPlugin` abstract class:

```python
from boxlab.dataset import Dataset, SplitRatio
from boxlab.dataset.plugins import ExporterPlugin
import json
from pathlib import Path

class CustomExporter(ExporterPlugin):
    @property
    def name(self) -> str:
        return "custom"

    @property
    def description(self) -> str:
        return "Export to custom JSON format"

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
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle splits if requested
        if split_ratio:
            splits = dataset.split(split_ratio, seed=seed)
        else:
            splits = {"all": list(dataset.images.keys())}

        # Export each split
        for split_name, image_ids in splits.items():
            split_data = {"images": [], "annotations": []}

            # Export logic here
            # ...

            # Write JSON file
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(split_data, f, indent=2)

# Register and use the exporter
from boxlab.dataset.plugins.registry import register_exporter
register_exporter("custom", CustomExporter)

exporter = get_exporter("custom")
exporter.export(
    dataset,
    output_dir="output/custom",
    split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    seed=42,
)
```

## Built-in Plugins

BoxLab includes the following built-in plugins:

- [COCO](coco.md): Load and export COCO format datasets
- [YOLO](yolo.md): Load and export YOLOv5/YOLOv8 format datasets

## Plugin Registry

Manage plugins using the registry system:

- [Registry](registry.md): Register, retrieve, and discover plugins

## Key Methods

### LoaderPlugin

- `name`: Unique plugin identifier
- `description`: Human-readable description
- `supported_extensions`: List of supported file extensions
- `load()`: Load dataset from path
- `validate()`: Check if loader can handle a path

### ExporterPlugin

- `name`: Unique plugin identifier
- `description`: Human-readable description
- `default_extension`: Default file extension
- `export()`: Export dataset to directory
- `get_default_config()`: Get default configuration

## See Also

- [Registry](registry.md): Plugin registration and discovery
- [COCO Plugin](coco.md): COCO format implementation
- [YOLO Plugin](yolo.md): YOLO format implementation
- [Dataset](../index.md): Core dataset management

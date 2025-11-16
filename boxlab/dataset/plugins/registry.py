from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from boxlab.dataset.plugins import ExporterPlugin
    from boxlab.dataset.plugins import LoaderPlugin

_LOADERS: dict[str, type[LoaderPlugin]] = {}
_EXPORTERS: dict[str, type[ExporterPlugin]] = {}


def register_loader(name: str, loader_class: type[LoaderPlugin]) -> None:
    """Register a loader plugin.

    Adds a loader plugin to the global registry, making it available for use
    throughout the application. The loader can then be retrieved by name using
    get_loader().

    Args:
        name: Unique identifier for the loader (e.g., 'coco', 'yolo'). Should
            be lowercase and descriptive of the format.
        loader_class: Loader class to register. Must be a subclass of LoaderPlugin.

    Raises:
        ValueError: If a loader with the same name is already registered.

    Example:
        Register a custom loader:

        ```python
        from boxlab.dataset.plugins import LoaderPlugin
        from boxlab.dataset.registry import register_loader


        class CustomLoader(LoaderPlugin):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "Custom format loader"

            def load(self, path, **kwargs):
                # Implementation
                pass


        # Register the loader
        register_loader("custom", CustomLoader)

        # Now it can be used
        loader = get_loader("custom")
        dataset = loader.load("data.custom")
        ```

    Example:
        Register built-in loaders at application startup:

        ```python
        from boxlab.dataset.loaders import COCOLoader, YOLOLoader
        from boxlab.dataset.registry import register_loader


        def setup_loaders():
            register_loader("coco", COCOLoader)
            register_loader("yolo", YOLOLoader)


        setup_loaders()
        ```
    """
    if name in _LOADERS:
        raise ValueError(f"Loader '{name}' is already registered")

    _LOADERS[name] = loader_class


def register_exporter(name: str, exporter_class: type[ExporterPlugin]) -> None:
    """Register an exporter plugin.

    Adds an exporter plugin to the global registry, making it available for use
    throughout the application. The exporter can then be retrieved by name using
    get_exporter().

    Args:
        name: Unique identifier for the exporter (e.g., 'coco', 'yolo'). Should
            be lowercase and descriptive of the format.
        exporter_class: Exporter class to register. Must be a subclass of
            ExporterPlugin.

    Raises:
        ValueError: If an exporter with the same name is already registered.

    Example:
        Register a custom exporter:

        ```python
        from boxlab.dataset.plugins import ExporterPlugin
        from boxlab.dataset.registry import register_exporter


        class CustomExporter(ExporterPlugin):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "Custom format exporter"

            def export(self, dataset, output_dir, **kwargs):
                # Implementation
                pass


        # Register the exporter
        register_exporter("custom", CustomExporter)

        # Now it can be used
        exporter = get_exporter("custom")
        exporter.export(dataset, "output/")
        ```

    Example:
        Register multiple exporters:

        ```python
        from boxlab.dataset.exporters import (
            COCOExporter,
            YOLOExporter,
            PascalVOCExporter,
        )
        from boxlab.dataset.registry import register_exporter


        def setup_exporters():
            register_exporter("coco", COCOExporter)
            register_exporter("yolo", YOLOExporter)
            register_exporter("pascal_voc", PascalVOCExporter)


        setup_exporters()
        ```
    """
    if name in _EXPORTERS:
        raise ValueError(f"Exporter '{name}' is already registered")

    _EXPORTERS[name] = exporter_class


def get_loader(name: str) -> LoaderPlugin:
    """Get a registered loader instance.

    Retrieves and instantiates a loader plugin from the registry by name.
    Each call creates a new instance of the loader.

    Args:
        name: Loader name. Must match a previously registered loader.

    Returns:
        A new instance of the requested LoaderPlugin.

    Raises:
        KeyError: If no loader with the given name is registered. The error
            message includes a list of available loaders.

    Example:
        Basic usage:

        ```python
        from boxlab.dataset.registry import get_loader

        # Get COCO loader and load dataset
        loader = get_loader("coco")
        dataset = loader.load("annotations/instances.json")

        print(f"Loaded {len(dataset)} images")
        ```

    Example:
        Error handling with available loaders:

        ```python
        from boxlab.dataset.registry import get_loader, list_loaders

        try:
            loader = get_loader("unknown_format")
            dataset = loader.load("data.txt")
        except KeyError as e:
            print(f"Error: {e}")
            print(f"Available loaders: {list_loaders()}")
        ```

    Example:
        Dynamic loader selection:

        ```python
        from pathlib import Path
        from boxlab.dataset.registry import get_loader


        def load_dataset(file_path: str):
            path = Path(file_path)

            # Determine format from extension
            if path.suffix == ".json":
                loader = get_loader("coco")
            elif path.suffix in [".yaml", ".yml"]:
                loader = get_loader("yolo")
            else:
                raise ValueError(
                    f"Unsupported format: {path.suffix}"
                )

            return loader.load(file_path)


        dataset = load_dataset("data/annotations.json")
        ```
    """
    if name not in _LOADERS:
        raise KeyError(f"Loader '{name}' not found. Available loaders: {list(_LOADERS.keys())}")

    return _LOADERS[name]()


def get_exporter(name: str) -> ExporterPlugin:
    """Get a registered exporter instance.

    Retrieves and instantiates an exporter plugin from the registry by name.
    Each call creates a new instance of the exporter.

    Args:
        name: Exporter name. Must match a previously registered exporter.

    Returns:
        A new instance of the requested ExporterPlugin.

    Raises:
        KeyError: If no exporter with the given name is registered. The error
            message includes a list of available exporters.

    Example:
        Basic usage:

        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.registry import get_exporter

        # Get COCO exporter and export dataset
        exporter = get_exporter("coco")
        exporter.export(
            dataset=my_dataset, output_dir="output/coco_format"
        )
        ```

    Example:
        Export to multiple formats:

        ```python
        from boxlab.dataset.registry import get_exporter
        from boxlab.dataset.types import SplitRatio

        dataset = my_dataset  # Your dataset
        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

        # Export to COCO format
        coco_exporter = get_exporter("coco")
        coco_exporter.export(
            dataset, "output/coco", split_ratio=split_ratio, seed=42
        )

        # Export to YOLO format
        yolo_exporter = get_exporter("yolo")
        yolo_exporter.export(
            dataset, "output/yolo", split_ratio=split_ratio, seed=42
        )
        ```

    Example:
        Error handling:

        ```python
        from boxlab.dataset.registry import (
            get_exporter,
            list_exporters,
        )


        def export_dataset(
            dataset, format_name: str, output_dir: str
        ):
            try:
                exporter = get_exporter(format_name)
                exporter.export(dataset, output_dir)
                print(
                    f"Exported to {format_name} format successfully"
                )
            except KeyError:
                available = list_exporters()
                print(f"Unknown format '{format_name}'")
                print(f"Available formats: {', '.join(available)}")
        ```
    """
    if name not in _EXPORTERS:
        raise KeyError(
            f"Exporter '{name}' not found. Available exporters: {list(_EXPORTERS.keys())}"
        )

    return _EXPORTERS[name]()


def list_loaders() -> list[str]:
    """List all registered loaders.

    Returns the names of all loader plugins currently registered in the system.
    Useful for displaying available formats or implementing format auto-detection.

    Returns:
        List of loader names as strings, in no particular order.

    Example:
        Display available loaders:

        ```python
        from boxlab.dataset.registry import list_loaders

        loaders = list_loaders()
        print("Available data formats:")
        for loader in loaders:
            print(f"  - {loader}")

        # Output:
        # Available data formats:
        #   - coco
        #   - yolo
        #   - pascal_voc
        ```

    Example:
        Check if a specific loader is available:

        ```python
        from boxlab.dataset.registry import list_loaders


        def is_format_supported(format_name: str) -> bool:
            return format_name in list_loaders()


        if is_format_supported("coco"):
            print("COCO format is supported")
        else:
            print("COCO format is not available")
        ```

    Example:
        Build a CLI help message:

        ```python
        from boxlab.dataset.registry import list_loaders


        def print_help():
            loaders = list_loaders()
            print("Usage: dataset-tool load --format FORMAT file")
            print(f"Supported formats: {', '.join(loaders)}")


        print_help()
        # Output: Usage: dataset-tool load --format FORMAT file
        # Supported formats: coco, yolo, pascal_voc
        ```
    """
    return list(_LOADERS.keys())


def list_exporters() -> list[str]:
    """List all registered exporters.

    Returns the names of all exporter plugins currently registered in the system.
    Useful for displaying available export formats or building conversion tools.

    Returns:
        List of exporter names as strings, in no particular order.

    Example:
        Display available export formats:

        ```python
        from boxlab.dataset.registry import list_exporters

        exporters = list_exporters()
        print("Available export formats:")
        for exporter in exporters:
            print(f"  - {exporter}")

        # Output:
        # Available export formats:
        #   - coco
        #   - yolo
        #   - pascal_voc
        ```

    Example:
        Validate user input:

        ```python
        from boxlab.dataset.registry import list_exporters


        def select_export_format(user_format: str) -> str:
            available = list_exporters()

            if user_format not in available:
                raise ValueError(
                    f"Invalid format '{user_format}'. "
                    f"Choose from: {', '.join(available)}"
                )

            return user_format


        # Usage
        try:
            format_choice = select_export_format("coco")
            print(f"Selected: {format_choice}")
        except ValueError as e:
            print(e)
        ```

    Example:
        Build a format converter:

        ```python
        from boxlab.dataset.registry import (
            list_exporters,
            get_loader,
            get_exporter,
        )


        def convert_dataset(
            input_file: str,
            input_format: str,
            output_formats: list[str],
            output_dir: str,
        ):
            # Load dataset
            loader = get_loader(input_format)
            dataset = loader.load(input_file)

            # Export to all requested formats
            available_exporters = list_exporters()
            for fmt in output_formats:
                if fmt not in available_exporters:
                    print(
                        f"Warning: Skipping unknown format '{fmt}'"
                    )
                    continue

                exporter = get_exporter(fmt)
                exporter.export(dataset, f"{output_dir}/{fmt}")
                print(f"Exported to {fmt}")


        # Convert COCO to YOLO and Pascal VOC
        convert_dataset(
            "data.json", "coco", ["yolo", "pascal_voc"], "output"
        )
        ```
    """
    return list(_EXPORTERS.keys())


def get_loader_info() -> dict[str, dict[str, t.Any]]:
    """Get information about all registered loaders.

    Retrieves detailed information about all registered loader plugins, including
    their names, descriptions, and supported file extensions. This is useful for
    building user interfaces, documentation, or diagnostic tools.

    Returns:
        Dictionary mapping loader names to their information dictionaries.
        Each info dictionary contains:
            - name (str): The loader's identifier
            - description (str): Human-readable description
            - supported_extensions (list[str]): List of file extensions

    Example:
        Display loader information:

        ```python
        from boxlab.dataset.registry import get_loader_info

        info = get_loader_info()

        for name, details in info.items():
            print(f"\nLoader: {name}")
            print(f"  Description: {details['description']}")
            print(
                f"  Extensions: {', '.join(details['supported_extensions'])}"
            )

        # Output:
        # Loader: coco
        #   Description: Load COCO JSON format datasets
        #   Extensions: .json
        #
        # Loader: yolo
        #   Description: Load YOLO format datasets
        #   Extensions: .yaml, .yml, .txt
        ```

    Example:
        Auto-detect format from file extension:

        ```python
        from pathlib import Path
        from boxlab.dataset.registry import (
            get_loader_info,
            get_loader,
        )


        def auto_load_dataset(file_path: str):
            file_ext = Path(file_path).suffix.lower()

            # Find compatible loader
            loader_info = get_loader_info()
            for name, info in loader_info.items():
                if file_ext in info["supported_extensions"]:
                    print(f"Detected format: {name}")
                    loader = get_loader(name)
                    return loader.load(file_path)

            raise ValueError(
                f"No loader found for extension '{file_ext}'"
            )


        # Usage
        dataset = auto_load_dataset(
            "annotations.json"
        )  # Auto-detects COCO
        ```

    Example:
        Generate documentation:

        ```python
        from boxlab.dataset.registry import get_loader_info


        def generate_loader_docs():
            info = get_loader_info()

            docs = "# Supported Input Formats\n\n"
            for name, details in sorted(info.items()):
                docs += f"## {name.upper()}\n\n"
                docs += f"{details['description']}\n\n"

                if details["supported_extensions"]:
                    exts = ", ".join(
                        details["supported_extensions"]
                    )
                    docs += f"**File Extensions:** {exts}\n\n"

            return docs


        print(generate_loader_docs())
        ```

    Example:
        Build a CLI with format hints:

        ```python
        import argparse
        from boxlab.dataset.registry import get_loader_info


        def create_parser():
            parser = argparse.ArgumentParser()

            # Get available formats
            loader_info = get_loader_info()
            format_choices = list(loader_info.keys())

            parser.add_argument(
                "--format",
                choices=format_choices,
                help="Input format",
            )

            # Add format descriptions to help
            help_text = "\n\nSupported formats:\n"
            for name, info in loader_info.items():
                help_text += f"  {name}: {info['description']}\n"

            parser.epilog = help_text

            return parser


        parser = create_parser()
        args = parser.parse_args()
        ```
    """
    info = {}
    for name, loader_class in _LOADERS.items():
        instance = loader_class()
        info[name] = {
            "name": instance.name,
            "description": instance.description,
            "supported_extensions": instance.supported_extensions,
        }
    return info


def get_exporter_info() -> dict[str, dict[str, t.Any]]:
    """Get information about all registered exporters.

    Retrieves detailed information about all registered exporter plugins, including
    their names, descriptions, and default configurations. This is useful for
    building user interfaces, configuration tools, or documentation.

    Returns:
        Dictionary mapping exporter names to their information dictionaries.
        Each info dictionary contains:
            - name (str): The exporter's identifier
            - description (str): Human-readable description
            - default_config (dict): Default configuration options

    Example:
        Display exporter information:

        ```python
        from boxlab.dataset.registry import get_exporter_info

        info = get_exporter_info()

        for name, details in info.items():
            print(f"\nExporter: {name}")
            print(f"  Description: {details['description']}")
            print(f"  Default Config:")
            for key, value in details["default_config"].items():
                print(f"    {key}: {value}")

        # Output:
        # Exporter: coco
        #   Description: Export to COCO JSON format
        #   Default Config:
        #     copy_images: True
        #     naming_strategy: original
        #     indent: 2
        ```

    Example:
        Build a configuration UI:

        ```python
        from boxlab.dataset.registry import get_exporter_info


        def get_export_options(format_name: str) -> dict:
            info = get_exporter_info()

            if format_name not in info:
                raise ValueError(f"Unknown format: {format_name}")

            # Get default config and allow overrides
            config = info[format_name]["default_config"].copy()
            return config


        # Get YOLO export options
        yolo_config = get_export_options("yolo")
        print(f"YOLO defaults: {yolo_config}")

        # Customize
        yolo_config["copy_images"] = False
        yolo_config["normalize_coords"] = True
        ```

    Example:
        Generate format comparison table:

        ```python
        from boxlab.dataset.registry import get_exporter_info


        def compare_exporters():
            info = get_exporter_info()

            print("Format Comparison:\n")
            print(
                f"{'Format':<15} {'Image Copy':<12} {'Description'}"
            )
            print("-" * 60)

            for name, details in sorted(info.items()):
                copy_images = details["default_config"].get(
                    "copy_images", "N/A"
                )
                desc = details["description"][:30]
                print(f"{name:<15} {str(copy_images):<12} {desc}")


        compare_exporters()
        # Output:
        # Format Comparison:
        #
        # Format          Image Copy   Description
        # ------------------------------------------------------------
        # coco            True         Export to COCO JSON format
        # yolo            True         Export to YOLO format
        ```

    Example:
        Validate export configuration:

        ```python
        from boxlab.dataset.registry import get_exporter_info


        def validate_export_config(
            format_name: str, config: dict
        ) -> dict:
            info = get_exporter_info()

            if format_name not in info:
                raise ValueError(f"Unknown format: {format_name}")

            # Start with defaults
            default_config = info[format_name]["default_config"]
            validated_config = default_config.copy()

            # Override with user config
            for key, value in config.items():
                if key in default_config:
                    validated_config[key] = value
                else:
                    print(
                        f"Warning: Unknown option '{key}' for {format_name}"
                    )

            return validated_config


        # Usage
        user_config = {
            "copy_images": False,
            "invalid_option": "test",  # Will trigger warning
        }

        final_config = validate_export_config("coco", user_config)
        print(final_config)
        ```

    Example:
        Create export presets:

        ```python
        from boxlab.dataset.registry import (
            get_exporter_info,
            get_exporter,
        )


        def create_export_preset(preset_name: str) -> dict:
            presets = {
                "quick": {
                    "copy_images": False,
                    "naming_strategy": "original",
                },
                "production": {
                    "copy_images": True,
                    "naming_strategy": "sequential",
                },
            }

            return presets.get(preset_name, {})


        def export_with_preset(
            dataset, format_name: str, preset_name: str
        ):
            # Get base config
            info = get_exporter_info()
            config = info[format_name]["default_config"].copy()

            # Apply preset
            preset = create_export_preset(preset_name)
            config.update(preset)

            # Export
            exporter = get_exporter(format_name)
            exporter.export(
                dataset, f"output/{preset_name}", **config
            )


        # Usage
        export_with_preset(my_dataset, "coco", "quick")
        ```
    """
    info = {}
    for name, exporter_class in _EXPORTERS.items():
        instance = exporter_class()
        info[name] = {
            "name": instance.name,
            "description": instance.description,
            "default_config": instance.get_default_config(),
        }
    return info

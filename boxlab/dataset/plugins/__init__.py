from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    import os

    from boxlab.dataset import Dataset
    from boxlab.dataset import SplitRatio


class NamingStrategy(t.Protocol):
    """Protocol for file naming strategies.

    This protocol defines the interface for generating file names when exporting
    datasets. Custom naming strategies can be implemented by creating classes that
    follow this protocol.

    Example:
        Implementing a custom naming strategy:

        ```python
        class CustomNamingStrategy:
            def gen_name(
                self, origin: str, source: str | None, image_id: str
            ) -> str:
                # Generate name with source prefix
                if source:
                    return f"{source}_{image_id}_{origin}"
                return f"{image_id}_{origin}"


        # Use with exporter
        strategy = CustomNamingStrategy()
        exporter.export(
            dataset, output_dir="output/", naming_strategy=strategy
        )
        ```

    Example:
        Simple naming strategy that preserves original names:

        ```python
        class OriginalNamingStrategy:
            def gen_name(
                self, origin: str, source: str | None, image_id: str
            ) -> str:
                return origin
        ```
    """

    def gen_name(self, origin: str, source: str | None, image_id: str, /) -> str:
        """Generate a new file name.

        Args:
            origin: Original file name (e.g., "image001.jpg").
            source: Source name if available (e.g., "camera1"), None otherwise.
            image_id: Unique image identifier (e.g., "img_12345").

        Returns:
            Generated file name as a string.

        Example:
            ```python
            strategy = MyNamingStrategy()
            new_name = strategy.gen_name(
                origin="photo.jpg",
                source="camera_front",
                image_id="001",
            )
            print(new_name)  # Output depends on implementation
            ```
        """


class LoaderPlugin(abc.ABC):
    """Base class for dataset loaders.

    LoaderPlugin provides the abstract interface for implementing dataset loaders
    that can read various object detection dataset formats. Subclasses must
    implement the abstract methods to support specific formats like COCO, YOLO, etc.

    This class handles dataset loading with validation and format detection
    capabilities. Each loader plugin should focus on a specific dataset format
    and implement the necessary parsing logic.

    Example:
        Implementing a custom loader:

        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.plugins import LoaderPlugin
        import json


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


        # Use the loader
        loader = CustomLoader()
        dataset = loader.load("path/to/dataset.json")
        ```

    Example:
        Using a loader with validation:

        ```python
        loader = CustomLoader()

        # Validate before loading
        if loader.validate("dataset.json"):
            dataset = loader.load("dataset.json")
            print(f"Loaded {len(dataset)} images")
        else:
            print("Invalid dataset file")
        ```
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Plugin name (e.g., 'coco', 'yolo').

        Returns:
            A unique lowercase string identifying this loader.

        Example:
            ```python
            class COCOLoader(LoaderPlugin):
                @property
                def name(self) -> str:
                    return "coco"
            ```
        """

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Plugin description.

        Returns:
            A human-readable description of what this loader does.

        Example:
            ```python
            class COCOLoader(LoaderPlugin):
                @property
                def description(self) -> str:
                    return "Load datasets in COCO JSON format"
            ```
        """

    @property
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions (e.g., ['.json', '.yaml']).

        Returns:
            List of file extensions this loader can handle, including the dot.
            Return empty list if not applicable.

        Example:
            ```python
            class COCOLoader(LoaderPlugin):
                @property
                def supported_extensions(self) -> list[str]:
                    return [".json"]


            class YOLOLoader(LoaderPlugin):
                @property
                def supported_extensions(self) -> list[str]:
                    return [".yaml", ".yml", ".txt"]
            ```
        """
        return []

    @abc.abstractmethod
    def load(
        self,
        path: str | os.PathLike[str],
        **kwargs: t.Any,
    ) -> Dataset:
        """Load dataset from path.

        This method should parse the dataset file(s) at the given path and
        construct a Dataset object with all images, annotations, and categories.

        Args:
            path: Path to dataset file or directory. Can be a JSON file,
                YAML file, or directory containing dataset files.
            **kwargs: Additional loader-specific parameters. Common options:
                - image_root (str): Root directory for image files
                - source_name (str): Name to tag this data source
                - strict (bool): Whether to fail on parse errors

        Returns:
            A populated Dataset instance containing all loaded data.

        Raises:
            FileNotFoundError: If the specified path doesn't exist.
            ValueError: If the dataset format is invalid or corrupted.
            PermissionError: If files cannot be read due to permissions.

        Example:
            ```python
            class COCOLoader(LoaderPlugin):
                def load(self, path, **kwargs):
                    dataset = Dataset(name="coco")
                    image_root = kwargs.get("image_root", ".")

                    with open(path) as f:
                        data = json.load(f)

                    # Load categories
                    for cat in data["categories"]:
                        dataset.add_category(cat["id"], cat["name"])

                    # Load images and annotations
                    # ... implementation details ...

                    return dataset


            # Usage
            loader = COCOLoader()
            dataset = loader.load(
                "annotations.json",
                image_root="/data/images",
                source_name="train2017",
            )
            ```
        """

    def validate(self, path: str | os.PathLike[str]) -> bool:
        """Check if this loader can handle the given path.

        Performs basic validation to determine if the file or directory at the
        given path appears to be in a format this loader can handle. This is
        typically used for automatic format detection.

        Args:
            path: Path to validate. Can be a file or directory.

        Returns:
            True if this loader can likely handle the path, False otherwise.

        Note:
            This method performs basic checks (existence, extension). It does not
            guarantee that load() will succeed, as it doesn't validate the full
            file contents.

        Example:
            ```python
            loader = COCOLoader()

            if loader.validate("dataset.json"):
                dataset = loader.load("dataset.json")
            else:
                print("Not a valid COCO format file")
            ```

        Example:
            Custom validation logic:

            ```python
            class YOLOLoader(LoaderPlugin):
                def validate(self, path):
                    # Call parent validation first
                    if not super().validate(path):
                        return False

                    # Additional YOLO-specific checks
                    path = pathlib.Path(path)
                    if path.is_file() and path.suffix in [
                        ".yaml",
                        ".yml",
                    ]:
                        # Check for YOLO-specific keys
                        with open(path) as f:
                            data = yaml.safe_load(f)
                            return "names" in data and "path" in data

                    return False
            ```
        """
        import pathlib

        path = pathlib.Path(path)

        if not path.exists():
            return False

        # Check file extension if supported_extensions is defined
        if self.supported_extensions:
            return path.suffix.lower() in self.supported_extensions

        return True


class ExporterPlugin(abc.ABC):
    """Base class for dataset exporters.

    ExporterPlugin provides the abstract interface for implementing dataset exporters
    that can write datasets to various object detection formats. Subclasses must
    implement the abstract methods to support specific formats like COCO, YOLO, etc.

    This class handles dataset export with support for train/val/test splits,
    custom naming strategies, and optional image copying. Each exporter plugin
    should focus on a specific output format.

    Example:
        Implementing a custom exporter:

        ```python
        from boxlab.dataset import Dataset, SplitRatio
        from boxlab.dataset.plugins import ExporterPlugin
        import json
        import shutil
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


        # Use the exporter
        exporter = CustomExporter()
        exporter.export(
            dataset,
            output_dir="output/custom",
            split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
            seed=42,
        )
        ```

    Example:
        Using an exporter with custom configuration:

        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.plugins import ExporterPlugin

        exporter = MyExporter()

        # Get default configuration
        config = exporter.get_default_config()
        print(
            config
        )  # {'copy_images': True, 'naming_strategy': 'original'}

        # Export with custom settings
        exporter.export(
            dataset=my_dataset,
            output_dir="output/",
            copy_images=False,
            indent=4,  # Custom parameter
        )
        ```
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Plugin name (e.g., 'coco', 'yolo').

        Returns:
            A unique lowercase string identifying this exporter.

        Example:
            ```python
            class COCOExporter(ExporterPlugin):
                @property
                def name(self) -> str:
                    return "coco"
            ```
        """

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Plugin description.

        Returns:
            A human-readable description of what this exporter does.

        Example:
            ```python
            class COCOExporter(ExporterPlugin):
                @property
                def description(self) -> str:
                    return "Export datasets to COCO JSON format"
            ```
        """

    @property
    def default_extension(self) -> str:
        """Default file extension for exported files.

        Returns:
            File extension string including the dot (e.g., ".json", ".txt").
            Return empty string if not applicable.

        Example:
            ```python
            class COCOExporter(ExporterPlugin):
                @property
                def default_extension(self) -> str:
                    return ".json"


            class YOLOExporter(ExporterPlugin):
                @property
                def default_extension(self) -> str:
                    return ".txt"
            ```
        """
        return ""

    @abc.abstractmethod
    def export(
        self,
        dataset: Dataset,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
        naming_strategy: NamingStrategy | None = None,
        copy_images: bool = True,
        **kwargs: t.Any,
    ) -> None:
        """Export dataset to output directory.

        This method should write the dataset to disk in the format supported by
        this exporter. It should handle creating output directories, optionally
        splitting the dataset, copying images, and writing annotation files.

        Args:
            dataset: The Dataset instance to export.
            output_dir: Path to the output directory. Will be created if it
                doesn't exist.
            split_ratio: Optional SplitRatio object defining train/val/test
                proportions. If None, exports entire dataset without splitting.
            seed: Random seed for reproducible splits. Only used if split_ratio
                is provided.
            naming_strategy: Optional NamingStrategy instance for generating
                image file names. If None, uses original file names.
            copy_images: If True, copies image files to output directory.
                If False, only writes annotation files.
            **kwargs: Additional exporter-specific parameters. Common options:
                - indent (int): JSON indentation level
                - include_metadata (bool): Whether to include extra metadata
                - compress (bool): Whether to compress output files

        Raises:
            ValueError: If export parameters are invalid (e.g., invalid split_ratio).
            OSError: If file operations fail (permission denied, disk full, etc.).
            DatasetError: If dataset is empty or malformed.

        Example:
            ```python
            from boxlab.dataset import Dataset, SplitRatio
            from pathlib import Path


            class COCOExporter(ExporterPlugin):
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

                    # Handle splits
                    if split_ratio:
                        splits = dataset.split(split_ratio, seed=seed)
                    else:
                        splits = {"all": list(dataset.images.keys())}

                    # Export each split
                    for split_name, image_ids in splits.items():
                        # Create COCO format dictionary
                        coco_data = {
                            "images": [],
                            "annotations": [],
                            "categories": [],
                        }

                        # Add categories
                        for (
                            cat_id,
                            cat_name,
                        ) in dataset.categories.items():
                            coco_data["categories"].append({
                                "id": cat_id,
                                "name": cat_name,
                            })

                        # Add images and annotations
                        # ... implementation ...

                        # Write JSON
                        output_file = output_dir / f"{split_name}.json"
                        with open(output_file, "w") as f:
                            json.dump(coco_data, f, indent=2)

                        # Copy images if requested
                        if copy_images:
                            # ... copy logic ...
                            pass


            # Usage
            exporter = COCOExporter()
            exporter.export(
                dataset=my_dataset,
                output_dir="output/coco",
                split_ratio=SplitRatio(train=0.8, val=0.1, test=0.1),
                seed=42,
                copy_images=True,
                indent=4,
            )
            ```

        Example:
            Exporting without splits:

            ```python
            exporter = COCOExporter()
            exporter.export(
                dataset=my_dataset,
                output_dir="output/full_dataset",
                copy_images=False,  # Only export annotations
            )
            ```
        """

    def get_default_config(self) -> dict[str, t.Any]:
        """Get default configuration for this exporter.

        Returns:
            Dictionary of default configuration values that will be used
            if not overridden in export() call.

        Note:
            Subclasses can override this to provide format-specific defaults.

        Example:
            ```python
            class YOLOExporter(ExporterPlugin):
                def get_default_config(self) -> dict[str, t.Any]:
                    return {
                        "copy_images": True,
                        "naming_strategy": "original",
                        "normalize_coords": True,
                        "include_yaml": True,
                    }


            # Usage
            exporter = YOLOExporter()
            config = exporter.get_default_config()
            print(config["normalize_coords"])  # True
            ```

        Example:
            Using default config in export:

            ```python
            class CustomExporter(ExporterPlugin):
                def get_default_config(self) -> dict[str, t.Any]:
                    return {
                        "copy_images": True,
                        "naming_strategy": "original",
                        "compression": "zip",
                    }

                def export(self, dataset, output_dir, **kwargs):
                    # Merge with defaults
                    config = self.get_default_config()
                    config.update(kwargs)

                    # Use configuration
                    if config["compression"] == "zip":
                        # ... compression logic ...
                        pass
            ```
        """
        return {
            "copy_images": True,
            "naming_strategy": "original",
        }

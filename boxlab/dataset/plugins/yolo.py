from __future__ import annotations

import logging
import os
import pathlib
import shutil
import typing as t

import PIL.Image as Image
import yaml

from boxlab.dataset import Dataset
from boxlab.dataset.plugins import ExporterPlugin
from boxlab.dataset.plugins import LoaderPlugin
from boxlab.dataset.plugins import NamingStrategy
from boxlab.dataset.plugins.naming import OriginalNaming
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import BBox
from boxlab.dataset.types import ImageInfo

if t.TYPE_CHECKING:
    from boxlab.dataset.types import SplitRatio

logger = logging.getLogger(__name__)


class YOLOLoader(LoaderPlugin):
    """YOLOv5/YOLOv8 format dataset loader.

    This loader handles datasets in YOLO format, which consists of:
    - A YAML configuration file (data.yaml) defining classes and paths
    - Images organized in directories (typically images/train, images/val,
      images/test)
    - Label files in TXT format with normalized coordinates (labels/train,
      etc.)

    The loader supports both YOLOv5 and YOLOv8 format specifications,
    automatically handling different category naming conventions (dict or list
    format in YAML).

    Label Format:
        Each line in a label file represents one object:
        <class_id> <x_center> <y_center> <width> <height>
        All coordinates are normalized to [0, 1] range.
    """

    @property
    def name(self) -> str:
        """Get the loader name.

        Returns:
            The string "yolo".
        """
        return "yolo"

    @property
    def description(self) -> str:
        """Get the loader description.

        Returns:
            Description string for YOLOv5/YOLOv8 format.
        """
        return "YOLOv5/YOLOv8 format"

    @property
    def supported_extensions(self) -> list[str]:
        """Get supported file extensions.

        Returns:
            List containing [".yaml", ".yml"].
        """
        return [".yaml", ".yml"]

    def load(
        self,
        path: str | os.PathLike[str],
        name: str | None = None,
        splits: str | list[str] | None = None,
        yaml_file: str = "data.yaml",
        **_kwargs: t.Any,
    ) -> Dataset:
        """Load YOLO format dataset.

        Loads a YOLO dataset from the specified directory. The directory should
        contain a YAML configuration file and subdirectories for images and
        labels.

        Args:
            path: Path to YOLO dataset root directory.
            name: Optional custom name for the dataset. If None, uses directory
                name.
            splits: Which split(s) to load. Can be:
                - None: Load all splits (train, val, test)
                - str: Load single split (e.g., "train")
                - list[str]: Load specific splits (e.g., ["train", "val"])
            yaml_file: Name of YAML configuration file. Defaults to "data.yaml".
            **_kwargs: Additional parameters (currently unused, reserved for
                future extensions).

        Returns:
            Loaded Dataset instance containing all images, annotations, and
            categories.

        Raises:
            FileNotFoundError: If the YAML configuration file is not found.
            ValueError: If YAML configuration is missing required 'names' field.
        """
        yolo_dir = pathlib.Path(path)
        yaml_path = yolo_dir / yaml_file

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Load YAML configuration
        with yaml_path.open(mode="r") as f:
            yaml_config = yaml.safe_load(f)

        dataset_name = name or yolo_dir.name
        dataset = Dataset(name=dataset_name)

        logger.info(f"Loading YOLOv5 dataset from {yolo_dir}")

        # Load categories
        self._load_categories(yaml_config, dataset)

        logger.info(f"Loaded {len(dataset.categories)} categories")

        # Determine splits to load
        splits_to_load = self._determine_splits(splits)

        # Load each split
        total_images = 0
        total_annotations = 0

        for split in splits_to_load:
            images_dir = yolo_dir / "images" / split
            labels_dir = yolo_dir / "labels" / split

            if not images_dir.exists():
                logger.warning(f"Images directory not found for {split}: {images_dir}")
                continue

            split_images, split_annotations = self._load_split(
                dataset, images_dir, labels_dir, total_images, total_annotations
            )

            total_images += split_images
            total_annotations += split_annotations

            logger.info(
                f"Loaded {split} split: {split_images} images, {split_annotations} annotations"
            )

        logger.info(f"Total loaded: {total_images} images, {total_annotations} annotations")

        return dataset

    def _load_categories(self, yaml_config: dict[str, t.Any], dataset: Dataset) -> None:
        """Load categories from YAML configuration.

        Handles two YAML formats:
        1. Dict format: names: {0: 'class1', 1: 'class2'}
        2. List format: names: ['class1', 'class2']

        Args:
            yaml_config: Parsed YAML configuration dictionary.
            dataset: Dataset instance to populate with categories.

        Raises:
            ValueError: If 'names' field is missing from YAML configuration.

        Note:
            Category IDs are 1-indexed internally (YOLO uses 0-indexed).
        """
        if "names" not in yaml_config:
            raise ValueError("YAML configuration must contain 'names' field")

        names = yaml_config["names"]

        if isinstance(names, dict):
            # Format: {0: 'class1', 1: 'class2'}
            for cat_id, cat_name in names.items():
                dataset.add_category(int(cat_id) + 1, cat_name)
        elif isinstance(names, list):
            # Format: ['class1', 'class2']
            for i, cat_name in enumerate(names):
                dataset.add_category(i + 1, cat_name)

    def _determine_splits(self, splits: str | list[str] | None) -> list[str]:
        """Determine which splits to load.

        Args:
            splits: User-specified splits (None, string, or list of strings).

        Returns:
            List of split names to load.
        """
        if splits is None:
            return ["train", "val", "test"]
        if isinstance(splits, str):
            return [splits]
        return splits

    def _load_split(
        self,
        dataset: Dataset,
        images_dir: pathlib.Path,
        labels_dir: pathlib.Path,
        image_id_offset: int,
        ann_id_offset: int,
    ) -> tuple[int, int]:
        """Load a single split (train, val, or test).

        Args:
            dataset: Dataset instance to populate.
            images_dir: Directory containing image files.
            labels_dir: Directory containing label files.
            image_id_offset: Starting offset for image IDs.
            ann_id_offset: Starting offset for annotation IDs.

        Returns:
            Tuple of (images_loaded, annotations_loaded) counts.
        """
        image_files = sorted(images_dir.glob("*"))
        image_files = [
            f
            for f in image_files
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]

        images_loaded = 0
        annotations_loaded = 0

        for img_path in image_files:
            try:
                # Load image
                with Image.open(img_path) as img:
                    width, height = img.size

                img_id = str(image_id_offset + images_loaded + 1)
                img_info = ImageInfo(
                    image_id=img_id,
                    file_name=img_path.name,
                    width=width,
                    height=height,
                    path=img_path,
                )
                dataset.add_image(img_info, source_name=dataset.name)
                images_loaded += 1

                # Load labels
                if labels_dir.exists():
                    label_path = labels_dir / (img_path.stem + ".txt")
                    if label_path.exists():
                        anns = self._load_annotations(
                            label_path,
                            img_id,
                            width,
                            height,
                            dataset,
                            ann_id_offset + annotations_loaded,
                        )
                        annotations_loaded += anns

            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

        return images_loaded, annotations_loaded

    def _load_annotations(
        self,
        label_path: pathlib.Path,
        img_id: str,
        width: int,
        height: int,
        dataset: Dataset,
        ann_id_start: int,
    ) -> int:
        """Load annotations from a YOLO label file.

        Each line in the label file represents one bounding box:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

        Args:
            label_path: Path to the label TXT file.
            img_id: Image ID these annotations belong to.
            width: Image width in pixels.
            height: Image height in pixels.
            dataset: Dataset instance to add annotations to.
            ann_id_start: Starting annotation ID for this file.

        Returns:
            Number of annotations successfully loaded.

        Note:
            Invalid lines are logged and skipped. Coordinates are converted
            from normalized [0,1] to absolute pixel values.
        """
        ann_count = 0

        with label_path.open(mode="r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(f"Invalid format in {label_path}:{line_num}")
                        continue

                    class_idx = int(parts[0])
                    cx_norm = float(parts[1])
                    cy_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Convert to absolute coordinates
                    cx = cx_norm * width
                    cy = cy_norm * height
                    w = w_norm * width
                    h = h_norm * height

                    bbox = BBox.from_cxcywh(cx, cy, w, h)

                    # Convert to 1-indexed category ID
                    cat_id = class_idx + 1
                    cat_name = dataset.get_category_name(cat_id)

                    if cat_name is None:
                        logger.warning(f"Unknown category ID: {cat_id} in {label_path}:{line_num}")
                        continue

                    annotation = Annotation(
                        bbox=bbox,
                        category_id=cat_id,
                        category_name=cat_name,
                        image_id=img_id,
                        annotation_id=str(ann_id_start + ann_count + 1),
                        area=bbox.area,
                        iscrowd=0,
                    )
                    dataset.add_annotation(annotation)
                    ann_count += 1

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing {label_path}:{line_num}: {e}")
                    continue

        return ann_count


class YOLOExporter(ExporterPlugin):
    """YOLOv5/YOLOv8 format dataset exporter.

    This exporter converts datasets to YOLO format, creating:
    - A data.yaml configuration file with class definitions and paths
    - Image files organized in split subdirectories (images/train, etc.)
    - Label files in TXT format with normalized coordinates (labels/train, etc.)

    The exporter supports:
    - Train/val/test splits or single dataset export
    - Custom naming strategies for files
    - Optional image copying (can export annotations only)
    - Unified or standard directory structure

    Output Structure (standard):
        output_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Output Structure (unified):
        output_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── annotations/
            ├── train/
            ├── val/
            └── test/
    """

    @property
    def name(self) -> str:
        """Get the exporter name.

        Returns:
            The string "yolo".
        """
        return "yolo"

    @property
    def description(self) -> str:
        """Get the exporter description.

        Returns:
            Description string for YOLOv5/YOLOv8 format.
        """
        return "YOLOv5/YOLOv8 format"

    @property
    def default_extension(self) -> str:
        """Get default file extension for label files.

        Returns:
            The string ".txt".
        """
        return ".txt"

    def export(
        self,
        dataset: Dataset,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
        naming_strategy: NamingStrategy | None = None,
        copy_images: bool = True,
        unified_structure: bool = False,
        **_kwargs: t.Any,
    ) -> None:
        """Export dataset to YOLO format.

        Creates a YOLO-compatible dataset with proper directory structure,
        label files, and YAML configuration.

        Args:
            dataset: Dataset instance to export.
            output_dir: Output directory path. Will be created if it doesn't
                exist.
            split_ratio: Optional SplitRatio for train/val/test division. If
                None, exports entire dataset as 'train' split.
            seed: Random seed for reproducible splits. Only used if split_ratio
                is provided.
            naming_strategy: Strategy for generating output file names. If None,
                uses OriginalNaming (preserves original filenames).
            copy_images: If True, copies image files to output directory. If
                False, only creates label files.
            unified_structure: If True, uses 'annotations' directory instead of
                labels'. Useful for compatibility with some training frameworks.
            **_kwargs: Additional parameters (currently unused, reserved for
                future extensions).

        Note:
            Category IDs in label files are 0-indexed (YOLO convention), even
            though the Dataset uses 1-indexed IDs internally.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        naming_strategy = naming_strategy or OriginalNaming()

        logger.info(f"Exporting YOLOv5 dataset to {output_dir}")

        if split_ratio is None:
            all_image_ids = list(dataset.images.keys())
            self._export_split(
                dataset,
                output_dir,
                "train",
                all_image_ids,
                naming_strategy,
                copy_images,
                unified_structure,
            )
            splits_to_write = ["train"]
        else:
            splits = dataset.split(split_ratio, seed)
            splits_to_write = []
            for split_name, image_ids in splits.items():
                if image_ids:
                    self._export_split(
                        dataset,
                        output_dir,
                        split_name,
                        image_ids,
                        naming_strategy,
                        copy_images,
                        unified_structure,
                    )
                    splits_to_write.append(split_name)

        # Create data.yaml
        self._create_yaml(dataset, output_dir, splits_to_write)

        logger.info(f"YOLOv5 dataset exported to: {output_dir}")

    def _export_split(
        self,
        dataset: Dataset,
        output_dir: pathlib.Path,
        split_name: str,
        image_ids: list[str],
        naming_strategy: NamingStrategy,
        copy_images: bool,
        unified_structure: bool,
    ) -> None:
        """Export a single split (train, val, or test).

        Args:
            dataset: Dataset instance to export from.
            output_dir: Base output directory.
            split_name: Name of the split (e.g., "train", "val", "test").
            image_ids: List of image IDs belonging to this split.
            naming_strategy: Strategy for generating file names.
            copy_images: Whether to copy image files.
            unified_structure: Whether to use 'annotations' instead of 'labels'
                directory.
        """
        # Setup directories
        if unified_structure:
            images_dir = output_dir / "images" / split_name
            labels_dir = output_dir / "annotations" / split_name
        else:
            images_dir = output_dir / "images" / split_name
            labels_dir = output_dir / "labels" / split_name

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        images_copied = 0
        labels_created = 0
        used_names: set[str] = set()

        for img_id in image_ids:
            img_info = dataset.get_image(img_id)
            if img_info is None:
                continue

            # Generate new filename
            source = dataset.get_image_source(img_id)
            new_filename = naming_strategy.gen_name(img_info.file_name, source, img_id)

            # Handle conflicts
            new_filename = self._resolve_filename_conflict(new_filename, used_names)
            used_names.add(new_filename)

            # Copy image
            if copy_images and img_info.path and img_info.path.exists():
                dest_img_path = images_dir / new_filename
                if not dest_img_path.exists():
                    try:
                        shutil.copy2(img_info.path, dest_img_path)
                        images_copied += 1
                    except OSError as e:
                        logger.error(f"Failed to copy {img_info.path}: {e}")
                        continue

            # Create label file
            label_file_name = pathlib.Path(new_filename).stem + ".txt"
            label_path = labels_dir / label_file_name

            anns = dataset.get_annotations(img_id)
            with label_path.open(mode="w") as f:
                for ann in anns:
                    cx, cy, w, h = ann.bbox.cxcywh

                    # Normalize
                    cx_norm = cx / img_info.width
                    cy_norm = cy / img_info.height
                    w_norm = w / img_info.width
                    h_norm = h / img_info.height

                    # 0-indexed class IDs
                    class_idx = ann.category_id - 1 if ann.category_id > 0 else ann.category_id

                    f.write(f"{class_idx} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            labels_created += 1

        logger.info(
            f"YOLOv5 {split_name} exported: {images_copied} images, {labels_created} labels"
        )

    def _create_yaml(self, dataset: Dataset, output_dir: pathlib.Path, splits: list[str]) -> None:
        """Create data.yaml configuration file.

        The YAML file contains:
        - Dataset path
        - Paths to train/val/test splits
        - Number of classes (nc)
        - Class names mapped to 0-indexed IDs

        Args:
            dataset: Dataset instance containing categories.
            output_dir: Directory where data.yaml will be created.
            splits: List of split names that were exported.
        """
        yaml_path = output_dir / "data.yaml"

        with yaml_path.open(mode="w") as f:
            f.write("# YOLOv5 dataset configuration\n")
            f.write(f"# Generated from {dataset.name}\n\n")
            f.write(f"path: {output_dir.absolute()}\n")

            for split_name in ["train", "val", "test"]:
                if split_name in splits:
                    f.write(f"{split_name}: images/{split_name}\n")

            f.write(f"\nnc: {dataset.num_categories()}\n\n")
            f.write("names:\n")
            for i, cat_id in enumerate(sorted(dataset.categories.keys())):
                f.write(f"  {i}: {dataset.categories[cat_id]}\n")

    def _resolve_filename_conflict(self, filename: str, used_names: set[str]) -> str:
        """Resolve filename conflicts by appending a counter.

        Args:
            filename: Original filename.
            used_names: Set of already used filenames.

        Returns:
            Unique filename, possibly with "_N" suffix where N is a counter.
        """
        if filename not in used_names:
            return filename

        stem = pathlib.Path(filename).stem
        suffix = pathlib.Path(filename).suffix
        counter = 1

        while f"{stem}_{counter}{suffix}" in used_names:
            counter += 1

        return f"{stem}_{counter}{suffix}"


# Register plugins
from boxlab.dataset.plugins.registry import register_exporter  # noqa: E402
from boxlab.dataset.plugins.registry import register_loader  # noqa: E402

register_loader("yolo", YOLOLoader)
register_exporter("yolo", YOLOExporter)

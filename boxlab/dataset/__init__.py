from __future__ import annotations

import collections as coll
import logging
import pathlib
import random
import typing as t

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

from boxlab.dataset.types import Annotation
from boxlab.dataset.types import DatasetStatistics
from boxlab.dataset.types import ImageInfo
from boxlab.dataset.types import SplitRatio
from boxlab.exceptions import CategoryConflictError
from boxlab.exceptions import DatasetError

logger = logging.getLogger(__name__)


class Dataset:
    """Base class for dataset management with multi-source support.

    This class provides comprehensive dataset management capabilities including
    loading, exporting, merging, and analyzing object detection datasets. It
    supports multiple data sources and provides flexible file naming strategies
    with detailed statistics.

    Features:
        - Load from COCO or YOLO format
        - Export to COCO or YOLO format
        - Multi-source dataset management
        - Flexible file naming strategies
        - Comprehensive statistics including per-source analysis
        - Category management with conflict resolution
        - Dataset splitting and merging
        - Visualization tools for samples and statistics

    Args:
        name: Dataset name identifier. Defaults to "dataset".

    Attributes:
        name (str): The dataset name identifier.
        images (dict[str, ImageInfo]): Mapping of image IDs to image
            information.
        annotations (dict[str, list[Annotation]]): Mapping of image IDs to
            their annotations.
        categories (dict[int, str]): Mapping of category IDs to category names.
        category_name_to_id (dict[str, int]): Reverse mapping of category names
            to IDs.
        source_info (dict[str, str]): Mapping of image IDs to their source
            names.

    Example:
        Basic dataset creation and usage:

        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.types import ImageInfo, Annotation, BBox

        # Create a new dataset
        dataset = Dataset(name="my_dataset")

        # Add categories
        dataset.add_category(1, "person")
        dataset.add_category(2, "car")

        # Add an image
        img_info = ImageInfo(
            image_id="001",
            file_name="image1.jpg",
            width=640,
            height=480,
            path="/path/to/image1.jpg",
        )
        dataset.add_image(img_info, source_name="camera1")

        # Add an annotation
        annotation = Annotation(
            bbox=BBox(x_min=10, y_min=20, x_max=100, y_max=150),
            category_id=1,
            category_name="person",
            image_id="001",
            annotation_id="ann_001",
        )
        dataset.add_annotation(annotation)

        # Get statistics
        stats = dataset.get_statistics()
        print(f"Total images: {stats['num_images']}")
        print(f"Total annotations: {stats['num_annotations']}")
        ```

    Example:
        Merging multiple datasets:

        ```python
        from boxlab.dataset import Dataset

        # Create two datasets
        dataset1 = Dataset(name="dataset_a")
        dataset2 = Dataset(name="dataset_b")

        # ... add data to both datasets ...

        # Merge datasets
        merged = dataset1.merge(
            dataset2,
            resolve_conflicts="skip",
            preserve_sources=True,
        )

        # Or use the + operator
        merged = dataset1 + dataset2
        ```
    """

    __slots__ = (
        "annotations",
        "categories",
        "category_name_to_id",
        "images",
        "name",
        "source_info",
    )

    def __init__(self, name: str = "dataset") -> None:
        self.name = name
        self.images: dict[str, ImageInfo] = {}
        self.annotations: dict[str, list[Annotation]] = coll.defaultdict(list)
        self.categories: dict[int, str] = {}
        self.category_name_to_id: dict[str, int] = {}
        self.source_info: dict[str, str] = {}  # image_id -> source_name
        logger.debug(f"Initialized dataset: {name}")

    # Category Management ======================================================

    def add_category(self, category_id: int, category_name: str) -> None:
        """Add a category to the dataset.

        Args:
            category_id: Unique identifier for the category.
            category_name: Human-readable name for the category.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            dataset.add_category(1, "person")
            dataset.add_category(2, "car")
            dataset.add_category(3, "bicycle")
            ```
        """
        self.categories[category_id] = category_name
        self.category_name_to_id[category_name] = category_id
        logger.debug(f"Added category: {category_id} -> {category_name}")

    def get_category_name(self, category_id: int) -> str | None:
        """Get category name by ID.

        Args:
            category_id: The category ID to look up.

        Returns:
            The category name if found, None otherwise.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            dataset.add_category(1, "person")

            name = dataset.get_category_name(1)
            print(name)  # Output: "person"
            ```
        """
        return self.categories.get(category_id)

    def get_category_id(self, category_name: str) -> int | None:
        """Get category ID by name.

        Args:
            category_name: The category name to look up.

        Returns:
            The category ID if found, None otherwise.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            dataset.add_category(1, "person")

            cat_id = dataset.get_category_id("person")
            print(cat_id)  # Output: 1
            ```
        """
        return self.category_name_to_id.get(category_name)

    def fix_duplicate_categories(self) -> dict[int, int]:
        """Fix duplicate category names by merging them.

        When multiple category IDs map to the same category name, this method
        consolidates them into a single canonical ID (the smallest ID) and
        remaps all affected annotations.

        Returns:
            A mapping from old category IDs to new (canonical) category IDs.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # Suppose categories were loaded with duplicates:
            # {1: "person", 2: "car", 5: "person"}

            mapping = dataset.fix_duplicate_categories()
            # mapping = {1: 1, 2: 2, 5: 1}
            # Now only categories {1: "person", 2: "car"} remain
            ```
        """
        logger.info(f"Fixing duplicate categories in dataset: {self.name}")

        name_to_ids: dict[str, list[int]] = coll.defaultdict(list)
        for cat_id, cat_name in self.categories.items():
            name_to_ids[cat_name].append(cat_id)

        category_mapping: dict[int, int] = {}
        duplicates_found = 0

        for cat_name, cat_ids in name_to_ids.items():
            if len(cat_ids) > 1:
                cat_ids.sort()
                canonical_id = cat_ids[0]
                duplicates_found += len(cat_ids) - 1

                logger.warning(
                    f"Found duplicate category '{cat_name}' with IDs {cat_ids}, "
                    f"keeping ID {canonical_id}"
                )

                for cat_id in cat_ids:
                    category_mapping[cat_id] = canonical_id
            else:
                category_mapping[cat_ids[0]] = cat_ids[0]

        if duplicates_found == 0:
            logger.info("No duplicate categories found")
            return category_mapping

        # Rebuild categories
        new_categories: dict[int, str] = {}
        new_category_name_to_id: dict[str, int] = {}

        for cat_name, cat_ids in name_to_ids.items():
            canonical_id = min(cat_ids)
            new_categories[canonical_id] = cat_name
            new_category_name_to_id[cat_name] = canonical_id

        self.categories = new_categories
        self.category_name_to_id = new_category_name_to_id

        # Remap annotations
        annotations_updated = 0
        for img_id in list(self.annotations.keys()):
            new_anns = []
            for ann in self.annotations[img_id]:
                new_cat_id = category_mapping[ann.category_id]
                if ann.category_id != new_cat_id:
                    new_ann = Annotation(
                        bbox=ann.bbox,
                        category_id=new_cat_id,
                        category_name=ann.category_name,
                        image_id=ann.image_id,
                        annotation_id=ann.annotation_id,
                        area=ann.area,
                        iscrowd=ann.iscrowd,
                    )
                    new_anns.append(new_ann)
                    annotations_updated += 1
                else:
                    new_anns.append(ann)

            self.annotations[img_id] = new_anns

        logger.info(
            f"Fixed {duplicates_found} duplicate categories, "
            f"updated {annotations_updated} annotations"
        )

        return category_mapping

    # Image Management =========================================================

    def add_image(self, image_info: ImageInfo, source_name: str | None = None) -> None:
        """Add image metadata to dataset with optional source tracking.

        Args:
            image_info: ImageInfo object containing image metadata.
            source_name: Optional name of the data source. Useful for tracking
                which dataset or camera the image came from.

        Example:
            ```python
            from boxlab.dataset import Dataset
            from boxlab.dataset.types import ImageInfo

            dataset = Dataset(name="my_dataset")

            img_info = ImageInfo(
                image_id="001",
                file_name="image1.jpg",
                width=1920,
                height=1080,
                path="/data/images/image1.jpg",
            )

            dataset.add_image(img_info, source_name="camera_front")
            ```
        """
        self.images[image_info.image_id] = image_info
        if source_name:
            self.source_info[image_info.image_id] = source_name
        logger.debug(f"Added image: {image_info.image_id} from source: {source_name}")

    def get_image(self, image_id: str) -> ImageInfo | None:
        """Get image information by ID.

        Args:
            image_id: The unique identifier of the image.

        Returns:
            ImageInfo object if found, None otherwise.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images ...

            img_info = dataset.get_image("001")
            if img_info:
                print(f"Image: {img_info.file_name}")
                print(f"Size: {img_info.width}x{img_info.height}")
            ```
        """
        return self.images.get(image_id)

    def get_image_source(self, image_id: str) -> str | None:
        """Get the source name for an image.

        Args:
            image_id: The unique identifier of the image.

        Returns:
            Source name if tracked, None otherwise.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images with sources ...

            source = dataset.get_image_source("001")
            print(f"Image source: {source}")  # Output: "camera_front"
            ```
        """
        return self.source_info.get(image_id)

    def get_sources(self) -> set[str]:
        """Get all unique source names in the dataset.

        Returns:
            Set of unique source names.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images from multiple sources ...

            sources = dataset.get_sources()
            print(f"Data sources: {sources}")
            # Output: {'camera_front', 'camera_rear', 'dataset_a'}
            ```
        """
        return set(self.source_info.values())

    # Annotation Management ====================================================

    def add_annotation(self, annotation: Annotation) -> None:
        """Add annotation to dataset.

        Args:
            annotation: Annotation object containing bounding box and category
                info.

        Example:
            ```python
            from boxlab.dataset import Dataset
            from boxlab.dataset.types import Annotation, BBox

            dataset = Dataset(name="my_dataset")

            annotation = Annotation(
                bbox=BBox(x_min=100, y_min=50, x_max=200, y_max=150),
                category_id=1,
                category_name="person",
                image_id="001",
                annotation_id="ann_001",
            )

            dataset.add_annotation(annotation)
            ```
        """
        self.annotations[annotation.image_id].append(annotation)

    def get_annotations(self, image_id: str) -> list[Annotation]:
        """Get all annotations for a specific image.

        Args:
            image_id: The unique identifier of the image.

        Returns:
            List of Annotation objects for the specified image. Returns empty
            list if no annotations found.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images and annotations ...

            annotations = dataset.get_annotations("001")
            print(f"Found {len(annotations)} annotations")

            for ann in annotations:
                print(
                    f"Category: {ann.category_name}, BBox: {ann.bbox}"
                )
            ```
        """
        return self.annotations.get(image_id, [])

    # Dataset Information ======================================================

    def num_images(self) -> int:
        """Get total number of images in the dataset.

        Returns:
            Number of images.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images ...

            print(f"Total images: {dataset.num_images()}")
            ```
        """
        return len(self.images)

    def num_annotations(self) -> int:
        """Get total number of annotations in the dataset.

        Returns:
            Total count of all annotations across all images.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add annotations ...

            print(f"Total annotations: {dataset.num_annotations()}")
            ```
        """
        return sum(len(anns) for anns in self.annotations.values())

    def num_categories(self) -> int:
        """Get total number of categories in the dataset.

        Returns:
            Number of unique categories.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add categories ...

            print(f"Total categories: {dataset.num_categories()}")
            ```
        """
        return len(self.categories)

    # Statistics ===============================================================

    @t.overload
    def get_statistics(self, by_source: t.Literal[False] = False) -> DatasetStatistics: ...
    @t.overload
    def get_statistics(self, by_source: t.Literal[True]) -> dict[str, DatasetStatistics]: ...
    def get_statistics(
        self,
        by_source: bool = False,
    ) -> DatasetStatistics | dict[str, DatasetStatistics]:
        """Calculate dataset statistics.

        Computes comprehensive statistics about the dataset including image
        counts, annotation counts, category distribution, and bounding box area
        metrics.

        Args:
            by_source: If True, return statistics grouped by source name.
                If False, return overall statistics for the entire dataset.

        Returns:
            If by_source is False: A DatasetStatistics object with overall
                stats.
            If by_source is True: A dict mapping source names to their
                respective DatasetStatistics objects.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add data ...

            # Get overall statistics
            stats = dataset.get_statistics()
            print(f"Images: {stats['num_images']}")
            print(f"Annotations: {stats['num_annotations']}")
            print(
                "Avg annotations per image: "
                f"{stats['avg_annotations_per_image']:.2f}"
            )

            # Get statistics by source
            stats_by_source = dataset.get_statistics(by_source=True)
            for source, source_stats in stats_by_source.items():
                print(f"\nSource: {source}")
                print(f"  Images: {source_stats['num_images']}")
            ```
        """
        if by_source:
            return self._get_statistics_by_source()

        return self._calculate_statistics(self.images.keys())

    def _calculate_statistics(self, image_ids: t.Iterable[str]) -> DatasetStatistics:
        """Calculate statistics for a specific set of images.

        Args:
            image_ids: Iterable of image IDs to calculate statistics for.

        Returns:
            DatasetStatistics object containing computed metrics.
        """
        image_ids = list(image_ids)

        total_annotations = 0
        category_counts: coll.Counter[str] = coll.Counter()
        annotations_per_image: list[int] = []
        bbox_areas: list[float] = []

        for img_id in image_ids:
            anns = self.get_annotations(img_id)
            annotations_per_image.append(len(anns))
            total_annotations += len(anns)

            for ann in anns:
                category_counts[ann.category_name] += 1
                bbox_areas.append(ann.get_area())

        return DatasetStatistics(
            num_images=len(image_ids),
            num_annotations=total_annotations,
            num_categories=self.num_categories(),
            category_distribution=dict(category_counts),
            avg_annotations_per_image=float(np.mean(annotations_per_image))
            if annotations_per_image
            else 0.0,
            std_annotations_per_image=float(np.std(annotations_per_image))
            if annotations_per_image
            else 0.0,
            min_annotations_per_image=min(annotations_per_image) if annotations_per_image else 0,
            max_annotations_per_image=max(annotations_per_image) if annotations_per_image else 0,
            avg_bbox_area=float(np.mean(bbox_areas)) if bbox_areas else 0.0,
            median_bbox_area=float(np.median(bbox_areas)) if bbox_areas else 0.0,
        )

    def _get_statistics_by_source(self) -> dict[str, DatasetStatistics]:
        """Get statistics grouped by source.

        Returns:
            Dictionary mapping source names to their DatasetStatistics.
        """
        stats_by_source = {}

        # Group images by source
        source_images: dict[str, list[str]] = coll.defaultdict(list)
        for img_id, source in self.source_info.items():
            source_images[source].append(img_id)

        # Calculate stats for each source
        for source, img_ids in source_images.items():
            stats_by_source[source] = self._calculate_statistics(img_ids)

        return stats_by_source

    def print_statistics(self, by_source: bool = False) -> None:
        """Print dataset statistics to console.

        Args:
            by_source: If True, print statistics for each source separately.
                If False, print overall statistics.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add data ...

            # Print overall statistics
            dataset.print_statistics()

            # Print per-source statistics
            dataset.print_statistics(by_source=True)
            ```
        """
        if by_source:
            self._print_statistics_by_source()
        else:
            self._print_single_statistics(self.get_statistics(by_source=False), self.name)

    def _print_single_statistics(self, stats: DatasetStatistics, title: str) -> None:
        """Print statistics for a single dataset or source.

        Args:
            stats: DatasetStatistics object to print.
            title: Title to display in the output.
        """
        print(f"\n{'=' * 60}")
        print(f"Dataset Statistics: {title}")
        print(f"{'=' * 60}")
        print(f"Total Images: {stats['num_images']}")
        print(f"Total Annotations: {stats['num_annotations']}")
        print(f"Total Categories: {stats['num_categories']}")
        print("\nAnnotations per Image:")
        print(f"  Average: {stats['avg_annotations_per_image']:.2f}")
        print(f"  Std Dev: {stats['std_annotations_per_image']:.2f}")
        print(f"  Min: {stats['min_annotations_per_image']}")
        print(f"  Max: {stats['max_annotations_per_image']}")
        print("\nBounding Box Area:")
        print(f"  Average: {stats['avg_bbox_area']:.2f}")
        print(f"  Median: {stats['median_bbox_area']:.2f}")
        print("\nCategory Distribution:")
        for cat_name, count in sorted(
            stats["category_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {cat_name}: {count}")
        print(f"{'=' * 60}\n")

    def _print_statistics_by_source(self) -> None:
        """Print statistics grouped by source."""
        stats_by_source = self.get_statistics(by_source=True)

        # Print overall statistics first
        print("\n" + "=" * 80)
        print(f"MULTI-SOURCE DATASET OVERVIEW: {self.name}")
        print("=" * 80)
        print(f"Total Sources: {len(stats_by_source)}")
        print(f"Sources: {', '.join(stats_by_source.keys())}")

        overall_stats = self.get_statistics(by_source=False)
        print(f"\nOverall Images: {overall_stats['num_images']}")
        print(f"Overall Annotations: {overall_stats['num_annotations']}")
        print("=" * 80)

        # Print per-source statistics
        for source, stats in stats_by_source.items():
            self._print_single_statistics(stats, f"{self.name} - Source: {source}")

    # Dataset Operations =======================================================

    def merge(
        self,
        other: Dataset,
        resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
        preserve_sources: bool = True,
    ) -> Dataset:
        """Merge another dataset into a new dataset.

        Creates a new dataset containing all images, annotations, and
        categories from both datasets. Handles category conflicts according to
        the specified resolution strategy.

        Args:
            other: Another Dataset object to merge with this one.
            resolve_conflicts: Strategy for handling category name conflicts:
                - "skip": Use the existing category ID (default)
                - "rename": Rename the conflicting category from other dataset
                - "error": Raise CategoryConflictError
            preserve_sources: If True, maintain source tracking information
                from both datasets.

        Returns:
            A new Dataset object containing the merged data.

        Raises:
            CategoryConflictError: If resolve_conflicts is "error" and a
                category name conflict is detected.
            DatasetError: If category mapping fails during merge.

        Example:
            ```python
            from boxlab.dataset import Dataset

            # Create two datasets
            dataset_a = Dataset(name="dataset_a")
            dataset_b = Dataset(name="dataset_b")

            # ... populate datasets ...

            # Merge with default settings
            merged = dataset_a.merge(dataset_b)

            # Merge with conflict resolution
            merged = dataset_a.merge(
                dataset_b,
                resolve_conflicts="rename",
                preserve_sources=True,
            )

            # Using the + operator (equivalent to merge with defaults)
            merged = dataset_a + dataset_b
            ```
        """
        logger.info(f"Merging '{other.name}' into '{self.name}'")

        merged = Dataset(name=f"{self.name}_merged")

        # Merge categories
        category_mapping: dict[int, int] = {}
        next_category_id = max(self.categories.keys()) + 1 if self.categories else 1

        for cat_id, cat_name in self.categories.items():
            merged.add_category(cat_id, cat_name)
            category_mapping[cat_id] = cat_id

        for cat_id, cat_name in other.categories.items():
            if cat_name in merged.category_name_to_id:
                if resolve_conflicts == "skip":
                    category_mapping[cat_id] = merged.category_name_to_id[cat_name]
                elif resolve_conflicts == "rename":
                    new_name = f"{cat_name}_other"
                    merged.add_category(next_category_id, new_name)
                    category_mapping[cat_id] = next_category_id
                    next_category_id += 1
                elif resolve_conflicts == "error":
                    raise CategoryConflictError(cat_name, f"Category name conflict: {cat_name}")
            else:
                merged.add_category(cat_id, cat_name)
                category_mapping[cat_id] = cat_id

        # Merge from self
        for img_id, img_info in self.images.items():
            source = self.source_info.get(img_id, self.name) if preserve_sources else None
            merged.add_image(img_info, source_name=source)
            for ann in self.get_annotations(img_id):
                merged.add_annotation(ann)

        # Merge from other
        image_id_offset = (
            max(int(img_id) for img_id in merged.images if img_id.isdigit()) + 1
            if merged.images
            else 1
        )

        for img_id, img_info in other.images.items():
            if img_id in merged.images:
                new_img_id = str(image_id_offset)
                image_id_offset += 1
            else:
                new_img_id = img_id

            new_img_info = ImageInfo(
                image_id=new_img_id,
                file_name=img_info.file_name,
                width=img_info.width,
                height=img_info.height,
                path=img_info.path,
            )

            source = other.source_info.get(img_id, other.name) if preserve_sources else None
            merged.add_image(new_img_info, source_name=source)

            for ann in other.get_annotations(img_id):
                new_cat_name = merged.get_category_name(category_mapping[ann.category_id])
                if new_cat_name is None:
                    raise DatasetError(f"Category mapping failed for {ann.category_id}")

                new_ann = Annotation(
                    bbox=ann.bbox,
                    category_id=category_mapping[ann.category_id],
                    category_name=new_cat_name,
                    image_id=new_img_id,
                    annotation_id=ann.annotation_id,
                    area=ann.area,
                    iscrowd=ann.iscrowd,
                )
                merged.add_annotation(new_ann)

        logger.info(f"Merge completed: {merged.num_images()} images")
        return merged

    def split(self, split_ratio: SplitRatio, seed: int | None = None) -> dict[str, list[str]]:
        """Split dataset into train, validation, and test sets.

        Randomly shuffles and divides the dataset images according to the
        specified ratios. Useful for creating training/validation/test splits
        for machine learning.

        Args:
            split_ratio: SplitRatio object defining the proportions for train,
                validation, and test sets. Must sum to 1.0.
            seed: Optional random seed for reproducible splits. If None, the
                split will be non-deterministic.

        Returns:
            Dictionary with keys "train", "val", and "test", each mapping to
            a list of image IDs in that split.

        Raises:
            ValueError: If split_ratio proportions don't sum to 1.0 (raised by
                split_ratio.validate()).

        Example:
            ```python
            from boxlab.dataset import Dataset
            from boxlab.dataset.types import SplitRatio

            dataset = Dataset(name="my_dataset")
            # ... populate dataset ...

            # Define split ratios: 70% train, 20% val, 10% test
            split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

            # Split with fixed seed for reproducibility
            splits = dataset.split(split_ratio, seed=42)

            print(f"Train images: {len(splits['train'])}")
            print(f"Val images: {len(splits['val'])}")
            print(f"Test images: {len(splits['test'])}")

            # Access specific split
            train_image_ids = splits["train"]
            ```
        """
        split_ratio.validate()

        image_ids = list(self.images.keys())
        if seed is not None:
            random.seed(seed)
        random.shuffle(image_ids)

        total = len(image_ids)
        train_end = int(total * split_ratio.train)
        val_end = train_end + int(total * split_ratio.val)

        return {
            "train": image_ids[:train_end],
            "val": image_ids[train_end:val_end],
            "test": image_ids[val_end:],
        }

    # Visualization ============================================================

    def visualize_sample(
        self,
        image_id: str,
        figsize: tuple[int, int] = (12, 8),
        show_labels: bool = True,
        save_path: pathlib.Path | None = None,
    ) -> None:
        """Visualize a single image with its annotations.

        Displays the image with bounding boxes and category labels overlaid.
        Each category is assigned a unique color, and annotations are drawn
        as rectangles with optional text labels.

        Args:
            image_id: The unique identifier of the image to visualize.
            figsize: Tuple of (width, height) for the matplotlib figure size.
                Defaults to (12, 8).
            show_labels: If True, display category names above bounding boxes.
                Defaults to True.
            save_path: Optional path to save the visualization as an image file.
                If None, only displays the plot.

        Raises:
            DatasetError: If the image is not found or has no path defined.

        Example:
            ```python
            from pathlib import Path
            from boxlab.dataset import Dataset

            dataset = Dataset(name="my_dataset")
            # ... populate dataset ...

            # Display a sample image
            dataset.visualize_sample("001")

            # Save visualization to file
            dataset.visualize_sample(
                "001",
                figsize=(16, 10),
                show_labels=True,
                save_path=Path("output/sample_001.png"),
            )
            ```
        """
        img_info = self.get_image(image_id)
        if img_info is None or img_info.path is None:
            raise DatasetError(f"Image {image_id} not found or has no path")

        img = Image.open(img_info.path)
        anns = self.get_annotations(image_id)
        source = self.get_image_source(image_id)

        _fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_categories()))  # type: ignore
        category_colors = {cat_id: colors[i] for i, cat_id in enumerate(self.categories.keys())}

        for ann in anns:
            bbox = ann.bbox
            color = category_colors[ann.category_id]

            rect = patches.Rectangle(
                (bbox.x_min, bbox.y_min),
                bbox.x_max - bbox.x_min,
                bbox.y_max - bbox.y_min,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            if show_labels:
                ax.text(
                    bbox.x_min,
                    bbox.y_min - 5,
                    ann.category_name,
                    color="white",
                    fontsize=10,
                    bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 2},
                )

        ax.axis("off")
        title = f"Image: {img_info.file_name} | Annotations: {len(anns)}"
        if source:
            title += f" | Source: {source}"
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Visualization saved to: {save_path}")

        plt.show()

    def visualize_category_distribution(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_path: pathlib.Path | None = None,
    ) -> None:
        """Visualize category distribution as a bar chart.

        Creates a bar chart showing the number of annotations for each category
        in the dataset. Useful for understanding class balance and distribution.

        Args:
            figsize: Tuple of (width, height) for the matplotlib figure size.
                Defaults to (12, 6).
            save_path: Optional path to save the visualization as an image file.
                If None, only displays the plot.

        Example:
            ```python
            from pathlib import Path
            from boxlab.dataset import Dataset

            dataset = Dataset(name="my_dataset")
            # ... populate dataset ...

            # Display category distribution
            dataset.visualize_category_distribution()

            # Save to file
            dataset.visualize_category_distribution(
                figsize=(16, 8),
                save_path=Path("output/category_distribution.png"),
            )
            ```
        """
        logger.info(f"Visualizing category distribution for dataset: {self.name}")

        stats = self.get_statistics(by_source=False)
        cat_dist = stats["category_distribution"]

        if not cat_dist:
            logger.warning("No categories to visualize")
            return

        categories = list(cat_dist.keys())
        counts = list(map(float, cat_dist.values()))

        _fig, ax = plt.subplots(1, figsize=figsize)
        bars = ax.bar(categories, counts, color="skyblue", edgecolor="navy", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Category Distribution - {self.name}", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Category distribution saved to: {save_path}")

        plt.show()

    # Magic Methods ============================================================

    def __add__(self, other: object) -> Dataset:
        """Enable merging datasets using the + operator.

        Args:
            other: Another Dataset object to merge.

        Returns:
            A new merged Dataset.

        Raises:
            TypeError: If other is not a Dataset instance.

        Example:
            ```python
            dataset_a = Dataset(name="dataset_a")
            dataset_b = Dataset(name="dataset_b")

            # Merge using + operator
            merged = dataset_a + dataset_b
            ```
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.merge(other)

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            Number of images.

        Example:
            ```python
            dataset = Dataset(name="my_dataset")
            # ... add images ...

            print(f"Dataset contains {len(dataset)} images")
            ```
        """
        return self.num_images()

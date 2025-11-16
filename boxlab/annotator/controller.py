from __future__ import annotations

import datetime
import logging
import pathlib
import typing as t

from boxlab.annotator.workspace import Workspace
from boxlab.dataset import Dataset
from boxlab.dataset.plugins.naming import OriginalNaming
from boxlab.dataset.plugins.naming import PrefixNaming
from boxlab.dataset.plugins.naming import SequentialNaming
from boxlab.dataset.plugins.naming import UUIDNaming
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import ImageInfo

if t.TYPE_CHECKING:
    from boxlab.dataset.plugins import NamingStrategy

logger = logging.getLogger(__name__)


class AnnotationController:
    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.current_split: str | None = None
        self.current_index: int = 0
        self.image_ids_by_split: dict[str, list[str]] = {}

        # Cache for modified annotations
        self.modified_annotations: dict[str, list[Annotation]] = {}

        # Original annotations
        self.original_annotations: dict[str, list[Annotation]] = {}

        # Workspace
        self.workspace: Workspace | None = None
        self.workspace_path: str | None = None  # Current workspace file path
        self.workspace_modified: bool = False  # Track if workspace has unsaved changes
        self.dataset_path: str = ""
        self.dataset_format: str = ""
        self.loaded_splits: list[str] = []

        # Audit mode
        self.audit_mode: bool = False
        self.audit_status: dict[str, str] = {}

        # Tags
        self.available_tags: list[str] = []
        self.image_tags: dict[str, list[str]] = {}

        # Audit comments
        self.audit_comments: dict[str, str] = {}

    def set_audit_comment(self, image_id: str, comment: str) -> None:
        """Set audit comment for an image."""
        self.audit_comments[image_id] = comment
        self.set_workspace_modified()

    def get_audit_comment(self, image_id: str) -> str:
        """Get audit comment for an image."""
        return self.audit_comments.get(image_id, "")

    def set_workspace_modified(self, modified: bool = True) -> None:
        """Mark workspace as modified or unmodified."""
        self.workspace_modified = modified

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved workspace changes."""
        return self.workspace_modified

    def add_tag(self, tag: str) -> None:
        """Add a new tag."""
        if tag and tag not in self.available_tags:
            self.available_tags.append(tag)
            self.set_workspace_modified()

    def set_image_tags(self, image_id: str, tags: list[str]) -> None:
        """Set tags for an image."""
        self.image_tags[image_id] = tags
        self.set_workspace_modified()

    def get_image_tags(self, image_id: str) -> list[str]:
        """Get tags for an image."""
        return self.image_tags.get(image_id, [])

    def update_annotations(self, image_id: str, annotations: list[Annotation]) -> None:
        """Cache annotation modifications for an image."""
        self.modified_annotations[image_id] = annotations.copy()
        self.set_workspace_modified()
        logger.debug(f"Cached {len(annotations)} annotations for image {image_id}")

    def set_audit_status(
        self, image_id: str, status: t.Literal["approved", "rejected", "pending"]
    ) -> None:
        """Set audit status for current image."""
        self.audit_status[image_id] = status
        self.set_workspace_modified()
        logger.info(f"Image {image_id} marked as {status}")

    def save_workspace(self, filepath: str | None = None) -> None:
        """Save current workspace to file.

        Args:
            filepath: Path to save to. If None, uses current workspace_path.
        """
        from boxlab.annotator.workspace import Workspace

        if filepath:
            self.workspace_path = filepath

        if not self.workspace_path:
            raise ValueError("No workspace path specified")

        workspace = Workspace()
        workspace.set_dataset_info(self.dataset_path, self.dataset_format, self.loaded_splits)
        workspace.set_current_position(self.current_split or "", self.current_index)

        # Save original annotations
        for img_id, anns in self.original_annotations.items():
            workspace.set_original_annotations(img_id, anns)

        # Save modified annotations
        for img_id, anns in self.modified_annotations.items():
            workspace.set_modified_annotations(img_id, anns)

        # Save audit info
        workspace.set_audit_mode(self.audit_mode)
        for img_id, status in self.audit_status.items():
            workspace.set_audit_status(img_id, status)

        # Save tags
        workspace.set_available_tags(self.available_tags)
        for img_id, tags in self.image_tags.items():
            workspace.set_image_tags(img_id, tags)

        # Save audit comments
        for img_id, comment in self.audit_comments.items():
            workspace.set_audit_comment(img_id, comment)

        workspace.save(self.workspace_path)
        self.set_workspace_modified(False)
        logger.info(f"Workspace saved to {self.workspace_path}")

    def load_workspace(self, filepath: str) -> None:
        """Load workspace from file."""
        from boxlab.annotator.workspace import Workspace

        workspace = Workspace.load(filepath)

        # Load dataset
        self.load_dataset(
            workspace.data["dataset_path"],
            workspace.data["dataset_format"],
            workspace.data["loaded_splits"],
        )

        # Restore position
        if workspace.data["current_split"]:
            self.current_split = workspace.data["current_split"]
            self.current_index = workspace.data["current_index"]

        # Restore modified annotations
        self.modified_annotations.clear()
        for img_id in workspace.data["modified_annotations"]:
            anns = workspace.get_modified_annotations(img_id)
            if anns:
                self.modified_annotations[img_id] = anns

        # Restore audit info
        self.audit_mode = workspace.data["audit_mode"]
        self.audit_status = workspace.data["audit_status"].copy()

        # Restore tags
        self.available_tags = workspace.data.get("available_tags", [])
        self.image_tags = workspace.data.get("image_tags", {})

        # Restore audit comments
        self.audit_comments = workspace.data.get("audit_comments", {})

        # Set workspace path and mark as unmodified
        self.workspace_path = filepath
        self.set_workspace_modified(False)

        logger.info(f"Workspace loaded from {filepath}")

    def generate_audit_report_json(self, output_path: str) -> None:
        """Generate JSON audit report with tags."""
        from boxlab.annotator.workspace import Workspace

        workspace = Workspace()
        workspace.set_dataset_info(self.dataset_path, self.dataset_format, self.loaded_splits)

        # Add all annotations (with prefixed IDs)
        for img_id, anns in self.original_annotations.items():
            workspace.set_original_annotations(img_id, anns)

        for img_id, anns in self.modified_annotations.items():
            workspace.set_modified_annotations(img_id, anns)

        # Add audit status
        workspace.data["audit_status"] = self.audit_status.copy()

        # Add tags
        workspace.set_available_tags(self.available_tags)
        for img_id, tags in self.image_tags.items():
            workspace.set_image_tags(img_id, tags)

        # Build image info map
        image_info_map: dict[str, tuple[str, str]] = {}
        for split_name, dataset in self.datasets.items():
            for img_id, img_info in dataset.images.items():
                # Extract original source from image_sources or use split name
                source: str = dataset.get_image_source(img_id) or split_name

                # If source is still "Unknown" and has underscore in filename
                if source == "Unknown" and "_" in img_info.file_name:
                    source = img_info.file_name.split("_")[0]

                image_info_map[img_id] = (img_info.file_name, source)

        # Generate JSON report (IDs will include prefixes, which is fine for tracking)
        workspace.generate_audit_report_json(output_path, image_info_map)
        logger.info(f"Audit report JSON generated: {output_path}")

    def load_raw_images(
        self,
        path: str,
        categories: list[str],
    ) -> None:
        """Load raw images from directory and create empty dataset.

        Args:
            path: Path to directory containing images
            categories: List of category names
        """
        import pathlib

        from boxlab.dataset import Dataset
        from boxlab.dataset.types import ImageInfo

        logger.info(f"Loading raw images from {path}")

        self.dataset_path = path
        self.dataset_format = "raw"
        self.loaded_splits = ["train"]

        self.datasets.clear()
        self.image_ids_by_split.clear()
        self.modified_annotations.clear()
        self.original_annotations.clear()

        # Create new empty dataset
        dataset = Dataset(name="raw_dataset")

        # Add categories
        for idx, cat_name in enumerate(categories, start=1):
            dataset.add_category(idx, cat_name)

        # Scan directory for images
        path_obj = pathlib.Path(path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        image_files: list[pathlib.Path] = []
        for ext in image_extensions:
            image_files.extend(path_obj.glob(f"*{ext}"))
            image_files.extend(path_obj.glob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError(f"No image files found in {path}")

        # Add images to dataset
        from PIL import Image

        for idx, img_path in enumerate(sorted(image_files), start=1):
            try:
                # Open image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size

                # Create UNIQUE image ID with split prefix
                image_id = f"raw_{idx}_{img_path.stem}"

                img_info = ImageInfo(
                    image_id=image_id,
                    file_name=img_path.name,
                    width=width,
                    height=height,
                    path=img_path,
                )

                dataset.add_image(img_info)
                self.original_annotations[image_id] = []

            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

        self.datasets["train"] = dataset
        self.image_ids_by_split["train"] = list(dataset.images.keys())

        self.current_split = "train"
        self.current_index = 0

        logger.info(f"Loaded {len(dataset.images)} raw images with {len(categories)} categories")

    def load_dataset(
        self,
        path: str,
        format_type: str,
        splits: list[str],
        categories: list[str] | None = None,
        initial_tags: list[str] | None = None,
    ) -> None:
        """Load dataset from path."""
        if format_type == "raw":
            if not categories:
                raise ValueError("Categories must be provided for raw image import")
            self.load_raw_images(path, categories)

            if initial_tags:
                self.available_tags = initial_tags.copy()

            return

        # Existing logic for COCO/YOLO
        self.dataset_path = path
        self.dataset_format = format_type
        self.loaded_splits = splits

        logger.info(f"Loading {format_type} dataset from {path}, splits: {splits}")

        self.datasets.clear()
        self.image_ids_by_split.clear()
        self.modified_annotations.clear()
        self.original_annotations.clear()

        import pathlib

        path_obj = pathlib.Path(path)

        if format_type == "coco":
            for split in splits:
                ann_file = path_obj / f"annotations_{split}.json"
                if ann_file.exists():
                    from boxlab.dataset.plugins.coco import COCOLoader

                    dataset = COCOLoader().load(str(ann_file), name=f"dataset_{split}")

                    # Ensure image_id uniqueness: add split prefix
                    self._ensure_unique_image_ids(dataset, split)

                    self.datasets[split] = dataset
                    self.image_ids_by_split[split] = list(dataset.images.keys())

                    # Store original annotations
                    for img_id in dataset.images:
                        self.original_annotations[img_id] = dataset.get_annotations(img_id).copy()
                else:
                    logger.warning(f"Annotation file not found: {ann_file}")

        elif format_type == "yolo":
            for split in splits:
                split_dir = path_obj / "images" / split
                if split_dir.exists():
                    from boxlab.dataset.plugins.yolo import YOLOLoader

                    dataset = YOLOLoader().load(path, splits=split, name=f"dataset_{split}")

                    self._ensure_unique_image_ids(dataset, split)

                    self.datasets[split] = dataset
                    self.image_ids_by_split[split] = list(dataset.images.keys())

                    # Store original annotations
                    for img_id in dataset.images:
                        self.original_annotations[img_id] = dataset.get_annotations(img_id).copy()
                else:
                    logger.warning(f"Split directory not found: {split_dir}")

        if splits and self.datasets:
            self.current_split = (
                splits[0] if splits[0] in self.datasets else next(iter(self.datasets.keys()))
            )
            self.current_index = 0

        # Initialize tags
        if initial_tags:
            self.available_tags = initial_tags.copy()

        logger.info(f"Loaded {len(self.datasets)} split(s)")

    def _ensure_unique_image_ids(self, dataset: Dataset, split: str) -> None:
        """Ensure image IDs are unique across splits by adding prefix.

        Args:
            dataset: Dataset to modify
            split: Split name (train/val/test)
        """
        # Check if IDs already have split prefix
        sample_id = next(iter(dataset.images.keys())) if dataset.images else None
        if sample_id and sample_id.startswith(f"{split}_"):
            # Already prefixed
            return

        # Need to add prefix
        new_images = {}
        new_annotations = {}

        for old_id, img_info in dataset.images.items():
            new_id = f"{split}_{old_id}"

            # Update image info
            new_img_info = ImageInfo(
                image_id=new_id,
                file_name=img_info.file_name,
                width=img_info.width,
                height=img_info.height,
                path=img_info.path,
            )
            new_images[new_id] = new_img_info

            # Update annotations
            if old_id in dataset.annotations:
                new_anns = []
                for ann in dataset.annotations[old_id]:
                    new_ann = Annotation(
                        bbox=ann.bbox,
                        category_id=ann.category_id,
                        category_name=ann.category_name,
                        image_id=new_id,  # Update to new ID
                        annotation_id=ann.annotation_id,
                        area=ann.area,
                        iscrowd=ann.iscrowd,
                    )
                    new_anns.append(new_ann)
                new_annotations[new_id] = new_anns

        # Replace with new data
        dataset.images = new_images
        dataset.annotations = new_annotations

        logger.info(f"Added '{split}_' prefix to {len(new_images)} image IDs")

    def auto_backup_workspace(self, backup_dir: str | pathlib.Path) -> str:
        """Create automatic backup of workspace.

        Args:
            backup_dir: Directory to save backup

        Returns:
            Path to backup file
        """
        import pathlib

        backup_dir = pathlib.Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"auto_backup_{timestamp}.cyw"
        backup_path = backup_dir / backup_filename

        self.save_workspace(str(backup_path))
        logger.info(f"Auto-backup created: {backup_path}")

        return str(backup_path)

    def enable_audit_mode(self, enabled: bool) -> None:
        """Enable or disable audit mode."""
        self.audit_mode = enabled

    def get_audit_status(self, image_id: str) -> str:
        """Get audit status for image."""
        return self.audit_status.get(image_id, "pending")

    def get_audit_statistics(self) -> dict[str, int]:
        """Get audit statistics."""
        stats = {"approved": 0, "rejected": 0, "pending": 0}

        for status in self.audit_status.values():
            if status in stats:
                stats[status] += 1

        total_images = self.total_images()
        reviewed_count = len(self.audit_status)
        stats["pending"] = total_images - reviewed_count + stats.get("pending", 0)

        return stats

    def generate_audit_report(self) -> str:
        """Generate audit report."""
        from datetime import datetime

        stats = self.get_audit_statistics()
        total = self.total_images()

        report = "=" * 60 + "\n"
        report += "AUDIT REPORT\n"
        report += "=" * 60 + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Dataset: {self.dataset_path}\n"
        report += f"Format: {self.dataset_format}\n\n"
        report += "Audit Statistics:\n"
        report += "-" * 60 + "\n"
        report += f"Total Images: {total}\n"
        report += (
            f"Approved: {stats['approved']} ({stats['approved'] / total * 100:.1f}%)\n"
            if total > 0
            else "Approved: 0\n"
        )
        report += (
            f"Rejected: {stats['rejected']} ({stats['rejected'] / total * 100:.1f}%)\n"
            if total > 0
            else "Rejected: 0\n"
        )
        report += (
            f"Pending: {stats['pending']} ({stats['pending'] / total * 100:.1f}%)\n"
            if total > 0
            else "Pending: 0\n"
        )
        report += "=" * 60 + "\n"

        return report

    def export_dataset(
        self,
        output_dir: str,
        format_type: str,
        naming_strategy: str,
    ) -> None:
        import pathlib

        logger.info(f"Exporting to {output_dir} as {format_type}")

        # Apply all modified annotations before export
        self._apply_all_modifications()

        # Merge all splits
        if len(self.datasets) > 1:
            # Remove split prefixes before merging to avoid ID conflicts
            cleaned_datasets = []
            for split_name, dataset in self.datasets.items():
                cleaned_dataset = self._remove_split_prefix(dataset, split_name)
                cleaned_datasets.append(cleaned_dataset)

            from boxlab.dataset.io import merge

            merged = merge(*cleaned_datasets, name="exported_dataset")
        else:
            # Single dataset, still need to clean prefix
            dataset = next(iter(self.datasets.values()))
            split_name = next(iter(self.datasets.keys()))
            merged = self._remove_split_prefix(dataset, split_name)

        strategy = self._get_naming_strategy(naming_strategy)
        output_path = pathlib.Path(output_dir)

        if format_type == "coco":
            from boxlab.dataset.plugins.coco import COCOExporter

            COCOExporter().export(merged, output_path, naming_strategy=strategy, copy_images=True)
        elif format_type == "yolo":
            from boxlab.dataset.plugins.yolo import YOLOExporter

            YOLOExporter().export(merged, output_path, naming_strategy=strategy, copy_images=True)

        logger.info(f"Export completed to {output_dir}")

    def _remove_split_prefix(self, dataset: Dataset, split_name: str) -> Dataset:
        """Remove split prefix from image IDs for export.

        Args:
            dataset: Dataset with prefixed IDs
            split_name: Split name (train/val/test)

        Returns:
            New dataset with clean IDs
        """
        from boxlab.dataset import Dataset
        from boxlab.dataset.types import Annotation
        from boxlab.dataset.types import ImageInfo

        # Create new dataset
        new_dataset = Dataset(name=dataset.name)
        new_dataset.categories = dataset.categories.copy()

        prefix = f"{split_name}_"

        # Process images and annotations
        for old_id, img_info in dataset.images.items():
            # Remove prefix if present
            new_id = old_id[len(prefix) :] if old_id.startswith(prefix) else old_id

            # Create new image info
            new_img_info = ImageInfo(
                image_id=new_id,
                file_name=img_info.file_name,
                width=img_info.width,
                height=img_info.height,
                path=img_info.path,
            )
            new_dataset.add_image(new_img_info)

            # Update annotations
            if old_id in dataset.annotations:
                new_anns = []
                for ann in dataset.annotations[old_id]:
                    new_ann = Annotation(
                        bbox=ann.bbox,
                        category_id=ann.category_id,
                        category_name=ann.category_name,
                        image_id=new_id,  # Use new ID
                        annotation_id=ann.annotation_id,
                        area=ann.area,
                        iscrowd=ann.iscrowd,
                    )
                    new_anns.append(new_ann)
                new_dataset.annotations[new_id] = new_anns

        # Preserve source information
        for old_id, source in dataset.source_info.items():
            new_id = old_id[len(prefix) :] if old_id.startswith(prefix) else old_id
            new_dataset.source_info[new_id] = source

        logger.info(f"Removed '{prefix}' prefix from {len(new_dataset.images)} image IDs")

        return new_dataset

    def _apply_all_modifications(self) -> None:
        """Apply all cached annotation modifications to datasets."""
        for img_id, anns in self.modified_annotations.items():
            # Find which split this image belongs to
            for _, dataset in self.datasets.items():
                if img_id in dataset.images:
                    # Update annotations in dataset
                    updated_anns = []
                    for ann in anns:
                        updated_ann = Annotation(
                            bbox=ann.bbox,
                            category_id=ann.category_id,
                            category_name=ann.category_name,
                            image_id=img_id,  # Keep the prefixed ID
                            annotation_id=ann.annotation_id,
                            area=ann.area,
                            iscrowd=ann.iscrowd,
                        )
                        updated_anns.append(updated_ann)

                    dataset.annotations[img_id] = updated_anns
                    logger.debug(f"Applied {len(updated_anns)} annotations to image {img_id}")
                    break

    def _get_naming_strategy(self, name: str) -> NamingStrategy:
        """Get naming strategy by name."""
        strategies: dict[str, NamingStrategy] = {
            "original": OriginalNaming(),
            "prefix": PrefixNaming(),
            "uuid": UUIDNaming(with_source_prefix=False),
            "uuid_prefix": UUIDNaming(with_source_prefix=True),
            "sequential": SequentialNaming(with_source_prefix=False),
            "sequential_prefix": SequentialNaming(with_source_prefix=True),
        }
        return strategies.get(name, OriginalNaming())

    def has_dataset(self) -> bool:
        """Check if dataset is loaded."""
        return len(self.datasets) > 0

    def get_splits(self) -> list[str]:
        return list(self.datasets.keys())

    def set_current_split(self, split: str) -> None:
        if split in self.datasets:
            self.current_split = split
            self.current_index = 0

    def get_images_in_split(self, split: str) -> list[tuple[str, str]]:
        dataset = self.datasets.get(split)
        if dataset:
            return [(img_id, img_info.file_name) for img_id, img_info in dataset.images.items()]
        return []

    def get_current_dataset(self) -> Dataset | None:
        if self.current_split:
            return self.datasets.get(self.current_split)
        return None

    def get_image_info(self, image_id: str) -> ImageInfo | None:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_image(image_id)
        return None

    def get_annotations(self, image_id: str) -> list[Annotation]:
        # Check if we have cached modifications
        if image_id in self.modified_annotations:
            return self.modified_annotations[image_id].copy()

        # Otherwise get from dataset
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_annotations(image_id)
        return []

    def get_image_source(self, image_id: str) -> str | None:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_image_source(image_id)
        return None

    def get_categories(self) -> dict[int, str]:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.categories
        return {}

    def next_image(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if self.current_index < len(images) - 1:
            self.current_index += 1
            return images[self.current_index]

        return None

    def prev_image(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if self.current_index > 0:
            self.current_index -= 1
            return images[self.current_index]

        return None

    def get_current_image_id(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if 0 <= self.current_index < len(images):
            return images[self.current_index]

        return None

    def get_current_index(self) -> int:
        return self.current_index + 1

    def get_split_size(self) -> int:
        if self.current_split:
            return len(self.image_ids_by_split.get(self.current_split, []))
        return 0

    def total_images(self) -> int:
        return sum(len(ids) for ids in self.image_ids_by_split.values())

    def get_dataset_info(self) -> dict[str, int]:
        total_images = self.total_images()
        total_annotations = sum(ds.num_annotations() for ds in self.datasets.values())
        num_categories = len(self.get_categories())

        return {
            "images": total_images,
            "annotations": total_annotations,
            "categories": num_categories,
            "splits": len(self.datasets),
        }

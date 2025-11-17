from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import typing as t

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

    class CocoInfo(t.TypedDict):
        description: str
        version: str
        year: int
        contributor: str

    class CocoCategory(t.TypedDict):
        id: int
        name: str
        supercategory: str

    class CocoImage(t.TypedDict):
        id: int
        file_name: str
        width: int
        height: int

    class CocoAnnotation(t.TypedDict):
        id: int
        image_id: int
        category_id: int
        bbox: list[float]
        area: float
        iscrowd: int
        segmentation: list[list[float]]

    class CocoDataset(t.TypedDict):
        info: CocoInfo
        images: list[CocoImage]
        annotations: list[CocoAnnotation]
        categories: list[CocoCategory]


logger = logging.getLogger(__name__)


class COCOLoader(LoaderPlugin):
    """COCO format dataset loader."""

    @property
    def name(self) -> str:
        return "coco"

    @property
    def description(self) -> str:
        return "Microsoft COCO (Common Objects in Context) format"

    @property
    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load(
        self,
        path: str | os.PathLike[str],
        name: str | None = None,
        images_dir: str | os.PathLike[str] | None = None,
        **_kwargs: t.Any,
    ) -> Dataset:
        """Load COCO dataset.

        Args:
            path: Path to COCO annotation JSON file
            name: Optional dataset name. If not provided, the filename (without
                extension) will be used.
            images_dir: Optional custom path to images directory
            **_kwargs: Additional parameters (ignored)

        Returns:
            Loaded Dataset
        """
        annotation_path = pathlib.Path(path)

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        # Auto-detect images directory
        if images_dir is not None:
            images_dir = pathlib.Path(images_dir)
        else:
            images_dir = self._find_images_dir(annotation_path)

        with annotation_path.open(mode="r") as f:
            coco_data = json.load(f)

        dataset_name = name or annotation_path.stem
        dataset = Dataset(name=dataset_name)

        logger.info(f"Loading COCO dataset from {annotation_path}")

        # Load categories
        for cat in coco_data.get("categories", []):
            dataset.add_category(cat["id"], cat["name"])

        logger.info(f"Loaded {len(dataset.categories)} categories")

        # Load images
        images_found = 0
        images_missing = 0

        for img in coco_data.get("images", []):
            img_path = self._find_image_path(img["file_name"], images_dir)

            if img_path:
                images_found += 1
            else:
                images_missing += 1

            img_info = ImageInfo(
                image_id=str(img["id"]),
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
                path=img_path,
            )
            dataset.add_image(img_info, source_name=dataset_name)

        logger.info(
            f"Loaded {len(dataset.images)} images (found: {images_found}, missing: {images_missing})"
        )

        # Load annotations
        for ann in coco_data.get("annotations", []):
            bbox_xywh = ann["bbox"]
            bbox = BBox.from_xywh(*bbox_xywh)

            cat_name = dataset.get_category_name(ann["category_id"])
            if cat_name is None:
                continue

            annotation = Annotation(
                bbox=bbox,
                category_id=ann["category_id"],
                category_name=cat_name,
                image_id=str(ann["image_id"]),
                annotation_id=str(ann["id"]),
                area=ann.get("area", bbox.area),
                iscrowd=ann.get("iscrowd", 0),
            )
            dataset.add_annotation(annotation)

        logger.info(f"Loaded {dataset.num_annotations()} annotations")

        return dataset

    def _find_images_dir(self, annotation_path: pathlib.Path) -> pathlib.Path | None:
        """Try to find images directory."""
        possible_locations = [
            annotation_path.parent / "images",
            annotation_path.parent.parent / "images",
            annotation_path.parent,
        ]

        for loc in possible_locations:
            if loc.exists() and loc.is_dir():
                return loc

        return None

    def _find_image_path(
        self, filename: str, images_dir: pathlib.Path | None
    ) -> pathlib.Path | None:
        """Try to find image file."""
        if images_dir is None:
            return None

        potential_path = images_dir / filename
        if potential_path.exists():
            return potential_path

        # Try flattened path
        flat_path = images_dir / pathlib.Path(filename).name
        if flat_path.exists():
            return flat_path

        return None


class COCOExporter(ExporterPlugin):
    """COCO format dataset exporter."""

    @property
    def name(self) -> str:
        return "coco"

    @property
    def description(self) -> str:
        return "Microsoft COCO (Common Objects in Context) format"

    @property
    def default_extension(self) -> str:
        return ".json"

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
        """Export to COCO format.

        Args:
            dataset: Dataset to export
            output_dir: Output directory
            split_ratio: Optional split ratios
            seed: Random seed
            naming_strategy: Naming strategy
            copy_images: Whether to copy images
            unified_structure: Use unified directory structure
            **_kwargs: Additional parameters (ignored)
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        naming_strategy = naming_strategy or OriginalNaming()

        logger.info(f"Exporting COCO dataset to {output_dir}")

        image_counter = 0
        ann_counter = 0
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
        else:
            splits = dataset.split(split_ratio, seed)
            for split_name, image_ids in splits.items():
                if image_ids:
                    image_counter, ann_counter = self._export_split(
                        dataset,
                        output_dir,
                        split_name,
                        image_ids,
                        naming_strategy,
                        copy_images,
                        unified_structure,
                        image_counter,
                        ann_counter,
                    )

        logger.info(f"COCO dataset exported to: {output_dir}")

    def _export_split(
        self,
        dataset: Dataset,
        output_dir: pathlib.Path,
        split_name: str,
        image_ids: list[str],
        naming_strategy: NamingStrategy,
        copy_images: bool,
        unified_structure: bool,
        image_counter: int = 0,
        ann_counter: int = 0,
    ) -> tuple[int, int]:
        """Export a single split."""
        # Setup directories
        if unified_structure:
            images_dir = output_dir / "images" / split_name
            annotations_dir = output_dir / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            annotations_dir.mkdir(parents=True, exist_ok=True)
            annotation_file = annotations_dir / f"instances_{split_name}.json"
        else:
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            annotation_file = output_dir / f"annotations_{split_name}.json"

        # Build COCO structure
        coco_output: CocoDataset = {
            "info": {
                "description": dataset.name,
                "version": "1.0",
                "year": 2025,
                "contributor": "6ixGODD",
            },
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Categories
        for cat_id, cat_name in dataset.categories.items():
            coco_output["categories"].append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "object",
            })

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

            # Add image entry
            curr_iid = image_counter
            image_counter += 1

            coco_output["images"].append({
                "id": curr_iid,
                "file_name": new_filename,
                "width": img_info.width,
                "height": img_info.height,
            })

            # Copy image
            if copy_images and img_info.path and img_info.path.exists():
                dest_path = images_dir / new_filename
                if not dest_path.exists():
                    shutil.copy2(img_info.path, dest_path)

            # Add annotations
            for ann in dataset.get_annotations(img_id):
                x, y, w, h = ann.bbox.xywh
                curr_ann_id = ann_counter
                ann_counter += 1

                coco_output["annotations"].append({
                    "id": curr_ann_id,
                    "image_id": curr_iid,
                    "category_id": ann.category_id,
                    "bbox": [x, y, w, h],
                    "area": ann.get_area(),
                    "iscrowd": ann.iscrowd,
                    "segmentation": [],
                })

        # Save annotation file
        with annotation_file.open(mode="w") as f:
            json.dump(coco_output, f, indent=2)

        logger.info(f"COCO {split_name} exported: {len(coco_output['images'])} images")

        return image_counter, ann_counter

    def _resolve_filename_conflict(self, filename: str, used_names: set[str]) -> str:
        """Resolve filename conflicts."""
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

register_loader("coco", COCOLoader)
register_exporter("coco", COCOExporter)

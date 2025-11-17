"""Tests for COCO format plugin."""

from __future__ import annotations

import json
import pathlib

import pytest

from boxlab.dataset import Dataset
from boxlab.dataset.plugins.coco import COCOExporter
from boxlab.dataset.plugins.coco import COCOLoader
from boxlab.dataset.types import SplitRatio


class TestCOCOLoader:
    """Test COCO format loading."""

    def test_load_coco_dataset(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test loading a COCO dataset."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file)

        # Check basic properties
        assert dataset.name == "annotations_train"
        assert dataset.num_images() == 4
        assert dataset.num_annotations() == 5
        assert dataset.num_categories() == 2

    def test_load_coco_categories(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test that categories are loaded correctly."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file)

        assert dataset.get_category_name(1) == "person"
        assert dataset.get_category_name(2) == "car"
        assert dataset.get_category_id("person") == 1
        assert dataset.get_category_id("car") == 2

    def test_load_coco_images(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test that images are loaded with correct metadata."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file)

        img1 = dataset.get_image("1")
        assert img1 is not None
        assert img1.file_name == "image1.jpg"
        assert img1.width == 640
        assert img1.height == 480
        assert img1.path is not None
        assert img1.path.exists()

    def test_load_coco_annotations(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test that annotations are loaded correctly."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file)

        # Image 1 should have 2 annotations
        anns1 = dataset.get_annotations("1")
        assert len(anns1) == 2

        # Check first annotation (person)
        ann1 = anns1[0]
        assert ann1.category_id == 1
        assert ann1.category_name == "person"
        assert ann1.bbox.x_min == 100
        assert ann1.bbox.y_min == 150
        assert ann1.bbox.x_max == 300  # 100 + 200
        assert ann1.bbox.y_max == 350  # 150 + 200

    def test_load_coco_bbox_conversion(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test COCO XYWH to XYXY conversion."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file)

        ann = dataset.get_annotations("1")[1]  # Second annotation (car)

        # COCO format: [50, 50, 100, 100] (x, y, w, h)
        assert ann.bbox.x_min == 50
        assert ann.bbox.y_min == 50
        assert ann.bbox.x_max == 150  # 50 + 100
        assert ann.bbox.y_max == 150  # 50 + 100

    def test_load_coco_with_custom_images_dir(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test loading with custom images directory."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        images_dir = tmp_coco_dataset / "images"

        dataset = loader.load(annotations_file, images_dir=images_dir)

        assert dataset.num_images() == 4

        # Check that image paths are correct
        img1 = dataset.get_image("1")
        assert img1.path == images_dir / "image1.jpg"

    def test_load_coco_missing_images(self, tmp_path: pathlib.Path) -> None:
        """Test loading COCO with missing image files."""
        # Create COCO JSON without image files
        coco_data = {
            "images": [{"id": 1, "file_name": "missing.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": [{"id": 1, "name": "test", "supercategory": "object"}],
        }

        annotations_file = tmp_path / "annotations_train.json"
        with annotations_file.open("w") as f:
            json.dump(coco_data, f)

        loader = COCOLoader()
        dataset = loader.load(annotations_file)

        # Should still load, but path will be None
        assert dataset.num_images() == 1
        img = dataset.get_image("1")
        assert img.path is None or not img.path.exists()

    def test_load_coco_with_custom_name(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test loading with custom dataset name."""
        loader = COCOLoader()

        annotations_file = tmp_coco_dataset / "annotations_train.json"
        dataset = loader.load(annotations_file, name="custom_name")

        assert dataset.name == "custom_name"


class TestCOCOExporter:
    """Test COCO format export."""

    def test_export_coco_basic(self, simple_dataset: Dataset, tmp_path: pathlib.Path) -> None:
        """Test basic COCO export."""
        exporter = COCOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        # Check that files were created
        assert (output_dir / "annotations_train.json").exists()

    def test_export_coco_structure(self, simple_dataset: Dataset, tmp_path: pathlib.Path) -> None:
        """Test exported COCO structure."""
        exporter = COCOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        # Load and verify JSON structure
        with (output_dir / "annotations_train.json").open() as f:
            coco_data = json.load(f)

        assert "info" in coco_data
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data

        assert len(coco_data["images"]) == 1
        assert len(coco_data["annotations"]) == 1
        assert len(coco_data["categories"]) == 3  # person, car, bicycle

    def test_export_coco_bbox_conversion(
        self, simple_dataset: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test that bboxes are converted to COCO XYWH format."""
        exporter = COCOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        with (output_dir / "annotations_train.json").open() as f:
            coco_data = json.load(f)

        ann = coco_data["annotations"][0]
        bbox = ann["bbox"]

        # Original: BBox(10, 20, 100, 150)
        # COCO format: [x, y, w, h]
        assert bbox[0] == 10  # x
        assert bbox[1] == 20  # y
        assert bbox[2] == 90  # width (100 - 10)
        assert bbox[3] == 130  # height (150 - 20)

    def test_export_coco_with_splits(
        self, dataset_with_categories: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test export with train/val/test splits."""
        # Add multiple images
        from boxlab.dataset.types import Annotation
        from boxlab.dataset.types import BBox
        from boxlab.dataset.types import ImageInfo

        for i in range(10):
            img = ImageInfo(f"img{i}", f"test{i}.jpg", 640, 480)
            dataset_with_categories.add_image(img)

            ann = Annotation(
                BBox(0, 0, 50, 50),
                1,
                "person",
                f"img{i}",
            )
            dataset_with_categories.add_annotation(ann)

        exporter = COCOExporter()
        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

        output_dir = tmp_path / "output"
        exporter.export(
            dataset_with_categories,
            output_dir,
            split_ratio=split_ratio,
            seed=42,
            copy_images=False,
            unified_structure=True,
        )

        # Check split files
        assert (output_dir / "annotations" / "instances_train.json").exists()
        assert (output_dir / "annotations" / "instances_val.json").exists()
        assert (output_dir / "annotations" / "instances_test.json").exists()

    def test_export_coco_roundtrip(
        self, tmp_coco_dataset: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Test COCO load -> export -> load roundtrip."""
        # Load original
        loader = COCOLoader()
        original = loader.load(tmp_coco_dataset / "annotations_train.json")

        # Export
        exporter = COCOExporter()
        output_dir = tmp_path / "exported"
        exporter.export(original, output_dir, copy_images=True)

        # Load exported
        reloaded = loader.load(output_dir / "annotations_train.json")

        # Compare
        assert reloaded.num_images() == original.num_images()
        assert reloaded.num_annotations() == original.num_annotations()
        assert reloaded.num_categories() == original.num_categories()


class TestCOCOEdgeCases:
    """Test COCO edge cases and error handling."""

    def test_load_nonexistent_file(self, tmp_path: pathlib.Path) -> None:
        """Test loading non-existent COCO file."""
        loader = COCOLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.json")

    def test_load_empty_coco(self, tmp_path: pathlib.Path) -> None:
        """Test loading empty COCO dataset."""
        coco_data = {"images": [], "annotations": [], "categories": []}

        annotations_file = tmp_path / "empty.json"
        with annotations_file.open("w") as f:
            json.dump(coco_data, f)

        loader = COCOLoader()
        dataset = loader.load(annotations_file)

        assert dataset.num_images() == 0
        assert dataset.num_annotations() == 0
        assert dataset.num_categories() == 0

    def test_export_empty_dataset(self, empty_dataset: Dataset, tmp_path: pathlib.Path) -> None:
        """Test exporting empty dataset."""
        exporter = COCOExporter()

        output_dir = tmp_path / "output"
        exporter.export(empty_dataset, output_dir, copy_images=False)

        # Should create file even if empty
        assert (output_dir / "annotations_train.json").exists()

        with (output_dir / "annotations_train.json").open() as f:
            coco_data = json.load(f)

        assert len(coco_data["images"]) == 0
        assert len(coco_data["annotations"]) == 0

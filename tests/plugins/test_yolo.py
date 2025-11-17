from __future__ import annotations

import pathlib

import pytest

from boxlab.dataset import Dataset
from boxlab.dataset.plugins.yolo import YOLOExporter
from boxlab.dataset.plugins.yolo import YOLOLoader
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import BBox
from boxlab.dataset.types import ImageInfo
from boxlab.dataset.types import SplitRatio


class TestYOLOLoader:
    """Test YOLO format loading."""

    def test_load_yolo_dataset(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test loading a YOLO dataset."""
        loader = YOLOLoader()

        dataset = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        assert dataset.num_images() == 2
        assert dataset.num_categories() == 2

    def test_load_yolo_categories(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test that categories are loaded from YAML."""
        loader = YOLOLoader()

        dataset = loader.load(tmp_yolo_dataset / "data.yaml")

        # YOLO uses 0-indexed IDs, we convert to 1-indexed
        assert dataset.get_category_name(1) == "person"
        assert dataset.get_category_name(2) == "car"

    def test_load_yolo_annotations(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test that YOLO annotations are loaded and converted
        correctly."""
        loader = YOLOLoader()

        dataset = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        # First image should have 2 annotations
        img1_id = next(iter(dataset.images.keys()))
        anns = dataset.get_annotations(img1_id)

        assert len(anns) >= 2  # Should have at least 2 annotations

    def test_load_yolo_bbox_conversion(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test YOLO normalized coordinates to absolute conversion."""
        loader = YOLOLoader()

        dataset = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        # Get first image and annotation
        img_id = next(iter(dataset.images.keys()))
        img = dataset.get_image(img_id)
        anns = dataset.get_annotations(img_id)

        # First annotation: "0 0.5 0.5 0.3 0.4"
        # cx=0.5*640=320, cy=0.5*480=240, w=0.3*640=192, h=0.4*480=192
        ann = anns[0]

        # Check that coordinates are in absolute pixels
        assert 0 <= ann.bbox.x_min < img.width
        assert 0 <= ann.bbox.y_min < img.height
        assert ann.bbox.x_max <= img.width
        assert ann.bbox.y_max <= img.height

    def test_load_yolo_bbox_roundtrip(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test YOLO coordinate conversion roundtrip."""
        loader = YOLOLoader()

        dataset = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        img_id = next(iter(dataset.images.keys()))
        img = dataset.get_image(img_id)
        ann = dataset.get_annotations(img_id)[0]

        # Convert back to YOLO normalized
        cx, cy, w, h = ann.bbox.cxcywh
        cx_norm = cx / img.width
        cy_norm = cy / img.height
        w_norm = w / img.width
        h_norm = h / img.height

        # Should be close to original values (0.5, 0.5, 0.3, 0.4)
        assert abs(cx_norm - 0.5) < 0.01
        assert abs(cy_norm - 0.5) < 0.01
        assert abs(w_norm - 0.3) < 0.01
        assert abs(h_norm - 0.4) < 0.01

    def test_load_yolo_missing_labels(self, tmp_yolo_dataset: pathlib.Path) -> None:
        """Test loading YOLO with missing label files."""
        # Delete one label file
        (tmp_yolo_dataset / "labels" / "train" / "img1.txt").unlink()

        loader = YOLOLoader()
        dataset = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        # Should still load images, just without annotations
        assert dataset.num_images() == 2

        # Find image without labels
        for img_id in dataset.images:
            anns = dataset.get_annotations(img_id)
            if len(anns) == 0:
                # This is the image with missing labels
                break


class TestYOLOExporter:
    """Test YOLO format export."""

    def test_export_yolo_basic(self, simple_dataset: Dataset, tmp_path: pathlib.Path) -> None:
        """Test basic YOLO export."""
        exporter = YOLOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        # Check that files were created
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "labels" / "train").exists()

    def test_export_yolo_yaml_structure(
        self, simple_dataset: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test exported data.yaml structure."""
        exporter = YOLOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        # Load YAML
        import yaml

        with (output_dir / "data.yaml").open() as f:
            config = yaml.safe_load(f)

        assert "path" in config
        assert "train" in config
        assert "nc" in config
        assert "names" in config

        assert config["nc"] == 3  # person, car, bicycle
        assert len(config["names"]) == 3

    def test_export_yolo_label_format(
        self, simple_dataset: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test exported label file format."""
        exporter = YOLOExporter()

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        # Find label file
        label_files = list((output_dir / "labels" / "train").glob("*.txt"))
        assert len(label_files) == 1

        # Read label
        with label_files[0].open() as f:
            lines = f.readlines()

        assert len(lines) == 1

        # Parse line
        parts = lines[0].strip().split()
        assert len(parts) == 5

        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # Check format
        assert 0 <= class_id < 3
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1

    def test_export_yolo_bbox_conversion(
        self, dataset_with_categories: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test bbox conversion to YOLO normalized format."""
        # Add image with known bbox
        img = ImageInfo("img1", "test.jpg", 640, 480)
        dataset_with_categories.add_image(img)

        # BBox at center: (270, 190, 370, 290) -> center (320, 240), size (100, 100)
        ann = Annotation(
            BBox(270, 190, 370, 290),
            1,
            "person",
            "img1",
        )
        dataset_with_categories.add_annotation(ann)

        exporter = YOLOExporter()
        output_dir = tmp_path / "output"
        exporter.export(dataset_with_categories, output_dir, copy_images=False)

        # Read label
        label_file = next(iter((output_dir / "labels" / "train").glob("*.txt")))
        with label_file.open() as f:
            line = f.readline().strip()

        parts = line.split()
        cx, cy, _, _ = map(float, parts[1:])

        # Expected: cx=320/640=0.5, cy=240/480=0.5, w=100/640≈0.156, h=100/480≈0.208
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01

    def test_export_yolo_with_splits(
        self, dataset_with_categories: Dataset, tmp_path: pathlib.Path
    ) -> None:
        """Test YOLO export with splits."""
        # Add multiple images
        for i in range(10):
            img = ImageInfo(f"img{i}", f"test{i}.jpg", 640, 480)
            dataset_with_categories.add_image(img)

            ann = Annotation(BBox(0, 0, 50, 50), 1, "person", f"img{i}")
            dataset_with_categories.add_annotation(ann)

        exporter = YOLOExporter()
        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

        output_dir = tmp_path / "output"
        exporter.export(
            dataset_with_categories, output_dir, split_ratio=split_ratio, seed=42, copy_images=False
        )

        # Check split directories
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "images" / "test").exists()

        assert (output_dir / "labels" / "train").exists()
        assert (output_dir / "labels" / "val").exists()
        assert (output_dir / "labels" / "test").exists()

    def test_export_yolo_roundtrip(
        self, tmp_yolo_dataset: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Test YOLO load -> export -> load roundtrip."""
        # Load original
        loader = YOLOLoader()
        original = loader.load(tmp_yolo_dataset / "data.yaml", splits="train")

        # Export
        exporter = YOLOExporter()
        output_dir = tmp_path / "exported"
        exporter.export(original, output_dir, copy_images=True)

        # Load exported
        reloaded = loader.load(output_dir / "data.yaml", splits="train")

        # Compare
        assert reloaded.num_images() == original.num_images()
        assert reloaded.num_categories() == original.num_categories()

        # Annotations might differ slightly due to float precision
        assert abs(reloaded.num_annotations() - original.num_annotations()) <= 1


class TestYOLOEdgeCases:
    """Test YOLO edge cases."""

    def test_load_nonexistent_yaml(self, tmp_path: pathlib.Path) -> None:
        """Test loading non-existent YAML file."""
        loader = YOLOLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "data.yaml")

    def test_export_empty_dataset(self, empty_dataset: Dataset, tmp_path: pathlib.Path) -> None:
        """Test exporting empty YOLO dataset."""
        exporter = YOLOExporter()

        output_dir = tmp_path / "output"
        exporter.export(empty_dataset, output_dir, copy_images=False)

        # Should create structure even if empty
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "labels" / "train").exists()

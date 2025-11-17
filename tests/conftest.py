from __future__ import annotations

import json
import pathlib

import pytest

from boxlab.annotator import AnnotationController
from boxlab.dataset import Dataset
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import BBox
from boxlab.dataset.types import ImageInfo


@pytest.fixture
def sample_bbox() -> BBox:
    """Create a sample bounding box."""
    return BBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=150.0)


@pytest.fixture
def sample_image_info() -> ImageInfo:
    """Create sample image metadata."""
    return ImageInfo(
        image_id="img_001",
        file_name="sample.jpg",
        width=640,
        height=480,
        path=None,
    )


@pytest.fixture
def sample_annotation(sample_bbox: BBox) -> Annotation:
    """Create a sample annotation."""
    return Annotation(
        bbox=sample_bbox,
        category_id=1,
        category_name="person",
        image_id="img_001",
        annotation_id="ann_001",
        area=None,
        iscrowd=0,
    )


@pytest.fixture
def empty_dataset() -> Dataset:
    """Create an empty dataset."""
    return Dataset(name="test_dataset")


@pytest.fixture
def dataset_with_categories(empty_dataset: Dataset) -> Dataset:
    """Create a dataset with predefined categories."""
    empty_dataset.add_category(1, "person")
    empty_dataset.add_category(2, "car")
    empty_dataset.add_category(3, "bicycle")
    return empty_dataset


@pytest.fixture
def simple_dataset(
    dataset_with_categories: Dataset,
    sample_image_info: ImageInfo,
    sample_annotation: Annotation,
) -> Dataset:
    """Create a simple dataset with one image and one annotation."""
    dataset_with_categories.add_image(sample_image_info, source_name="test_source")
    dataset_with_categories.add_annotation(sample_annotation)
    return dataset_with_categories


@pytest.fixture
def tmp_coco_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary COCO dataset structure."""
    # Create directories
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create sample COCO JSON
    coco_data = {
        "info": {
            "description": "Test Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "test",
        },
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600},
            {"id": 3, "file_name": "image3.jpg", "width": 1024, "height": 768},
            {"id": 4, "file_name": "image4.jpg", "width": 1280, "height": 720},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 200],  # x, y, w, h
                "area": 40000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [200, 200, 150, 150],
                "area": 22500,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 2,
                "bbox": [300, 300, 250, 250],
                "area": 62500,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 4,
                "category_id": 1,
                "bbox": [400, 100, 100, 300],
                "area": 30000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }

    # Save JSON
    annotations_file = tmp_path / "annotations_train.json"
    with annotations_file.open("w") as f:
        json.dump(coco_data, f)

    # Create dummy images (1x1 pixel)
    from PIL import Image

    for img_data in coco_data["images"]:
        img = Image.new("RGB", (img_data["width"], img_data["height"]), color="red")
        img.save(images_dir / img_data["file_name"])

    return tmp_path


@pytest.fixture
def tmp_yolo_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary YOLO dataset structure."""
    # Create directories
    images_train = tmp_path / "images" / "train"
    labels_train = tmp_path / "labels" / "train"
    images_train.mkdir(parents=True)
    labels_train.mkdir(parents=True)

    # Create data.yaml
    yaml_content = f"""# YOLO dataset config
path: {tmp_path}
train: images/train

nc: 2

names:
  0: person
  1: car
"""

    (tmp_path / "data.yaml").write_text(yaml_content)

    # Create dummy images and labels
    from PIL import Image

    # Image 1
    img1 = Image.new("RGB", (640, 480), color="blue")
    img1.save(images_train / "img1.jpg")

    # Label 1 (2 objects)
    label1 = "0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.15 0.2\n"
    (labels_train / "img1.txt").write_text(label1)

    # Image 2
    img2 = Image.new("RGB", (800, 600), color="green")
    img2.save(images_train / "img2.jpg")

    # Label 2 (1 object)
    label2 = "0 0.6 0.7 0.25 0.3\n"
    (labels_train / "img2.txt").write_text(label2)

    return tmp_path


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up test plugins from registry after each test."""
    from boxlab.dataset.plugins.registry import _EXPORTERS
    from boxlab.dataset.plugins.registry import _LOADERS

    # Save original state
    _LOADERS.copy()
    _EXPORTERS.copy()

    yield

    # Restore original state (remove test plugins)
    test_keys = [k for k in _LOADERS if k.startswith("test_") or k == "dummy"]
    for key in test_keys:
        _LOADERS.pop(key, None)

    test_keys = [k for k in _EXPORTERS if k.startswith("test_") or k == "dummy"]
    for key in test_keys:
        _EXPORTERS.pop(key, None)


@pytest.fixture
def controller() -> AnnotationController:
    """Create a new controller instance."""
    return AnnotationController()


@pytest.fixture
def controller_with_images(
    controller: AnnotationController, tmp_coco_dataset: pathlib.Path
) -> AnnotationController:
    """Create controller with loaded COCO dataset."""
    controller.load_dataset(str(tmp_coco_dataset), "coco", ["train"])
    return controller


@pytest.fixture
def controller_with_splits(
    controller: AnnotationController, tmp_coco_dataset: pathlib.Path
) -> AnnotationController:
    """Create controller with multiple splits."""
    from boxlab.dataset import Dataset
    from boxlab.dataset.plugins.coco import COCOLoader

    # Load dataset
    loader = COCOLoader()
    dataset = loader.load(tmp_coco_dataset / "annotations_train.json")

    # Split it
    from boxlab.dataset.types import SplitRatio

    splits = dataset.split(SplitRatio(train=0.7, val=0.3, test=0.0), seed=42)

    # Manually set up controller with splits
    for split_name, img_ids in splits.items():
        if img_ids:
            split_dataset = Dataset(name=f"{dataset.name}_{split_name}")

            # Copy categories
            for cat_id, cat_name in dataset.categories.items():
                split_dataset.add_category(cat_id, cat_name)

            # Copy images and annotations for this split
            for img_id in img_ids:
                img_info = dataset.get_image(img_id)
                if img_info:
                    split_dataset.add_image(img_info, dataset.get_image_source(img_id))

                    for ann in dataset.get_annotations(img_id):
                        split_dataset.add_annotation(ann)

            controller.datasets[split_name] = split_dataset
            controller.image_ids_by_split[split_name] = img_ids

    controller.current_split = "train"
    controller.set_workspace_modified(True)

    return controller

"""Tests for Dataset core functionality."""

from __future__ import annotations

import pytest

from boxlab.dataset import Dataset
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import BBox
from boxlab.dataset.types import ImageInfo
from boxlab.dataset.types import SplitRatio
from boxlab.exceptions import CategoryConflictError
from boxlab.exceptions import ValidationError


class TestDatasetCreation:
    """Test dataset creation."""

    def test_create_empty_dataset(self) -> None:
        """Test creating an empty dataset."""
        dataset = Dataset(name="test")

        assert dataset.name == "test"
        assert len(dataset.images) == 0
        assert len(dataset.annotations) == 0
        assert len(dataset.categories) == 0

    def test_dataset_default_name(self) -> None:
        """Test dataset with default name."""
        dataset = Dataset()
        assert dataset.name == "dataset"


class TestCategoryManagement:
    """Test category management."""

    def test_add_category(self, empty_dataset: Dataset) -> None:
        """Test adding a category."""
        empty_dataset.add_category(1, "person")

        assert empty_dataset.num_categories() == 1
        assert empty_dataset.get_category_name(1) == "person"
        assert empty_dataset.get_category_id("person") == 1

    def test_add_multiple_categories(self, empty_dataset: Dataset) -> None:
        """Test adding multiple categories."""
        empty_dataset.add_category(1, "person")
        empty_dataset.add_category(2, "car")
        empty_dataset.add_category(3, "bicycle")

        assert empty_dataset.num_categories() == 3
        assert empty_dataset.get_category_name(2) == "car"
        assert empty_dataset.get_category_id("bicycle") == 3

    def test_get_nonexistent_category(self, empty_dataset: Dataset) -> None:
        """Test getting non-existent category."""
        assert empty_dataset.get_category_name(999) is None
        assert empty_dataset.get_category_id("nonexistent") is None

    def test_fix_duplicate_categories(self, empty_dataset: Dataset) -> None:
        """Test fixing duplicate category names."""
        # Add duplicate category names with different IDs
        empty_dataset.add_category(1, "person")
        empty_dataset.add_category(5, "person")  # Duplicate name
        empty_dataset.add_category(2, "car")

        # Add images and annotations
        img = ImageInfo("img1", "test.jpg", 640, 480)
        empty_dataset.add_image(img)

        ann1 = Annotation(BBox(0, 0, 10, 10), 1, "person", "img1")
        ann2 = Annotation(BBox(10, 10, 20, 20), 5, "person", "img1")
        empty_dataset.add_annotation(ann1)
        empty_dataset.add_annotation(ann2)

        # Fix duplicates
        mapping = empty_dataset.fix_duplicate_categories()

        # Should keep ID 1, remap 5 to 1
        assert mapping[1] == 1
        assert mapping[5] == 1
        assert mapping[2] == 2

        # Only 2 categories should remain
        assert empty_dataset.num_categories() == 2
        assert empty_dataset.get_category_name(1) == "person"
        assert empty_dataset.get_category_name(5) is None

        # All annotations should now use ID 1
        anns = empty_dataset.get_annotations("img1")
        assert all(ann.category_id == 1 for ann in anns if ann.category_name == "person")


class TestImageManagement:
    """Test image management."""

    def test_add_image(self, empty_dataset: Dataset, sample_image_info: ImageInfo) -> None:
        """Test adding an image."""
        empty_dataset.add_image(sample_image_info)

        assert empty_dataset.num_images() == 1
        assert empty_dataset.get_image("img_001") == sample_image_info

    def test_add_image_with_source(
        self, empty_dataset: Dataset, sample_image_info: ImageInfo
    ) -> None:
        """Test adding an image with source tracking."""
        empty_dataset.add_image(sample_image_info, source_name="camera1")

        assert empty_dataset.get_image_source("img_001") == "camera1"

    def test_get_sources(self, empty_dataset: Dataset) -> None:
        """Test getting all sources."""
        img1 = ImageInfo("img1", "test1.jpg", 640, 480)
        img2 = ImageInfo("img2", "test2.jpg", 640, 480)
        img3 = ImageInfo("img3", "test3.jpg", 640, 480)

        empty_dataset.add_image(img1, source_name="camera1")
        empty_dataset.add_image(img2, source_name="camera2")
        empty_dataset.add_image(img3, source_name="camera1")

        sources = empty_dataset.get_sources()
        assert sources == {"camera1", "camera2"}

    def test_get_nonexistent_image(self, empty_dataset: Dataset) -> None:
        """Test getting non-existent image."""
        assert empty_dataset.get_image("nonexistent") is None


class TestAnnotationManagement:
    """Test annotation management."""

    def test_add_annotation(self, simple_dataset: Dataset) -> None:
        """Test adding an annotation."""
        anns = simple_dataset.get_annotations("img_001")

        assert len(anns) == 1
        assert anns[0].category_name == "person"
        assert anns[0].image_id == "img_001"

    def test_add_multiple_annotations(self, dataset_with_categories: Dataset) -> None:
        """Test adding multiple annotations to same image."""
        img = ImageInfo("img1", "test.jpg", 640, 480)
        dataset_with_categories.add_image(img)

        ann1 = Annotation(BBox(0, 0, 10, 10), 1, "person", "img1")
        ann2 = Annotation(BBox(20, 20, 30, 30), 2, "car", "img1")
        ann3 = Annotation(BBox(40, 40, 50, 50), 1, "person", "img1")

        dataset_with_categories.add_annotation(ann1)
        dataset_with_categories.add_annotation(ann2)
        dataset_with_categories.add_annotation(ann3)

        anns = dataset_with_categories.get_annotations("img1")
        assert len(anns) == 3

    def test_get_annotations_empty(self, empty_dataset: Dataset) -> None:
        """Test getting annotations for image with no annotations."""
        anns = empty_dataset.get_annotations("nonexistent")
        assert anns == []


class TestDatasetStatistics:
    """Test dataset statistics."""

    def test_num_images(self, simple_dataset: Dataset) -> None:
        """Test counting images."""
        assert simple_dataset.num_images() == 1

    def test_num_annotations(self, simple_dataset: Dataset) -> None:
        """Test counting annotations."""
        assert simple_dataset.num_annotations() == 1

    def test_num_categories(self, simple_dataset: Dataset) -> None:
        """Test counting categories."""
        assert simple_dataset.num_categories() == 3  # person, car, bicycle

    def test_get_statistics(self, dataset_with_categories: Dataset) -> None:
        """Test getting dataset statistics."""
        # Add multiple images and annotations
        for i in range(5):
            img = ImageInfo(f"img{i}", f"test{i}.jpg", 640, 480)
            dataset_with_categories.add_image(img)

            # Add varying number of annotations
            for j in range(i + 1):
                ann = Annotation(
                    BBox(j * 10, j * 10, j * 10 + 50, j * 10 + 50),
                    (j % 3) + 1,
                    dataset_with_categories.get_category_name((j % 3) + 1),
                    f"img{i}",
                )
                dataset_with_categories.add_annotation(ann)

        stats = dataset_with_categories.get_statistics()

        assert stats["num_images"] == 5
        assert stats["num_annotations"] == 15  # 1+2+3+4+5
        assert stats["num_categories"] == 3
        assert stats["avg_annotations_per_image"] == 3.0
        assert stats["min_annotations_per_image"] == 1
        assert stats["max_annotations_per_image"] == 5

    def test_get_statistics_by_source(self, empty_dataset: Dataset) -> None:
        """Test getting statistics grouped by source."""
        # Add images from different sources
        for source in ["camera1", "camera2"]:
            for i in range(3):
                img_id = f"{source}_img{i}"
                img = ImageInfo(img_id, f"{img_id}.jpg", 640, 480)
                empty_dataset.add_image(img, source_name=source)

                empty_dataset.add_category(1, "person")
                ann = Annotation(BBox(0, 0, 50, 50), 1, "person", img_id)
                empty_dataset.add_annotation(ann)

        stats_by_source = empty_dataset.get_statistics(by_source=True)

        assert len(stats_by_source) == 2
        assert stats_by_source["camera1"]["num_images"] == 3
        assert stats_by_source["camera2"]["num_images"] == 3


class TestDatasetSplit:
    """Test dataset splitting."""

    def test_split_dataset(self, dataset_with_categories: Dataset) -> None:
        """Test splitting dataset into train/val/test."""
        # Add 10 images
        for i in range(10):
            img = ImageInfo(f"img{i}", f"test{i}.jpg", 640, 480)
            dataset_with_categories.add_image(img)

        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)
        splits = dataset_with_categories.split(split_ratio, seed=42)

        assert len(splits["train"]) == 7
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 1

        # Check all images are accounted for
        all_images = splits["train"] + splits["val"] + splits["test"]
        assert len(all_images) == 10
        assert len(set(all_images)) == 10  # No duplicates

    def test_split_reproducible(self, dataset_with_categories: Dataset) -> None:
        """Test that split is reproducible with same seed."""
        for i in range(10):
            img = ImageInfo(f"img{i}", f"test{i}.jpg", 640, 480)
            dataset_with_categories.add_image(img)

        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)

        splits1 = dataset_with_categories.split(split_ratio, seed=42)
        splits2 = dataset_with_categories.split(split_ratio, seed=42)

        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]
        assert splits1["test"] == splits2["test"]

    def test_split_invalid_ratio(self) -> None:
        """Test that invalid split ratio raises error."""
        bad_split = SplitRatio(train=0.5, val=0.3, test=0.1)

        with pytest.raises(ValidationError):
            bad_split.validate()


class TestDatasetMerge:
    """Test dataset merging."""

    def test_merge_simple(self, dataset_with_categories: Dataset) -> None:
        """Test merging two simple datasets."""
        # Create dataset 1
        ds1 = dataset_with_categories
        img1 = ImageInfo("img1", "test1.jpg", 640, 480)
        ds1.add_image(img1)
        ann1 = Annotation(BBox(0, 0, 50, 50), 1, "person", "img1")
        ds1.add_annotation(ann1)

        # Create dataset 2
        ds2 = Dataset(name="ds2")
        ds2.add_category(1, "person")
        ds2.add_category(4, "dog")
        img2 = ImageInfo("img2", "test2.jpg", 640, 480)
        ds2.add_image(img2)
        ann2 = Annotation(BBox(0, 0, 50, 50), 4, "dog", "img2")
        ds2.add_annotation(ann2)

        # Merge
        merged = ds1.merge(ds2)

        assert merged.num_images() == 2
        assert merged.num_annotations() == 2
        assert merged.num_categories() == 4  # person, car, bicycle, dog

    def test_merge_with_source_preservation(self, dataset_with_categories: Dataset) -> None:
        """Test merging with source tracking preserved."""
        ds1 = dataset_with_categories
        img1 = ImageInfo("img1", "test1.jpg", 640, 480)
        ds1.add_image(img1, source_name="camera1")

        ds2 = Dataset(name="ds2")
        ds2.add_category(1, "person")
        img2 = ImageInfo("img2", "test2.jpg", 640, 480)
        ds2.add_image(img2, source_name="camera2")

        merged = ds1.merge(ds2, preserve_sources=True, regen_ids=False)

        assert merged.get_image_source("img1") == "camera1"
        assert merged.get_image_source("img2") == "camera2"

    def test_merge_category_conflict_skip(self) -> None:
        """Test merging with category conflict resolution - skip."""
        ds1 = Dataset("ds1")
        ds1.add_category(1, "person")

        ds2 = Dataset("ds2")
        ds2.add_category(5, "person")  # Same name, different ID

        merged = ds1.merge(ds2, resolve_conflicts="skip")

        # Should use ID 1 for "person" from ds1
        assert merged.get_category_id("person") == 1
        assert merged.num_categories() == 1

    def test_merge_category_conflict_rename(self) -> None:
        """Test merging with category conflict resolution - rename."""
        ds1 = Dataset("ds1")
        ds1.add_category(1, "person")

        ds2 = Dataset("ds2")
        ds2.add_category(5, "person")

        merged = ds1.merge(ds2, resolve_conflicts="rename")

        # Should have both categories
        assert merged.num_categories() == 2
        assert "person_other" in merged.category_name_to_id

    def test_merge_category_conflict_error(self) -> None:
        """Test merging with category conflict resolution - error."""
        ds1 = Dataset("ds1")
        ds1.add_category(1, "person")

        ds2 = Dataset("ds2")
        ds2.add_category(5, "person")

        with pytest.raises(CategoryConflictError):
            ds1.merge(ds2, resolve_conflicts="error")

    def test_merge_operator(self, dataset_with_categories: Dataset) -> None:
        """Test merging using + operator."""
        ds1 = dataset_with_categories
        img1 = ImageInfo("img1", "test1.jpg", 640, 480)
        ds1.add_image(img1)

        ds2 = Dataset("ds2")
        ds2.add_category(1, "person")
        img2 = ImageInfo("img2", "test2.jpg", 640, 480)
        ds2.add_image(img2)

        merged = ds1 + ds2

        assert merged.num_images() == 2


class TestDatasetMagicMethods:
    """Test dataset magic methods."""

    def test_len(self, simple_dataset: Dataset) -> None:
        """Test len() returns number of images."""
        assert len(simple_dataset) == 1

    def test_add_operator(self, dataset_with_categories: Dataset) -> None:
        """Test + operator for merging."""
        ds1 = dataset_with_categories
        ds2 = Dataset("ds2")

        merged = ds1 + ds2
        assert isinstance(merged, Dataset)

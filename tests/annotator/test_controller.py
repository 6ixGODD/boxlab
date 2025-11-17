"""Tests for AnnotationController business logic."""

from __future__ import annotations

from pathlib import Path

from boxlab.annotator.controller import AnnotationController
from boxlab.dataset.types import Annotation
from boxlab.dataset.types import BBox
from boxlab.dataset.types import SplitRatio


class TestControllerInitialization:
    """Test controller initialization."""

    def test_create_controller(self) -> None:
        """Test creating a new controller."""
        controller = AnnotationController()

        assert not controller.has_dataset()
        assert controller.workspace_path is None
        assert not controller.workspace_modified
        assert not controller.audit_mode

    def test_controller_default_state(self) -> None:
        """Test controller default state."""
        controller = AnnotationController()

        assert controller.current_split is None
        assert controller.current_index == 0
        assert len(controller.available_tags) == 0
        assert len(controller.datasets) == 0


class TestControllerDatasetLoading:
    """Test dataset loading through controller."""

    def test_load_coco_dataset(
        self, controller: AnnotationController, tmp_coco_dataset: Path
    ) -> None:
        """Test loading COCO dataset."""
        controller.load_dataset(str(tmp_coco_dataset), "coco", ["train"])

        assert controller.has_dataset()
        assert controller.total_images() == 4
        assert len(controller.get_splits()) == 1

    def test_load_yolo_dataset(
        self, controller: AnnotationController, tmp_yolo_dataset: Path
    ) -> None:
        """Test loading YOLO dataset."""
        controller.load_dataset(str(tmp_yolo_dataset), "yolo", ["train"])

        assert controller.has_dataset()
        assert controller.total_images() == 2

    def test_set_audit_status_sets_modified(
        self, controller: AnnotationController, tmp_coco_dataset: Path
    ) -> None:
        """Test that loading dataset marks workspace as modified."""
        controller.load_dataset(str(tmp_coco_dataset), "coco", ["train"])
        controller.set_audit_status(controller.get_current_image_id(), "approved")

        # Should be modified after loading
        assert controller.workspace_modified


class TestControllerNavigation:
    """Test navigation through images."""

    def test_next_image(self, controller_with_images: AnnotationController) -> None:
        """Test navigating to next image."""
        controller = controller_with_images
        assert controller.current_index == 0

        first_id = controller.get_current_image_id()
        next_id = controller.next_image()

        assert next_id is not None
        assert next_id != first_id
        assert controller.current_index == 1

    def test_prev_image(self, controller_with_images: AnnotationController) -> None:
        """Test navigating to previous image."""
        controller = controller_with_images
        assert controller.current_index == 0

        # Go forward first
        controller.next_image()
        controller.next_image()

        # Then go back
        prev_id = controller.prev_image()

        assert prev_id is not None
        assert controller.current_index == 1

    def test_next_image_at_end(self, controller_with_images: AnnotationController) -> None:
        """Test next_image at end of list stays at end."""
        controller = controller_with_images

        # Navigate to end
        while controller.next_image():
            pass

        last_index = controller.current_index

        # Try to go further
        result = controller.next_image()

        assert result is None
        assert controller.current_index == last_index

    def test_prev_image_at_start(self, controller_with_images: AnnotationController) -> None:
        """Test prev_image at start stays at start."""
        controller = controller_with_images

        # Already at start
        result = controller.prev_image()

        assert result is None
        assert controller.current_index == 0


class TestControllerSplits:
    """Test split management."""

    def test_get_splits(self, controller_with_splits: AnnotationController) -> None:
        """Test getting available splits."""
        splits = controller_with_splits.get_splits()

        assert "train" in splits
        assert "val" in splits

    def test_set_current_split(self, controller_with_splits: AnnotationController) -> None:
        """Test setting current split."""
        controller_with_splits.set_current_split("val")

        assert controller_with_splits.current_split == "val"
        assert controller_with_splits.current_index == 0

    def test_get_images_in_split(self, controller_with_splits: AnnotationController) -> None:
        """Test getting images for a specific split."""
        train_images = controller_with_splits.get_images_in_split("train")
        val_images = controller_with_splits.get_images_in_split("val")

        assert len(train_images) > 0
        assert len(val_images) > 0

        # Should be different sets
        train_ids = {img[0] for img in train_images}
        val_ids = {img[0] for img in val_images}
        assert not train_ids.intersection(val_ids)


class TestControllerAnnotations:
    """Test annotation management."""

    def test_get_annotations(self, controller_with_images: AnnotationController) -> None:
        """Test getting annotations for an image."""
        controller = controller_with_images
        img_id = controller.get_current_image_id()

        anns = controller.get_annotations(img_id)

        assert isinstance(anns, list)

    def test_update_annotations(self, controller_with_images: AnnotationController) -> None:
        """Test updating annotations."""
        controller = controller_with_images
        img_id = controller.get_current_image_id()

        # Create new annotation
        new_ann = Annotation(
            BBox(50, 50, 100, 100),
            1,
            "person",
            img_id,
        )

        controller.update_annotations(img_id, [new_ann])

        # Should mark as modified
        assert controller.workspace_modified

        # Should have modification recorded
        assert img_id in controller.modified_annotations

    def test_get_categories(self, controller_with_images: AnnotationController) -> None:
        """Test getting category list."""
        categories = controller_with_images.get_categories()

        assert isinstance(categories, dict)
        assert len(categories) > 0


class TestControllerAudit:
    """Test audit functionality."""

    def test_enable_audit_mode(self, controller_with_images: AnnotationController) -> None:
        """Test enabling audit mode."""
        controller = controller_with_images

        controller.enable_audit_mode(True)

        assert controller.audit_mode

    def test_set_audit_status(self, controller_with_images: AnnotationController) -> None:
        """Test setting audit status."""
        controller = controller_with_images
        controller.enable_audit_mode(True)

        img_id = controller.get_current_image_id()
        controller.set_audit_status(img_id, "approved")

        assert controller.get_audit_status(img_id) == "approved"

    def test_set_audit_comment(self, controller_with_images: AnnotationController) -> None:
        """Test setting audit comment."""
        controller = controller_with_images
        controller.enable_audit_mode(True)

        img_id = controller.get_current_image_id()
        controller.set_audit_comment(img_id, "Test comment")

        assert controller.get_audit_comment(img_id) == "Test comment"

    def test_get_audit_statistics(self, controller_with_images: AnnotationController) -> None:
        """Test getting audit statistics."""
        controller = controller_with_images
        controller.enable_audit_mode(True)

        # Set some statuses
        images = list(controller.datasets[controller.current_split].images.keys())
        controller.set_audit_status(images[0], "approved")
        if len(images) > 1:
            controller.set_audit_status(images[1], "rejected")

        stats = controller.get_audit_statistics()

        assert "approved" in stats
        assert "rejected" in stats
        assert "pending" in stats


class TestControllerTags:
    """Test tag management."""

    def test_add_tag(self, controller_with_images: AnnotationController) -> None:
        """Test adding a tag."""
        controller = controller_with_images

        controller.add_tag("test_tag")

        assert "test_tag" in controller.available_tags

    def test_set_image_tags(self, controller_with_images: AnnotationController) -> None:
        """Test setting tags for an image."""
        controller = controller_with_images
        controller.add_tag("tag1")
        controller.add_tag("tag2")

        img_id = controller.get_current_image_id()
        controller.set_image_tags(img_id, ["tag1", "tag2"])

        tags = controller.get_image_tags(img_id)
        assert set(tags) == {"tag1", "tag2"}

    def test_get_image_tags_empty(self, controller_with_images: AnnotationController) -> None:
        """Test getting tags for image with no tags."""
        controller = controller_with_images
        img_id = controller.get_current_image_id()

        tags = controller.get_image_tags(img_id)

        assert tags == []


class TestControllerExport:
    """Test dataset export."""

    def test_export_dataset(
        self, controller_with_images: AnnotationController, tmp_path: Path
    ) -> None:
        """Test exporting dataset."""
        controller = controller_with_images

        output_dir = tmp_path / "export"
        controller.export_dataset(str(output_dir), "coco", "preserve")

        # Check output exists
        assert output_dir.exists()
        assert (output_dir / "annotations_train.json").exists()

    def test_export_with_split_ratio(
        self, controller_with_images: AnnotationController, tmp_path: Path
    ) -> None:
        """Test exporting with split ratio."""
        controller = controller_with_images

        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)
        output_dir = tmp_path / "export"

        controller.export_dataset(str(output_dir), "coco", "preserve", split_ratio)

        # Should create split files
        assert (output_dir / "annotations_train.json").exists()


class TestControllerUnsavedChanges:
    """Test unsaved changes tracking."""

    def test_has_unsaved_changes_after_annotation(
        self, controller_with_images: AnnotationController
    ) -> None:
        """Test unsaved changes after modifying annotations."""
        controller = controller_with_images

        # Save first
        controller.set_workspace_modified(False)
        assert not controller.has_unsaved_changes()

        # Modify
        img_id = controller.get_current_image_id()
        controller.update_annotations(img_id, [])

        assert controller.has_unsaved_changes()

    def test_has_unsaved_changes_after_audit(
        self, controller_with_images: AnnotationController
    ) -> None:
        """Test unsaved changes after audit operations."""
        controller = controller_with_images
        controller.enable_audit_mode(True)

        controller.set_workspace_modified(False)

        img_id = controller.get_current_image_id()
        controller.set_audit_status(img_id, "approved")

        assert controller.has_unsaved_changes()

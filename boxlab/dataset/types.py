from __future__ import annotations

import pathlib
import typing as t

from boxlab.exceptions import ValidationError


class BBox(t.NamedTuple):
    """Bounding box representation in XYXY format.

    This class represents a rectangular bounding box using the XYXY coordinate format
    (top-left and bottom-right corners). It provides conversions to other common formats
    and utility methods for area calculation.

    The internal representation uses XYXY format (x_min, y_min, x_max, y_max) which is
    the most common format for object detection tasks.

    Attributes:
        x_min: Minimum X coordinate of the bounding box.
        y_min: Minimum Y coordinate of the bounding box.
        x_max: Maximum X coordinate of the bounding box.
        y_max: Maximum Y coordinate of the bounding box.

    Example:
        ```python
        from boxlab.dataset.types import BBox

        # Create from XYXY coordinates
        bbox = BBox(x_min=10, y_min=20, x_max=100, y_max=150)

        # Access coordinates
        print(f"Top-left: ({bbox.x_min}, {bbox.y_min})")
        print(f"Bottom-right: ({bbox.x_max}, {bbox.y_max})")

        # Get area
        print(f"Area: {bbox.area}")  # 11700
        ```

    Example:
        ```python
        # Convert to different formats
        bbox = BBox(10, 20, 100, 150)

        xyxy = bbox.xyxy  # (10, 20, 100, 150)
        xywh = bbox.xywh  # (10, 20, 90, 130)
        cxcywh = bbox.cxcywh  # (55.0, 85.0, 90, 130)
        ```

    Example:
        ```python
        # Create from other formats
        bbox1 = BBox.from_xywh(x=10, y=20, w=90, h=130)
        bbox2 = BBox.from_cxcywh(cx=55, cy=85, w=90, h=130)

        print(bbox1 == bbox2)  # True
        ```
    """

    x_min: float
    """Minimum X coordinate of the bounding box."""

    y_min: float
    """Minimum Y coordinate of the bounding box."""

    x_max: float
    """Maximum X coordinate of the bounding box."""

    y_max: float
    """Maximum Y coordinate of the bounding box."""

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return bounding box in XYXY format.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max).
        """
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Convert to XYWH format (COCO format).

        Returns:
            Tuple of (x, y, width, height) where (x, y) is the top-left corner.

        Example:
            ```python
            bbox = BBox(10, 20, 100, 150)
            x, y, w, h = bbox.xywh
            # x=10, y=20, w=90, h=130
            ```
        """
        return self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min

    @property
    def cxcywh(self) -> tuple[float, float, float, float]:
        """Convert to center format (YOLO format).

        Returns:
            Tuple of (center_x, center_y, width, height).

        Example:
            ```python
            bbox = BBox(10, 20, 100, 150)
            cx, cy, w, h = bbox.cxcywh
            # cx=55.0, cy=85.0, w=90, h=130
            ```
        """
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return self.x_min + width / 2, self.y_min + height / 2, width, height

    @property
    def area(self) -> float:
        """Calculate bounding box area.

        Returns:
            Area in square pixels (width * height).

        Example:
            ```python
            bbox = BBox(0, 0, 10, 20)
            print(bbox.area)  # 200.0
            ```
        """
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create BBox from XYWH format (COCO format).

        Args:
            x: Left X coordinate.
            y: Top Y coordinate.
            w: Width of the box.
            h: Height of the box.

        Returns:
            BBox instance.

        Example:
            ```python
            # COCO format: [x, y, width, height]
            bbox = BBox.from_xywh(x=10, y=20, w=90, h=130)
            print(bbox.xyxy)  # (10, 20, 100, 150)
            ```
        """
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create BBox from center format (YOLO format).

        Args:
            cx: Center X coordinate.
            cy: Center Y coordinate.
            w: Width of the box.
            h: Height of the box.

        Returns:
            BBox instance.

        Example:
            ```python
            # YOLO format: [center_x, center_y, width, height]
            bbox = BBox.from_cxcywh(cx=55, cy=85, w=90, h=130)
            print(bbox.xyxy)  # (10.0, 20.0, 100.0, 150.0)
            ```
        """
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class Annotation(t.NamedTuple):
    """Object annotation with bounding box and category information.

    Represents a single object annotation in an image, including its spatial location
    (bounding box) and semantic information (category). Follows COCO annotation conventions.

    Attributes:
        bbox: Bounding box of the annotated object.
        category_id: Integer category identifier.
        category_name: Human-readable category name.
        image_id: ID of the image this annotation belongs to.
        annotation_id: Optional unique identifier for this annotation.
        area: Optional pre-computed area. If None, calculated from bbox.
        iscrowd: Crowd annotation flag (0=single object, 1=crowd of objects).

    Example:
        ```python
        from boxlab.dataset.types import Annotation, BBox

        # Create an annotation
        annotation = Annotation(
            bbox=BBox(x_min=10, y_min=20, x_max=100, y_max=150),
            category_id=1,
            category_name="person",
            image_id="img_001",
            annotation_id="ann_001",
            area=11700.0,
            iscrowd=0,
        )

        print(f"Object: {annotation.category_name}")
        print(f"Location: {annotation.bbox.xyxy}")
        print(f"Area: {annotation.get_area()}")
        ```

    Example:
        ```python
        # Create annotation without pre-computed area
        annotation = Annotation(
            bbox=BBox(0, 0, 100, 100),
            category_id=2,
            category_name="car",
            image_id="img_002",
        )

        # Area is computed from bbox automatically
        print(annotation.get_area())  # 10000.0
        ```
    """

    bbox: BBox
    """Bounding box of the annotation."""

    category_id: int
    """Category ID of the annotation."""

    category_name: str
    """Category name of the annotation."""

    image_id: str
    """ID of the image the annotation belongs to."""

    annotation_id: str | None = None
    """Unique ID of the annotation."""

    area: float | None = None
    """Area of the annotation, if available."""

    iscrowd: int = 0
    """Crowd annotation flag (0 or 1)."""

    def get_area(self) -> float:
        """Get annotation area.

        Returns pre-computed area if available, otherwise calculates from bbox.

        Returns:
            Area in square pixels.

        Example:
            ```python
            # With pre-computed area
            ann1 = Annotation(
                bbox=BBox(0, 0, 10, 10),
                category_id=1,
                category_name="obj",
                image_id="1",
                area=100.0,
            )
            print(ann1.get_area())  # 100.0 (uses pre-computed)

            # Without pre-computed area
            ann2 = Annotation(
                bbox=BBox(0, 0, 10, 10),
                category_id=1,
                category_name="obj",
                image_id="1",
            )
            print(ann2.get_area())  # 100.0 (computed from bbox)
            ```
        """
        return self.area if self.area is not None else self.bbox.area


class ImageInfo(t.NamedTuple):
    """Image metadata container.

    Stores essential information about an image in the dataset, including dimensions
    and file location.

    Attributes:
        image_id: Unique identifier for the image.
        file_name: Filename of the image (e.g., "image001.jpg").
        width: Image width in pixels.
        height: Image height in pixels.
        path: Optional filesystem path to the image file.

    Example:
        ```python
        from pathlib import Path
        from boxlab.dataset.types import ImageInfo

        # Create image metadata
        img_info = ImageInfo(
            image_id="img_001",
            file_name="photo.jpg",
            width=1920,
            height=1080,
            path=Path("/data/images/photo.jpg"),
        )

        print(f"Image: {img_info.file_name}")
        print(f"Resolution: {img_info.width}x{img_info.height}")
        print(
            f"Exists: {img_info.path.exists() if img_info.path else 'N/A'}"
        )
        ```

    Example:
        ```python
        # Create without path (for export scenarios)
        img_info = ImageInfo(
            image_id="img_002",
            file_name="image002.jpg",
            width=640,
            height=480,
        )
        ```
    """

    image_id: str
    """Unique identifier for the image."""

    file_name: str
    """Filename of the image."""

    width: int
    """Width of the image."""

    height: int
    """Height of the image."""

    path: pathlib.Path | None = None
    """Optional filesystem path to the image."""


class DatasetStatistics(t.TypedDict):
    """Dataset statistics container.

    TypedDict containing comprehensive statistical information about a dataset,
    including counts, distributions, and aggregate metrics.

    This structure is returned by Dataset.get_statistics() and related methods.

    Attributes:
        num_images: Total number of images in the dataset.
        num_annotations: Total number of annotations across all images.
        num_categories: Number of unique object categories.
        category_distribution: Mapping of category names to their annotation counts.
        avg_annotations_per_image: Mean number of annotations per image.
        std_annotations_per_image: Standard deviation of annotations per image.
        min_annotations_per_image: Minimum annotations found in any image.
        max_annotations_per_image: Maximum annotations found in any image.
        avg_bbox_area: Mean bounding box area in square pixels.
        median_bbox_area: Median bounding box area in square pixels.

    Example:
        ```python
        from boxlab.dataset import Dataset

        dataset = Dataset(name="my_dataset")
        # ... populate dataset ...

        # Get statistics
        stats = dataset.get_statistics()

        print(f"Images: {stats['num_images']}")
        print(f"Annotations: {stats['num_annotations']}")
        print(f"Categories: {stats['num_categories']}")
        print(
            f"Avg objects per image: {stats['avg_annotations_per_image']:.2f}"
        )

        # Category distribution
        for category, count in stats[
            "category_distribution"
        ].items():
            print(f"  {category}: {count}")
        ```

    Example:
        ```python
        # Analyze bounding box sizes
        stats = dataset.get_statistics()

        print(f"Average bbox area: {stats['avg_bbox_area']:.2f}")
        print(f"Median bbox area: {stats['median_bbox_area']:.2f}")

        # Check for annotation imbalance
        min_anns = stats["min_annotations_per_image"]
        max_anns = stats["max_annotations_per_image"]
        if max_anns > min_anns * 10:
            print(
                "Warning: Large annotation count variance detected"
            )
        ```
    """

    num_images: int
    """Number of images in the dataset."""

    num_annotations: int
    """Number of annotations in the dataset."""

    num_categories: int
    """Number of unique categories in the dataset."""

    category_distribution: dict[str, int]
    """Distribution of annotations per category."""

    avg_annotations_per_image: float
    """Average number of annotations per image."""

    std_annotations_per_image: float
    """Standard deviation of annotations per image."""

    min_annotations_per_image: int
    """Minimum number of annotations in any image."""

    max_annotations_per_image: int
    """Maximum number of annotations in any image."""

    avg_bbox_area: float
    """Average bounding box area."""

    median_bbox_area: float
    """Median bounding box area."""


class SplitRatio(t.NamedTuple):
    """Dataset split ratios for train/val/test division.

    Defines the proportions for splitting a dataset into training, validation,
    and test sets. All ratios must sum to 1.0.

    Attributes:
        train: Training set ratio (default: 0.8).
        val: Validation set ratio (default: 0.1).
        test: Test set ratio (default: 0.1).

    Raises:
        ValidationError: If ratios don't sum to 1.0 (within tolerance of 1e-6).

    Example:
        ```python
        from boxlab.dataset.types import SplitRatio

        # 70% train, 20% val, 10% test
        split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)
        split_ratio.validate()  # OK

        # Use with dataset
        dataset.split(split_ratio, seed=42)
        ```

    Example:
        ```python
        # 80% train, 20% val, no test set
        split_ratio = SplitRatio(train=0.8, val=0.2, test=0.0)
        split_ratio.validate()  # OK
        ```

    Example:
        ```python
        # Invalid split (doesn't sum to 1.0)
        try:
            split_ratio = SplitRatio(train=0.7, val=0.2, test=0.2)
            split_ratio.validate()
        except ValidationError as e:
            print(f"Error: {e}")
            # Output: Split ratios must sum to 1.0, got 1.1
        ```

    Example:
        ```python
        # Default split (80/10/10)
        split_ratio = SplitRatio()
        print(f"Train: {split_ratio.train}")  # 0.8
        print(f"Val: {split_ratio.val}")  # 0.1
        print(f"Test: {split_ratio.test}")  # 0.1
        ```
    """

    train: float = 0.8
    """Training set ratio."""

    val: float = 0.1
    """Validation set ratio."""

    test: float = 0.1
    """Test set ratio."""

    def validate(self) -> None:
        """Validate that split ratios sum to 1.0.

        Checks that train + val + test equals 1.0 within a tolerance of 1e-6.

        Raises:
            ValidationError: If ratios don't sum to 1.0.

        Example:
            ```python
            split_ratio = SplitRatio(train=0.7, val=0.2, test=0.1)
            split_ratio.validate()  # OK

            bad_split = SplitRatio(train=0.5, val=0.3, test=0.1)
            bad_split.validate()  # Raises ValidationError
            ```
        """
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-6:
            raise ValidationError(f"Split ratios must sum to 1.0, got {total}")

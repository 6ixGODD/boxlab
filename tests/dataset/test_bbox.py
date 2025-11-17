from __future__ import annotations

from boxlab.dataset.types import BBox


class TestBBoxCreation:
    """Test BBox creation."""

    def test_create_bbox_xyxy(self) -> None:
        """Test creating bbox from XYXY coordinates."""
        bbox = BBox(x_min=10, y_min=20, x_max=100, y_max=150)

        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 150

    def test_create_bbox_from_xywh(self) -> None:
        """Test creating bbox from XYWH format."""
        bbox = BBox.from_xywh(x=10, y=20, w=90, h=130)

        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 150

    def test_create_bbox_from_cxcywh(self):
        """Test creating bbox from center format."""
        bbox = BBox.from_cxcywh(cx=55, cy=85, w=90, h=130)

        assert bbox.x_min == 10.0
        assert bbox.y_min == 20.0
        assert bbox.x_max == 100.0
        assert bbox.y_max == 150.0


class TestBBoxConversions:
    """Test BBox format conversions."""

    def test_xyxy_property(self):
        """Test XYXY format property."""
        bbox = BBox(10, 20, 100, 150)
        assert bbox.xyxy == (10, 20, 100, 150)

    def test_xywh_conversion(self):
        """Test conversion to XYWH format."""
        bbox = BBox(10, 20, 100, 150)
        x, y, w, h = bbox.xywh

        assert x == 10
        assert y == 20
        assert w == 90  # 100 - 10
        assert h == 130  # 150 - 20

    def test_cxcywh_conversion(self) -> None:
        """Test conversion to center format."""
        bbox = BBox(10, 20, 100, 150)
        cx, cy, w, h = bbox.cxcywh

        assert cx == 55.0  # 10 + 90/2
        assert cy == 85.0  # 20 + 130/2
        assert w == 90
        assert h == 130

    def test_area_calculation(self) -> None:
        """Test bounding box area calculation."""
        bbox = BBox(0, 0, 10, 20)
        assert bbox.area == 200.0

        bbox2 = BBox(10, 20, 100, 150)
        assert bbox2.area == 11700.0  # 90 * 130


class TestBBoxRoundtrips:
    """Test coordinate conversion roundtrips."""

    def test_xywh_roundtrip(self) -> None:
        """Test XYWH conversion roundtrip."""
        original = BBox(10, 20, 100, 150)

        # Convert to XYWH and back
        x, y, w, h = original.xywh
        restored = BBox.from_xywh(x, y, w, h)

        assert restored == original

    def test_cxcywh_roundtrip(self) -> None:
        """Test center format conversion roundtrip."""
        original = BBox(10, 20, 100, 150)

        # Convert to center format and back
        cx, cy, w, h = original.cxcywh
        restored = BBox.from_cxcywh(cx, cy, w, h)

        assert restored == original

    def test_float_precision_roundtrip(self) -> None:
        """Test roundtrip with floating point coordinates."""
        original = BBox(10.5, 20.7, 100.3, 150.9)

        cx, cy, w, h = original.cxcywh
        restored = BBox.from_cxcywh(cx, cy, w, h)

        # Check with small tolerance for float precision
        assert abs(restored.x_min - original.x_min) < 1e-10
        assert abs(restored.y_min - original.y_min) < 1e-10
        assert abs(restored.x_max - original.x_max) < 1e-10
        assert abs(restored.y_max - original.y_max) < 1e-10


class TestBBoxEdgeCases:
    """Test BBox edge cases."""

    def test_zero_area_bbox(self) -> None:
        """Test bbox with zero area."""
        bbox = BBox(10, 20, 10, 20)
        assert bbox.area == 0.0

    def test_negative_coordinates(self) -> None:
        """Test bbox with negative coordinates."""
        bbox = BBox(-10, -20, 50, 100)
        assert bbox.area == 7200.0  # 60 * 120

    def test_large_coordinates(self) -> None:
        """Test bbox with large coordinates."""
        bbox = BBox(0, 0, 10000, 10000)
        assert bbox.area == 100000000.0

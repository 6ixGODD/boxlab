import pytest


@pytest.fixture
def sample_bbox():
    """Sample bounding box for testing."""
    from boxlab.dataset.types import BBox

    return BBox(x_min=10, y_min=20, x_max=100, y_max=150)


@pytest.fixture
def sample_annotation():
    """Sample annotation for testing."""
    from boxlab.dataset.types import Annotation
    from boxlab.dataset.types import BBox

    bbox = BBox(x_min=10, y_min=20, x_max=100, y_max=150)
    return Annotation(bbox=bbox, category_id=1, category_name="test", image_id="img_001")

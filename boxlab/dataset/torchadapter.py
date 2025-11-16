from __future__ import annotations

import logging
import typing as t

from boxlab.dataset import Dataset
from boxlab.dataset.types import Annotation
from boxlab.exceptions import DatasetError
from boxlab.exceptions import DatasetNotFoundError
from boxlab.exceptions import RequiredModuleNotFoundError
from boxlab.exceptions import ValidationError

if t.TYPE_CHECKING:
    import PIL.Image as Image
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from torchvision import transforms as T  # noqa: N812
else:
    try:
        import PIL.Image as Image
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from torchvision import transforms as T  # noqa: N812
    except ImportError as e:
        missing_module = e.name or ("torch", "torchvision", "Pillow")

        def _raise_import_error(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
            raise RequiredModuleNotFoundError(
                *missing_module,
                message=f"Required module '{missing_module}' not found. "
                "Please install with: pip install torch torchvision pillow",
            )

        # Create placeholder classes that raise errors when instantiated
        class TorchDataset:  # type: ignore[no-redef]
            def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:  # noqa
                _raise_import_error()

        class torch:  # type: ignore[no-redef]  # noqa
            Tensor = object

            @staticmethod
            def tensor(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def as_tensor(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def zeros(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

        class T:  # type: ignore[no-redef]
            @staticmethod
            def Resize(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def RandomHorizontalFlip(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def ColorJitter(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def RandomAffine(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def Normalize(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def ToTensor(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

            @staticmethod
            def Compose(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()

        class Image:  # type: ignore[no-redef]
            @staticmethod
            def open(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:  # noqa
                _raise_import_error()


logger = logging.getLogger(__name__)


class Target(t.TypedDict):
    boxes: torch.Tensor  # shape (N, 4)
    """Bounding boxes in the specified format."""

    labels: torch.Tensor  # shape (N,)
    """Class labels for each bounding box."""

    image_id: torch.Tensor  # shape (1,)
    """Image identifier."""

    area: torch.Tensor  # shape (N,)
    """Area of each bounding box."""

    iscrowd: torch.Tensor  # shape (N,)
    """Crowd flag for each bounding box."""


ImageTargetPair: t.TypeAlias = tuple[t.Any, Target]
"""Type alias for a tuple of (image, target)."""


class TorchDatasetAdapter(TorchDataset[ImageTargetPair]):  # type: ignore[misc]
    """Adapter to convert custom datasets to torchvision-compatible format.

    This adapter wraps BaseDataset instances and provides a PyTorch Dataset interface
    suitable for use with DataLoader and torchvision transforms.

    Note:
        This adapter requires torch, torchvision, and pillow to be installed.
        Install with: pip install torch torchvision pillow

    Args:
        dataset: Source dataset (COCODataset or YOLOv5Dataset)
        transform: Optional torchvision transforms for images
        target_transform: Optional transforms for targets/annotations
        return_format: Format for bounding boxes ('xyxy', 'xywh', 'cxcywh')
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: t.Callable[..., torch.Tensor] | None = None,  # type: ignore[valid-type]
        target_transform: t.Callable[[Target], Target] | None = None,
        return_format: t.Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.return_format = return_format

        # Create index to image_id mapping
        self.image_ids = list(dataset.images.keys())

        logger.info(
            f"Created TorchvisionDatasetAdapter with {len(self.image_ids)} images, "
            f"bbox format: {return_format}"
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> ImageTargetPair:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, target) where target is a dict containing:
                - boxes: Tensor of shape (N, 4) with bounding boxes
                - labels: Tensor of shape (N,) with class labels
                - image_id: Image identifier
                - area: Tensor of shape (N,) with box areas
                - iscrowd: Tensor of shape (N,) with crowd flags
        """
        image_id = self.image_ids[idx]
        img_info = self.dataset.get_image(image_id)

        if img_info is None:
            raise DatasetError(f"Image {image_id} not found in dataset")

        if img_info.path is None or not img_info.path.exists():
            raise DatasetNotFoundError(str(img_info.path), f"Image file not found: {img_info.path}")

        # Load image
        img: torch.Tensor | Image.Image
        img = Image.open(img_info.path).convert("RGB")

        # Get annotations
        annotations = self.dataset.get_annotations(image_id)

        # Prepare target dict
        target = self._prepare_target(annotations, img_info.image_id)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _prepare_target(self, annotations: list[Annotation], image_id: str) -> Target:
        """Prepare target dictionary from annotations.

        Args:
            annotations: List of annotations
            image_id: Image identifier

        Returns:
            Target dictionary
        """
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # Get bounding box in requested format
            if self.return_format == "xyxy":
                box = ann.bbox.xyxy
            elif self.return_format == "xywh":
                box = ann.bbox.xywh
            elif self.return_format == "cxcywh":
                box = ann.bbox.cxcywh
            else:
                raise ValidationError(f"Unknown return format: {self.return_format}")

            boxes.append(box)
            labels.append(ann.category_id)
            areas.append(ann.get_area())
            iscrowd.append(ann.iscrowd)

        target: Target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),  # type: ignore[dict-item]
            "labels": torch.as_tensor(labels, dtype=torch.int64)  # type: ignore[dict-item]
            if labels
            else torch.zeros((0,), dtype=torch.int64),  # type: ignore[dict-item]
            "image_id": torch.tensor([int(image_id)]),  # type: ignore[dict-item]
            "area": torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),  # type: ignore[dict-item]
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)  # type: ignore[dict-item]
            if iscrowd
            else torch.zeros((0,), dtype=torch.int64),  # type: ignore[dict-item]
        }

        return target

    def collate(self, batch: list[ImageTargetPair]) -> tuple[list[torch.Tensor], list[Target]]:  # type: ignore[valid-type]
        """Custom collate function for DataLoader.

        This is useful when images have different numbers of objects.

        Args:
            batch: List of (image, target) tuples

        Returns:
            Tuple of (images, targets) lists
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets


def build_torchdataset(
    dataset: Dataset,
    image_size: int | tuple[int, int] | None = None,
    augment: bool = False,
    normalize: bool = False,
    *transforms: t.Callable[..., t.Any],
    return_format: t.Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
) -> TorchDatasetAdapter:
    """Create a torch-compatible dataset with standard transforms.

    Note:
        This function requires torch, torchvision, and pillow to be installed.
        Install with: pip install torch torchvision pillow

    Args:
        dataset: Source dataset (COCODataset or YOLOv5Dataset)
        image_size: Target image size (single int or (height, width))
        augment: Whether to apply data augmentation
        normalize: Whether to normalize images using ImageNet statistics
        *transforms: Additional user-defined transforms
        return_format: Format for bounding boxes

    Returns:
        TorchDatasetAdapter instance
    """
    logger.info(
        f"Creating torchvision dataset: size={image_size}, augment={augment}, normalize={normalize}"
    )

    transform_list: list[t.Callable[..., t.Any]] = []

    # Resize
    if image_size is not None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_list.append(T.Resize(image_size))

    # Augmentation
    if augment:
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    # Convert to tensor (must be before normalization)
    transform_list.append(T.ToTensor())

    # Normalization (must be after ToTensor)
    if normalize:
        transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Additional user-defined transforms
    transform_list.extend(transforms)

    # Compose all transforms
    transform = T.Compose(transform_list)

    return TorchDatasetAdapter(
        dataset=dataset,
        transform=transform,
        return_format=return_format,
    )

from __future__ import annotations

import logging
import typing as t

import PIL.Image as Image

from boxlab.dataset import Dataset
from boxlab.dataset.types import Annotation
from boxlab.exceptions import DatasetError
from boxlab.exceptions import DatasetNotFoundError
from boxlab.exceptions import RequiredModuleNotFoundError
from boxlab.exceptions import ValidationError

if t.TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from torchvision import transforms as T  # noqa: N812
else:
    try:
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from torchvision import transforms as T  # noqa: N812
    except ImportError as e:
        missing_module = e.name or ("torch", "torchvision")

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
    """Target dictionary containing annotation information for an image.

    This TypedDict defines the structure of target data returned by the dataset adapter,
    following torchvision's object detection format conventions.

    Attributes:
        boxes: Bounding boxes tensor of shape (N, 4) where N is the number of objects.
            Format depends on the adapter's return_format setting.
        labels: Class label tensor of shape (N,) containing integer category IDs.
        image_id: Image identifier tensor of shape (1,).
        area: Area values tensor of shape (N,) for each bounding box.
        iscrowd: Crowd flag tensor of shape (N,) indicating if object is a crowd.
    """

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
"""Type alias for a tuple of (image, target).

The image can be either a PIL Image or a torch.Tensor depending on
whether transforms have been applied.
"""


class TorchDatasetAdapter(TorchDataset[ImageTargetPair]):  # type: ignore[misc]
    """Adapter to convert BoxLab datasets to PyTorch-compatible format.

    This adapter wraps Dataset instances and provides a PyTorch Dataset interface
    suitable for use with DataLoader and torchvision transforms. It handles image
    loading, annotation formatting, and coordinate conversion.

    The adapter follows torchvision's object detection conventions, making it compatible
    with models like Faster R-CNN, Mask R-CNN, and other detection architectures.

    Args:
        dataset: Source BoxLab Dataset instance.
        transform: Optional torchvision transforms pipeline for images. Applied to
            PIL Images before returning.
        target_transform: Optional transforms for targets/annotations. Applied to
            the target dictionary.
        return_format: Format for bounding boxes. Options:
            - "xyxy": [x_min, y_min, x_max, y_max]
            - "xywh": [x_min, y_min, width, height]
            - "cxcywh": [center_x, center_y, width, height]

    Attributes:
        dataset: The wrapped Dataset instance.
        transform: Image transformation pipeline.
        target_transform: Target transformation pipeline.
        return_format: Bounding box format string.
        image_ids: List of image IDs for indexing.

    Note:
        This adapter requires torch, torchvision, and pillow to be installed.
        Install with: `pip install torch torchvision pillow`

    Raises:
        RequiredModuleNotFoundError: If torch, torchvision, or PIL are not installed.

    Example:
        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.torchadapter import TorchDatasetAdapter
        from torchvision import transforms as T

        # Create dataset
        dataset = Dataset(name="my_dataset")
        # ... populate dataset ...

        # Create adapter with transforms
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        torch_dataset = TorchDatasetAdapter(
            dataset, transform=transform, return_format="xyxy"
        )

        # Use with DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(
            torch_dataset,
            batch_size=4,
            collate_fn=torch_dataset.collate,
        )
        ```

    Example:
        ```python
        # Iterate over dataset
        for image, target in torch_dataset:
            print(f"Image shape: {image.shape}")
            print(f"Boxes: {target['boxes'].shape}")
            print(f"Labels: {target['labels']}")
        ```
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
            f"Created TorchDatasetAdapter with {len(self.image_ids)} images, "
            f"bbox format: {return_format}"
        )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> ImageTargetPair:
        """Get a sample by index.

        Loads the image and its annotations, applies transforms, and returns them
        in PyTorch-compatible format.

        Args:
            idx: Sample index (0-based integer).

        Returns:
            Tuple of (image, target) where:
                - image: PIL Image or torch.Tensor (if transform applied)
                - target: Dictionary containing:
                    - boxes: Tensor of shape (N, 4) with bounding boxes
                    - labels: Tensor of shape (N,) with class labels (1-indexed)
                    - image_id: Tensor with image identifier
                    - area: Tensor of shape (N,) with box areas
                    - iscrowd: Tensor of shape (N,) with crowd flags

        Raises:
            DatasetError: If image is not found in dataset.
            DatasetNotFoundError: If image file does not exist on disk.

        Example:
            ```python
            # Get first sample
            image, target = torch_dataset[0]

            # Access target components
            boxes = target["boxes"]  # Shape: (N, 4)
            labels = target["labels"]  # Shape: (N,)
            image_id = target["image_id"]
            ```
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

        Converts BoxLab annotations to PyTorch tensor format with the specified
        bounding box format.

        Args:
            annotations: List of Annotation objects for the image.
            image_id: Image identifier string.

        Returns:
            Target dictionary with torch tensors.

        Raises:
            ValidationError: If return_format is unknown.

        Note:
            Empty annotations return zero-sized tensors with correct shapes.
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

        This collate function is useful when images have different numbers of objects,
        which is common in object detection. Instead of stacking tensors (which requires
        same dimensions), it returns lists of tensors and targets.

        Args:
            batch: List of (image, target) tuples from __getitem__.

        Returns:
            Tuple of (images, targets) where:
                - images: List of image tensors
                - targets: List of target dictionaries

        Example:
            ```python
            from torch.utils.data import DataLoader

            loader = DataLoader(
                torch_dataset,
                batch_size=4,
                collate_fn=torch_dataset.collate,
                shuffle=True,
            )

            for images, targets in loader:
                # images: list of 4 tensors
                # targets: list of 4 target dicts
                for img, tgt in zip(images, targets):
                    print(f"Image: {img.shape}")
                    print(f"Objects: {len(tgt['boxes'])}")
            ```
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
    """Create a PyTorch-compatible dataset with standard transforms.

    This convenience function builds a TorchDatasetAdapter with commonly used
    transforms for object detection, including resizing, augmentation, and
    normalization.

    Args:
        dataset: Source BoxLab Dataset instance.
        image_size: Target image size. Can be:
            - int: Square resize (size, size)
            - tuple: (height, width)
            - None: No resizing
        augment: Whether to apply data augmentation. Includes:
            - Random horizontal flip (p=0.5)
            - Color jitter (brightness, contrast, saturation, hue)
            - Random affine (rotation, translation, scale)
        normalize: Whether to normalize images using ImageNet statistics:
            - mean=[0.485, 0.456, 0.406]
            - std=[0.229, 0.224, 0.225]
        *transforms: Additional user-defined transforms to append.
        return_format: Bounding box format ("xyxy", "xywh", or "cxcywh").

    Returns:
        TorchDatasetAdapter instance with configured transforms.

    Note:
        This function requires torch, torchvision, and pillow to be installed.
        Install with: `pip install torch torchvision pillow`

    Raises:
        RequiredModuleNotFoundError: If required packages are not installed.

    Example:
        ```python
        from boxlab.dataset import Dataset
        from boxlab.dataset.torchadapter import build_torchdataset
        from torch.utils.data import DataLoader

        # Create dataset
        dataset = Dataset(name="my_dataset")
        # ... populate dataset ...

        # Build training dataset with augmentation
        train_dataset = build_torchdataset(
            dataset,
            image_size=640,
            augment=True,
            normalize=True,
            return_format="xyxy",
        )

        # Build validation dataset without augmentation
        val_dataset = build_torchdataset(
            dataset,
            image_size=640,
            augment=False,
            normalize=True,
            return_format="xyxy",
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=train_dataset.collate,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=val_dataset.collate,
        )
        ```

    Example:
        ```python
        # Add custom transforms
        from torchvision import transforms as T

        custom_transform = T.GaussianBlur(kernel_size=3)

        torch_dataset = build_torchdataset(
            dataset,
            image_size=640,
            augment=True,
            normalize=True,
            custom_transform,  # Added after normalization
            return_format="cxcywh"
        )
        ```

    Example:
        ```python
        # Different image sizes
        torch_dataset = build_torchdataset(
            dataset,
            image_size=(800, 600),  # height x width
            augment=False,
            normalize=False,
        )
        ```
    """
    logger.info(
        f"Building PyTorch dataset: size={image_size}, augment={augment}, "
        f"normalize={normalize}, format={return_format}"
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

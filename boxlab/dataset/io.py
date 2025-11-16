from __future__ import annotations

import logging
import os
import typing as t

from boxlab.dataset import Dataset
from boxlab.dataset import SplitRatio
from boxlab.dataset.plugins import NamingStrategy
from boxlab.dataset.plugins.naming import OriginalNaming
from boxlab.dataset.plugins.naming import PrefixNaming
from boxlab.dataset.plugins.naming import SequentialNaming
from boxlab.dataset.plugins.naming import UUIDNaming
from boxlab.dataset.plugins.registry import get_exporter
from boxlab.dataset.plugins.registry import get_loader
from boxlab.dataset.plugins.registry import list_exporters
from boxlab.dataset.plugins.registry import list_loaders

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | os.PathLike[str],
    name: str | None = None,
    format: str = "auto",
    **kwargs: t.Any,
) -> Dataset:
    """Load dataset from file or directory.

    Args:
        path: Path to dataset
        name: Optional name for the dataset
        format: Dataset format ('coco', 'yolo', or 'auto' to detect)
        **kwargs: Additional format-specific parameters

    Returns:
        Loaded Dataset

    Raises:
        ValueError: If format is unsupported or auto-detection fails
        FileNotFoundError: If path doesn't exist
    """
    import pathlib

    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Auto-detect format
    if format == "auto":
        format = _detect_format(path)
        logger.info(f"Auto-detected format: {format}")

    # Get loader
    try:
        loader = get_loader(format)
    except KeyError as e:
        available = list_loaders()
        raise ValueError(f"Unsupported format: {format}. Available formats: {available}") from e

    # Load dataset
    return loader.load(path, name, **kwargs)


def export_dataset(
    dataset: Dataset,
    output_dir: str | os.PathLike[str],
    format: str,
    split_ratio: SplitRatio | None = None,
    seed: int | None = None,
    naming: str | NamingStrategy = "original",
    copy_images: bool = True,
    **kwargs: t.Any,
) -> None:
    """Export dataset to specified format.

    Args:
        dataset: Dataset to export
        output_dir: Output directory path
        format: Export format ('coco', 'yolo', etc.)
        split_ratio: Optional train/val/test split ratios
        seed: Random seed for reproducibility
        naming: File naming strategy ('original', 'prefix', 'uuid', 'sequential')
                or a NamingStrategy instance
        copy_images: Whether to copy image files
        **kwargs: Additional format-specific parameters

    Raises:
        ValueError: If format is unsupported

    Examples:
        >>> # Export to COCO format
        >>> export_dataset(dataset, "./output", format="coco")

        >>> # Export to YOLO with splits
        >>> export_dataset(
        ...     dataset,
        ...     "./output",
        ...     format="yolo",
        ...     split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
        ...     naming="prefix",
        ... )
    """
    # Get exporter
    try:
        exporter = get_exporter(format)
    except KeyError as e:
        available = list_exporters()
        raise ValueError(f"Unsupported format: {format}. Available formats: {available}") from e

    # Get naming strategy
    naming_strategy = _get_naming_strategy(naming)

    # Export
    exporter.export(
        dataset,
        output_dir,
        split_ratio=split_ratio,
        seed=seed,
        naming_strategy=naming_strategy,
        copy_images=copy_images,
        **kwargs,
    )


def _detect_format(path: os.PathLike[str]) -> str:
    """Auto-detect dataset format from path.

    Args:
        path: Path to dataset

    Returns:
        Detected format name

    Raises:
        ValueError: If format cannot be detected
    """
    import pathlib

    path = pathlib.Path(path)

    # Check if it's a COCO JSON file
    if path.is_file() and path.suffix.lower() == ".json":
        return "coco"

    # Check if it's a YOLO directory
    if path.is_dir():
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        if (yaml_files and (path / "images").exists()) or (
            path / "labels"
        ).exists():  # Check for YOLO structure
            return "yolo"

    raise ValueError(f"Could not auto-detect format for: {path}. Please specify format explicitly.")


def _get_naming_strategy(naming: str | NamingStrategy) -> NamingStrategy:
    """Get naming strategy from string or instance.

    Args:
        naming: Strategy name or instance

    Returns:
        NamingStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    if isinstance(naming, str):
        strategies = {
            "original": OriginalNaming(),
            "prefix": PrefixNaming(),
            "uuid": UUIDNaming(with_source_prefix=False),
            "uuid_prefix": UUIDNaming(with_source_prefix=True),
            "sequential": SequentialNaming(with_source_prefix=False),
            "sequential_prefix": SequentialNaming(with_source_prefix=True),
        }

        if naming not in strategies:
            raise ValueError(
                f"Unknown naming strategy: {naming}. Available: {list(strategies.keys())}"
            )

        return strategies[naming]

    return naming


def get_supported_formats() -> dict[str, dict[str, t.Any]]:
    """Get information about all supported formats.

    Returns:
        Dictionary with loader and exporter information

    Examples:
        >>> formats = get_supported_formats()
        >>> print(formats["loaders"])
        {'coco': {...}, 'yolo': {...}}
    """
    from boxlab.dataset.plugins.registry import get_exporter_info
    from boxlab.dataset.plugins.registry import get_loader_info

    return {
        "loaders": get_loader_info(),
        "exporters": get_exporter_info(),
    }


def merge(
    *datasets: Dataset,
    name: str = "merged_dataset",
    resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
    preserve_sources: bool = True,
    fix_duplicates: bool = True,
) -> Dataset:
    """Merge multiple datasets into one.

    Args:
        *datasets: Datasets to merge
        name: Name for the merged dataset
        resolve_conflicts: How to handle category conflicts
        preserve_sources: Whether to preserve source information
        fix_duplicates: Whether to fix duplicate category names

    Returns:
        Merged dataset
    """
    import logging

    from boxlab.exceptions import DatasetMergeError

    logger = logging.getLogger(__name__)

    if not datasets:
        raise DatasetMergeError("No datasets provided for merging")

    logger.info(f"Merging {len(datasets)} datasets into '{name}'")

    # Fix duplicates if requested
    if fix_duplicates:
        for dataset in datasets:
            dataset.fix_duplicate_categories()

    merged = Dataset(name=name)

    for i, dataset in enumerate(datasets):
        logger.debug(f"Merging dataset {i + 1}/{len(datasets)}: {dataset.name}")
        merged = merged.merge(
            dataset,
            resolve_conflicts=resolve_conflicts,
            preserve_sources=preserve_sources,
        )

    logger.info(f"Successfully merged {len(datasets)} datasets")

    return merged

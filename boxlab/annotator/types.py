from __future__ import annotations

import typing as t


class AuditReportMetadata(t.TypedDict):
    version: str
    """Version of the audit report schema."""

    generated_at: str
    """ISO 8601 timestamp of when the report was generated."""

    dataset_path: str
    """Path to the audited dataset."""

    dataset_format: str
    """Format of the audited dataset."""


class AuditReportStatisticsTags(t.TypedDict):
    available_tags: list[str]
    """List of available tags in the dataset."""

    tagged_images: int
    """Number of images that have been tagged."""


class AuditReportStatistics(t.TypedDict):
    total_images: int
    """Total number of images in the dataset."""

    audit_status: dict[str, int]
    """Statistics on the audit status of images."""

    tags: AuditReportStatisticsTags
    """Tagging statistics."""


class AuditReportImageEntry(t.TypedDict, total=False):
    image_id: str
    """Unique identifier for the image."""

    filename: str
    """Filename of the image."""

    source: str
    """Source of the image."""

    original_annotations: list[dict[str, t.Any]]
    """Original annotations for the image."""

    modified_annotations: list[dict[str, t.Any]] | None
    """Modified annotations for the image, if any changes were made."""

    has_changes: bool
    """Indicates whether the annotations were modified."""

    audit_status: str
    """Audit status of the image."""

    audit_comment: str | None
    """Optional comment from the audit."""

    tags: list[str]
    """List of tags associated with the image."""

    timestamp: str
    """ISO 8601 timestamp of when the image was last modified."""


class AuditReport(t.TypedDict):
    metadata: AuditReportMetadata
    """Metadata about the audit report."""

    statistics: AuditReportStatistics
    """Statistical summary of the audit."""

    images: list[AuditReportImageEntry]
    """List of audited image entries."""

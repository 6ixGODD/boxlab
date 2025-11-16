from __future__ import annotations

import datetime
import json
import pathlib
import typing as t

if t.TYPE_CHECKING:
    from boxlab.dataset.types import Annotation


class WorkspaceData(t.TypedDict):
    """Workspace data structure."""

    version: str
    created_at: str
    modified_at: str
    dataset_path: str
    dataset_format: str
    loaded_splits: list[str]
    current_split: str
    current_index: int
    original_annotations: dict[str, list[dict[str, t.Any]]]
    modified_annotations: dict[str, list[dict[str, t.Any]]]
    audit_mode: bool
    audit_status: dict[str, str]
    audit_comments: dict[str, str]  # image_id -> comment
    available_tags: list[str]
    image_tags: dict[str, list[str]]


class Workspace:
    """Manage annotation workspace for save/load sessions."""

    __version__ = "1.0"  # Bump for audit comments

    def __init__(self) -> None:
        """Initialize workspace."""
        self.data: WorkspaceData = {
            "version": self.__version__,
            "created_at": datetime.datetime.now().isoformat(),
            "modified_at": datetime.datetime.now().isoformat(),
            "dataset_path": "",
            "dataset_format": "",
            "loaded_splits": [],
            "current_split": "",
            "current_index": 0,
            "original_annotations": {},
            "modified_annotations": {},
            "audit_mode": False,
            "audit_status": {},
            "audit_comments": {},
            "available_tags": [],
            "image_tags": {},
        }

    def save(self, filepath: str | pathlib.Path) -> None:
        """Save workspace to file."""
        self.data["modified_at"] = datetime.datetime.now().isoformat()

        filepath = pathlib.Path(filepath)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | pathlib.Path) -> Workspace:
        """Load workspace from file."""
        workspace = cls()

        filepath = pathlib.Path(filepath)
        with filepath.open(encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Migrate old versions
        version = loaded_data.get("version", "1.0")
        if version == "1.0":
            loaded_data["version"] = "1.2"
            loaded_data["available_tags"] = []
            loaded_data["image_tags"] = {}
            loaded_data["audit_comments"] = {}
        elif version == "1.1":
            loaded_data["version"] = "1.2"
            loaded_data["audit_comments"] = {}

        workspace.data = loaded_data
        return workspace

    def set_audit_comment(self, image_id: str, comment: str) -> None:
        """Set audit comment for an image."""
        self.data["audit_comments"][image_id] = comment

    def get_audit_comment(self, image_id: str) -> str:
        """Get audit comment for an image."""
        return self.data["audit_comments"].get(image_id, "")

    def set_dataset_info(self, path: str, format_type: str, splits: list[str]) -> None:
        """Set dataset information."""
        self.data["dataset_path"] = path
        self.data["dataset_format"] = format_type
        self.data["loaded_splits"] = splits

    def set_current_position(self, split: str, index: int) -> None:
        """Set current viewing position."""
        self.data["current_split"] = split
        self.data["current_index"] = index

    def set_original_annotations(self, image_id: str, annotations: list[Annotation]) -> None:
        """Store original annotations for an image."""
        if image_id in self.data["original_annotations"]:
            return

        ann_dicts = []
        for ann in annotations:
            ann_dict = {
                "bbox": {
                    "x_min": ann.bbox.x_min,
                    "y_min": ann.bbox.y_min,
                    "x_max": ann.bbox.x_max,
                    "y_max": ann.bbox.y_max,
                },
                "category_id": ann.category_id,
                "category_name": ann.category_name,
                "image_id": ann.image_id,
                "annotation_id": ann.annotation_id,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
            }
            ann_dicts.append(ann_dict)

        self.data["original_annotations"][image_id] = ann_dicts

    def set_modified_annotations(self, image_id: str, annotations: list[Annotation]) -> None:
        """Store modified annotations for an image."""
        ann_dicts = []
        for ann in annotations:
            ann_dict = {
                "bbox": {
                    "x_min": ann.bbox.x_min,
                    "y_min": ann.bbox.y_min,
                    "x_max": ann.bbox.x_max,
                    "y_max": ann.bbox.y_max,
                },
                "category_id": ann.category_id,
                "category_name": ann.category_name,
                "image_id": ann.image_id,
                "annotation_id": ann.annotation_id,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
            }
            ann_dicts.append(ann_dict)

        self.data["modified_annotations"][image_id] = ann_dicts

    def get_modified_annotations(self, image_id: str) -> list[Annotation] | None:
        """Get modified annotations for an image."""
        from boxlab.dataset.types import Annotation
        from boxlab.dataset.types import BBox

        if image_id not in self.data["modified_annotations"]:
            return None

        ann_dicts = self.data["modified_annotations"][image_id]
        annotations = []

        for ann_dict in ann_dicts:
            bbox_dict = ann_dict["bbox"]
            bbox = BBox(
                x_min=bbox_dict["x_min"],
                y_min=bbox_dict["y_min"],
                x_max=bbox_dict["x_max"],
                y_max=bbox_dict["y_max"],
            )

            ann = Annotation(
                bbox=bbox,
                category_id=ann_dict["category_id"],
                category_name=ann_dict["category_name"],
                image_id=ann_dict["image_id"],
                annotation_id=ann_dict["annotation_id"],
                area=ann_dict["area"],
                iscrowd=ann_dict["iscrowd"],
            )
            annotations.append(ann)

        return annotations

    def set_audit_mode(self, enabled: bool) -> None:
        """Enable/disable audit mode."""
        self.data["audit_mode"] = enabled

    def set_audit_status(self, image_id: str, status: str) -> None:
        """Set audit status for an image."""
        self.data["audit_status"][image_id] = status

    def get_audit_status(self, image_id: str) -> str:
        """Get audit status for an image."""
        return self.data["audit_status"].get(image_id, "pending")

    def set_available_tags(self, tags: list[str]) -> None:
        """Set available tags."""
        self.data["available_tags"] = tags

    def add_tag(self, tag: str) -> None:
        """Add a new tag to available tags."""
        if tag not in self.data["available_tags"]:
            self.data["available_tags"].append(tag)

    def set_image_tags(self, image_id: str, tags: list[str]) -> None:
        """Set tags for an image."""
        self.data["image_tags"][image_id] = tags

    def get_image_tags(self, image_id: str) -> list[str]:
        """Get tags for an image."""
        return self.data["image_tags"].get(image_id, [])

    def get_audit_statistics(self) -> dict[str, int]:
        """Get audit statistics."""
        stats = {"approved": 0, "rejected": 0, "pending": 0}

        for status in self.data["audit_status"].values():
            if status in stats:
                stats[status] += 1

        return stats

    def generate_audit_report_json(
        self,
        output_path: str | pathlib.Path,
        image_info_map: dict[str, tuple[str, str]],
    ) -> None:
        """Generate audit report as JSON file.

        Args:
            output_path: Path to save JSON report
            image_info_map: Mapping of image_id to (filename, source)
        """
        output_path = pathlib.Path(output_path)

        # Build report data
        report = {
            "metadata": {
                "version": self.__version__,
                "generated_at": datetime.datetime.now().isoformat(),
                "dataset_path": self.data["dataset_path"],
                "dataset_format": self.data["dataset_format"],
            },
            "statistics": {
                "total_images": len(
                    set(self.data["original_annotations"].keys())
                    | set(self.data["audit_status"].keys())
                ),
                "audit_status": self.get_audit_statistics(),
                "tags": {
                    "available_tags": self.data["available_tags"],
                    "tagged_images": len(self.data["image_tags"]),
                },
            },
            "images": [],
        }

        # Get all image IDs
        all_image_ids = set(self.data["original_annotations"].keys()) | set(
            self.data["audit_status"].keys()
        )

        for img_id in sorted(all_image_ids):
            filename, source = image_info_map.get(img_id, (img_id, "Unknown"))

            # Original annotations
            original_anns = self.data["original_annotations"].get(img_id, [])

            # Modified annotations
            modified_anns = self.data["modified_annotations"].get(img_id, original_anns)

            # Check if there are changes
            has_changes = modified_anns != original_anns

            # Audit status
            audit_status = self.data["audit_status"].get(img_id, "pending")

            # Tags
            tags = self.data["image_tags"].get(img_id, [])

            # Audit comment
            comment = self.data["audit_comments"].get(img_id, "")

            image_entry = {
                "image_id": img_id,
                "filename": filename,
                "source": source,
                "original_annotations": original_anns,
                "modified_annotations": modified_anns if has_changes else None,
                "has_changes": has_changes,
                "audit_status": audit_status,
                "audit_comment": comment,
                "tags": tags,
                "timestamp": self.data["modified_at"],
            }

            report["images"].append(image_entry)

        # Save JSON
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

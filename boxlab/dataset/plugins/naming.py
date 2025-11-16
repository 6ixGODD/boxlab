from __future__ import annotations

import pathlib
import uuid


class OriginalNaming:
    """Keep original filenames."""

    def gen_name(self, origin: str, _source: str | None, _image_id: str, /) -> str:
        return origin


class PrefixNaming:
    """Add source prefix to filenames."""

    def gen_name(self, origin: str, source: str | None, _image_id: str, /) -> str:
        if source:
            stem = pathlib.Path(origin).stem
            suffix = pathlib.Path(origin).suffix
            return f"{source}_{stem}{suffix}"
        return origin


class UUIDNaming:
    """Generate UUID-based filenames."""

    def __init__(self, with_source_prefix: bool = True):
        self.with_source_prefix = with_source_prefix

    def gen_name(self, origin: str, source_name: str | None, _image_id: str, /) -> str:
        suffix = pathlib.Path(origin).suffix
        unique_id = str(uuid.uuid4())[:8]

        if self.with_source_prefix and source_name:
            return f"{source_name}_{unique_id}{suffix}"
        return f"{unique_id}{suffix}"


class SequentialNaming:
    """Generate sequential numbered filenames."""

    def __init__(self, with_source_prefix: bool = True):
        self.with_source_prefix = with_source_prefix
        self.counter = 0

    def gen_name(self, original_name: str, source_name: str | None, _image_id: str) -> str:
        suffix = pathlib.Path(original_name).suffix
        self.counter += 1

        if self.with_source_prefix and source_name:
            return f"{source_name}_{self.counter:06d}{suffix}"
        return f"{self.counter:06d}{suffix}"

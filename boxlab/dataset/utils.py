from __future__ import annotations

import uuid


def gen_uid(prefix: str = "", suffix: str = "", with_hyphens: bool = False) -> str:
    """Generate a unique identifier with optional prefix and suffix.

    Args:
        prefix (str): A string to prepend to the identifier.
        suffix (str): A string to append to the identifier.
        with_hyphens (bool): Whether to include hyphens in the UUID.

    Returns:
        str: The generated unique identifier.
    """
    unique_id = str(uuid.uuid4()) if with_hyphens else uuid.uuid4().hex
    return f"{prefix}{unique_id}{suffix}"

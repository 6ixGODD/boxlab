from __future__ import annotations

import typing as t


class BoxlabError(Exception):
    """Base exception for all Boxlab-related errors.

    This is the root exception class for the Boxlab library. All custom exceptions
    in Boxlab inherit from this class, making it easy to catch any Boxlab-specific
    error.

    Each exception has an associated error code for programmatic error handling
    and a default message that can be overridden.

    Attributes:
        default_message: Default error message for this exception type.
        code: Unique error code identifying this exception type.
        message: The actual error message for this instance.

    Example:
        ```python
        from boxlab.exceptions import BoxlabError

        try:
            # Some Boxlab operation
            pass
        except BoxlabError as e:
            print(f"Boxlab error occurred: {e}")
            print(f"Error code: {e.code}")
        ```

    Example:
        ```python
        # Create custom exception
        class CustomError(BoxlabError):
            default_message = "Custom error occurred"
            code = 999


        raise CustomError("Something went wrong")
        ```
    """

    default_message: t.ClassVar[str] = "An error occurred in Boxlab."
    code: t.ClassVar[int] = 1

    def __init__(self, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Custom error message. If None, uses default_message.
        """
        self.message = message or self.default_message
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error string with code.

        Returns:
            Formatted string in the form "[Error {code}] {message}".
        """
        return f"[Error {self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of the exception.

        Returns:
            String showing class name, code, and message.
        """
        return f"{self.__class__.__name__}(code={self.code}, message={self.message!r})"


class RequiredModuleNotFoundError(BoxlabError):
    """Exception raised when a required module is not found.

    This exception is raised when attempting to use functionality that requires
    optional dependencies (e.g., torch, torchvision, PIL) that are not installed.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 2.
        module_name: Tuple of missing module names.

    Example:
        ```python
        from boxlab.exceptions import RequiredModuleNotFoundError

        try:
            from boxlab.dataset.torchadapter import (
                build_torchdataset,
            )
        except RequiredModuleNotFoundError as e:
            print(f"Missing modules: {e.module_name}")
            print(
                "Install with: pip install torch torchvision pillow"
            )
        ```

    Example:
        ```python
        # Raise for multiple missing modules
        raise RequiredModuleNotFoundError(
            "torch",
            "torchvision",
            message="PyTorch packages required",
        )
        ```
    """

    default_message = "Required module {module_name} not found. Please install it to proceed."
    code: t.ClassVar[int] = 2

    def __init__(self, *module_name: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            *module_name: One or more names of missing modules.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.module_name = module_name
        full_message = message or self.default_message.format(module_name=module_name)
        super().__init__(full_message)

    def __repr__(self) -> str:
        """Return detailed representation of the exception.

        Returns:
            String showing class name, module names, and message.
        """
        return (
            f"{self.__class__.__name__}(module_name={self.module_name!r}, message={self.message!r})"
        )


class DatasetError(BoxlabError):
    """Base exception for dataset-related errors.

    This is the base class for all dataset operation errors. Catch this exception
    to handle any dataset-related issue.

    Attributes:
        default_message: Default error message.
        code: Error code 10.

    Example:
        ```python
        from boxlab.exceptions import DatasetError

        try:
            dataset.some_operation()
        except DatasetError as e:
            print(f"Dataset operation failed: {e}")
        ```
    """

    default_message = "Dataset operation failed."
    code: t.ClassVar[int] = 10


class DatasetNotFoundError(DatasetError):
    """Exception raised when a dataset is not found.

    Raised when attempting to load or access a dataset that doesn't exist at
    the specified path.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 11.
        path: The path where the dataset was expected.

    Example:
        ```python
        from boxlab.exceptions import DatasetNotFoundError
        from boxlab.dataset.io import load_dataset

        try:
            dataset = load_dataset("/nonexistent/path")
        except DatasetNotFoundError as e:
            print(f"Dataset not found at: {e.path}")
        ```

    Example:
        ```python
        # Raise with custom message
        raise DatasetNotFoundError(
            "/data/missing.json",
            message="COCO annotations file not found",
        )
        ```
    """

    default_message = "Dataset not found at path: {path}"
    code: t.ClassVar[int] = 11

    def __init__(self, path: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            path: The path where the dataset was expected.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.path = path
        full_message = message or self.default_message.format(path=path)
        super().__init__(full_message)


class DatasetFormatError(DatasetError):
    """Exception raised for invalid dataset format.

    Raised when a dataset file or structure doesn't match the expected format,
    or when the format is not recognized.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 12.
        format: The format that caused the error.

    Example:
        ```python
        from boxlab.exceptions import DatasetFormatError
        from boxlab.dataset.io import load_dataset

        try:
            dataset = load_dataset("data.txt", format="unknown")
        except DatasetFormatError as e:
            print(f"Invalid format: {e.format}")
        ```

    Example:
        ```python
        # Raise for malformed JSON
        raise DatasetFormatError(
            "coco",
            message="COCO JSON missing required 'images' field",
        )
        ```
    """

    default_message = "Invalid dataset format: {format}"
    code: t.ClassVar[int] = 12

    def __init__(self, format: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            format: The format identifier that caused the error.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.format = format
        full_message = message or self.default_message.format(format=format)
        super().__init__(full_message)


class DatasetLoadError(DatasetError):
    """Exception raised when dataset loading fails.

    Raised when an error occurs during the dataset loading process, such as
    corrupted files, parsing errors, or I/O failures.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 13.
        reason: Description of why loading failed.

    Example:
        ```python
        from boxlab.exceptions import DatasetLoadError
        from boxlab.dataset.io import load_dataset

        try:
            dataset = load_dataset("corrupted.json")
        except DatasetLoadError as e:
            print(f"Load failed: {e.reason}")
        ```

    Example:
        ```python
        # Raise for parsing error
        raise DatasetLoadError(
            reason="JSON parse error at line 42",
            message="Failed to parse COCO annotations",
        )
        ```
    """

    default_message = "Failed to load dataset: {reason}"
    code: t.ClassVar[int] = 13

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Description of why loading failed.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class DatasetExportError(DatasetError):
    """Exception raised when dataset export fails.

    Raised when an error occurs during dataset export, such as I/O failures,
    permission errors, or format conversion issues.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 14.
        reason: Description of why export failed.

    Example:
        ```python
        from boxlab.exceptions import DatasetExportError
        from boxlab.dataset.io import export_dataset

        try:
            export_dataset(dataset, "/readonly/path", format="coco")
        except DatasetExportError as e:
            print(f"Export failed: {e.reason}")
        ```

    Example:
        ```python
        # Raise for permission error
        raise DatasetExportError(
            reason="Permission denied",
            message="Cannot write to output directory",
        )
        ```
    """

    default_message = "Failed to export dataset: {reason}"
    code: t.ClassVar[int] = 14

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Description of why export failed.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class DatasetMergeError(DatasetError):
    """Exception raised when dataset merge fails.

    Raised when an error occurs during dataset merging operations, such as
    incompatible datasets or merge conflicts.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 15.
        reason: Description of why merge failed.

    Example:
        ```python
        from boxlab.exceptions import DatasetMergeError
        from boxlab.dataset.io import merge

        try:
            merged = merge()  # No datasets provided
        except DatasetMergeError as e:
            print(f"Merge failed: {e.reason}")
        ```

    Example:
        ```python
        # Raise for incompatible datasets
        raise DatasetMergeError(
            reason="Incompatible category structures",
            message="Cannot merge datasets with different category schemas",
        )
        ```
    """

    default_message = "Failed to merge datasets: {reason}"
    code: t.ClassVar[int] = 15

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Description of why merge failed.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class CategoryConflictError(DatasetError):
    """Exception raised when category conflicts occur.

    Raised during dataset merging when category names conflict and the conflict
    resolution strategy is set to 'error'.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 16.
        category_name: The category name that caused the conflict.

    Example:
        ```python
        from boxlab.exceptions import CategoryConflictError
        from boxlab.dataset.io import merge

        try:
            merged = merge(ds1, ds2, resolve_conflicts="error")
        except CategoryConflictError as e:
            print(f"Category conflict: {e.category_name}")
        ```

    Example:
        ```python
        # Raise for duplicate category
        raise CategoryConflictError(
            "person",
            message="Category 'person' exists with different IDs",
        )
        ```
    """

    default_message = "Category conflict detected: {category_name}"
    code: t.ClassVar[int] = 16

    def __init__(self, category_name: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            category_name: The category name that caused the conflict.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.category_name = category_name
        full_message = message or self.default_message.format(category_name=category_name)
        super().__init__(full_message)


class ValidationError(BoxlabError):
    """Exception raised for validation errors.

    Raised when data fails validation checks, such as invalid split ratios,
    malformed bounding boxes, or constraint violations.

    Attributes:
        default_message: Template for the default error message.
        code: Error code 20.
        reason: Description of what failed validation.

    Example:
        ```python
        from boxlab.exceptions import ValidationError
        from boxlab.dataset.types import SplitRatio

        try:
            split = SplitRatio(0.5, 0.3, 0.1)
            split.validate()
        except ValidationError as e:
            print(f"Validation failed: {e.reason}")
        ```

    Example:
        ```python
        # Raise for invalid bbox
        raise ValidationError(
            reason="x_max must be greater than x_min",
            message="Invalid bounding box coordinates",
        )
        ```
    """

    default_message = "Validation failed: {reason}"
    code: t.ClassVar[int] = 20

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Description of what failed validation.
            message: Custom error message. If None, uses formatted default_message.
        """
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)

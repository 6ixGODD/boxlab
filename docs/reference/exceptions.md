# Exceptions

::: boxlab.exceptions
options:
show_root_heading: true
show_source: true
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true

## Overview

BoxLab uses a hierarchical exception system for clear and specific error handling. All exceptions inherit from
`BoxlabError`, making it easy to catch all BoxLab-specific errors or handle specific error types.

Each exception has:

- A unique **error code** for programmatic handling
- A **default message** template
- Additional **context attributes** relevant to the error type

## Exception Hierarchy

```
BoxlabError (code: 1)
├── RequiredModuleNotFoundError (code: 2)
├── DatasetError (code: 10)
│   ├── DatasetNotFoundError (code: 11)
│   ├── DatasetFormatError (code: 12)
│   ├── DatasetLoadError (code: 13)
│   ├── DatasetExportError (code: 14)
│   ├── DatasetMergeError (code: 15)
│   └── CategoryConflictError (code: 16)
└── ValidationError (code: 20)
```

## Error Codes

Error codes can be used for programmatic error handling:

| Code | Exception                   | Category        |
|------|-----------------------------|-----------------|
| 1    | BoxlabError                 | Base            |
| 2    | RequiredModuleNotFoundError | Dependencies    |
| 10   | DatasetError                | Dataset (base)  |
| 11   | DatasetNotFoundError        | Dataset I/O     |
| 12   | DatasetFormatError          | Dataset Format  |
| 13   | DatasetLoadError            | Dataset Loading |
| 14   | DatasetExportError          | Dataset Export  |
| 15   | DatasetMergeError           | Dataset Merge   |
| 16   | CategoryConflictError       | Dataset Merge   |
| 20   | ValidationError             | Validation      |


## See Also

- [Dataset Core](dataset/index.md) - Core dataset functionality
- [I/O Operations](dataset/io.md) - Loading and exporting
- [Types](dataset/types.md) - Data structures and validation

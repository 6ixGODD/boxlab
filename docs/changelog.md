# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Data augmentation support in GUI annotator
- Audit report analysis tools
- Additional format plugins (Pascal VOC, TFRecord)
- Documentation tutorials (especially for GUI annotator)

## [0.1.1] - 2025-01-16

### Fixed
- Fixed broken documentation links throughout the documentation site
- Fixed Windows Poetry installation issues in GitHub Actions workflows
- Fixed ANSI color formatting in CLI output display
- Fixed mkdocs strict mode build failures

### Changed
- Simplified CI workflows with direct pip installation of Poetry
- Improved CLI help formatter with better spacing and readability
- Updated documentation navigation structure
- Enhanced README with more comprehensive examples

### Added
- Added changelog and license pages to documentation
- Added About section to documentation site
- Added better error handling for empty test suites in CI


## [0.1.0] - 2025-01-16

### Added

#### Core Features
- Initial release of BoxLab
- Dataset management system with multi-source support
- COCO and YOLO format support
- Bidirectional format conversion (COCO ↔ YOLO)
- Dataset merging with conflict resolution
- Dataset splitting with customizable ratios
- Comprehensive statistics and analysis

#### CLI Tools
- `boxlab dataset info` - Display dataset information and statistics
- `boxlab dataset convert` - Convert between formats with flexible options
- `boxlab dataset merge` - Merge multiple datasets with source tracking
- `boxlab dataset visualize` - Generate visualizations and plots
- `boxlab annotator` - Launch GUI annotation application
- Rich formatted output with tables and progress indicators
- Verbose mode for detailed debugging

#### GUI Annotator
- Interactive annotation viewer and editor
- Visual bounding box editing with drag-to-resize
- Multi-split navigation (train/val/test)
- Audit workflow with approve/reject functionality
- Image tagging system
- Workspace persistence (.cyw files)
- Auto-backup on crashes
- Keyboard shortcuts for efficient workflows
- Real-time statistics display

#### Python API
- `Dataset` class for dataset management
- `load_dataset()` and `export_dataset()` functions
- Type-safe data structures (`BBox`, `Annotation`, `ImageInfo`)
- Plugin system for custom format support
- `LoaderPlugin` and `ExporterPlugin` base classes
- Plugin registry for format discovery
- Multiple naming strategies (original, UUID, sequential, prefix)

#### PyTorch Integration
- `TorchDatasetAdapter` for PyTorch Dataset compatibility
- `build_torchdataset()` convenience function
- Built-in augmentation support
- Flexible transform pipelines
- Custom collate functions
- Multiple coordinate formats (xyxy, xywh, cxcywh)
- Normalization options

#### Documentation
- Complete user guides (installation, quick start, CLI reference)
- Comprehensive API reference with examples
- Advanced usage guide (PyTorch, plugins, automation)
- Interactive annotator guide
- MkDocs Material theme documentation site

#### Development
- Poetry for dependency management
- Ruff for linting and formatting
- MyPy for type checking
- Pytest for testing
- GitHub Actions workflows (tests, docs, publish)
- Pre-commit hooks

### Format Support

#### COCO Format
- Full COCO JSON specification support
- Image metadata preservation
- Category management
- Annotation area calculation
- Source tracking in merged datasets

#### YOLO Format
- YOLOv5/YOLOv8 format support
- data.yaml configuration
- Normalized coordinate handling
- Split-based directory structure
- Automatic path resolution

### Plugin System
- `COCOLoader` and `COCOExporter`
- `YOLOLoader` and `YOLOExporter`
- Extensible architecture for custom formats
- Format auto-detection
- Validation hooks

### Dependencies
- Python 3.10+ support
- NumPy for numerical operations
- Pillow for image handling
- Matplotlib for visualizations
- PyYAML for YAML parsing
- Pandas for data analysis
- Optional PyTorch integration

### System Support
- Cross-platform (Linux, macOS, Windows)
- Python 3.10, 3.11, 3.12 support
- Terminal color support with auto-detection
- Graceful degradation without optional dependencies

## Version History

### Semantic Versioning

BoxLab follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Cycle

- **Major releases**: Significant new features or breaking changes
- **Minor releases**: New features, improvements, non-breaking changes
- **Patch releases**: Bug fixes, documentation updates, minor improvements

### Deprecation Policy

- Deprecated features will be marked in documentation and emit warnings
- Deprecated features will be supported for at least one minor version
- Breaking changes will be clearly documented in changelog
- Migration guides will be provided for major version updates

## Categories

### Added
New features or functionality.

### Changed
Changes in existing functionality.

### Deprecated
Features that will be removed in upcoming releases.

### Removed
Features that have been removed.

### Fixed
Bug fixes.

### Security
Security vulnerability fixes.

## Links

- [PyPI Package](https://pypi.org/project/boxlab/)
- [GitHub Repository](https://github.com/6ixGODD/boxlab)
- [Documentation](https://6ixgodd.github.io/boxlab)
- [Issue Tracker](https://github.com/6ixGODD/boxlab/issues)

---

**Note**: Dates are in ISO 8601 format (YYYY-MM-DD). All notable changes for each version are documented above.

[Unreleased]: https://github.com/6ixGODD/boxlab/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/6ixGODD/boxlab/releases/tag/v0.1.1
[0.1.0]: https://github.com/6ixGODD/boxlab/releases/tag/v0.1.0

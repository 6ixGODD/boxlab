# Annotator

::: boxlab.annotator.AnnotatorApp
options:
show_root_heading: true
show_source: false
heading_level: 2
members_order: source
show_signature_annotations: true
separate_signature: true
members:
- __init__
- run
- import_dataset
- export_dataset
- load_workspace
- save_workspace
- save_workspace_as
- next_image
- prev_image
- approve_current
- reject_current

## Overview

The Boxlab Annotator is a desktop GUI application for viewing, editing, and auditing object detection datasets. It
provides an intuitive interface for working with COCO and YOLO format datasets, with features designed for efficient
annotation workflows.

## Key Features

### Dataset Management

- Import COCO and YOLO format datasets
- Import raw image directories
- Export to COCO or YOLO with flexible naming strategies
- Multi-split support (train/val/test)
- Workspace persistence (.cyw files)

### Annotation Editing

- Visual bounding box display
- Drag-to-resize with corner and edge handles
- Point-and-click selection
- Delete and undo operations
- Category assignment
- Real-time validation

### Audit Workflow

- Approve/reject images
- Add audit comments
- Filter by audit status
- Generate audit reports
- Track audit statistics
- JSON report export

### Additional Features

- Image tagging system
- Auto-backup on crashes
- Keyboard shortcuts
- Zoom and pan
- Status indicators
- Image metadata display

## Application Layout

```
┌──────────────────────────────────────────────────────────────┐
│ Menu Bar (File | View | Audit | Help)                        │
├─────────────┬───────────────────────────────┬────────────────┤
│             │                               │                │
│   Image     │        Canvas                 │   Info Panel   │
│   List      │     (Annotations)             │   - Metadata   │
│   Panel     │                               │   - Tags       │
│   - Splits  │                               │   - Stats      │
│   - Filter  │                               │   - Audit      │
│             │                               │                │
│             ├───────────────────────────────┤                │
│             │    Control Panel              │                │
│             │  [◀ Prev] [Split] [Next ▶]    │                │
│             │  [Edit] [Category] [Audit]    │                │
├─────────────┴───────────────────────────────┴────────────────┤
│ Status Bar                                                   │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from boxlab.annotator import AnnotatorApp

# Create and run the application
app = AnnotatorApp()
app.run()
```

### Command Line

```bash
# Run from command line
python -m boxlab annotator

# Or use the installed command
boxlab annotator
```

## Workflows

### Import and View Workflow

1. Launch the annotator
2. File → Import Dataset...
3. Select format (COCO/YOLO/Raw)
4. Choose dataset path
5. Select splits to load
6. Browse images with arrow keys or image list

### Annotation Editing Workflow

1. Import dataset
2. Enable Edit Mode (Control Panel)
3. Select category
4. Click and drag to create bbox
5. Drag corners/edges to resize
6. Click bbox to select
7. Press Delete to remove
8. Ctrl+Z to undo
9. File → Save Workspace

### Audit Workflow

1. Import dataset
2. Enable Audit Mode (Control Panel)
3. Review current image
4. Press F1 to approve or F2 to reject
5. Add comments (optional)
6. Automatically moves to next image
7. View progress in Info Panel
8. Export audit report when complete

### Workspace Management

1. Work on dataset with edits/audits
2. File → Save Workspace (Ctrl+S)
3. Saves as .cyw file
4. File → Open Workspace (Ctrl+O)
5. Restores complete state

## Keyboard Shortcuts

### Navigation

- `←` Previous image
- `→` Next image

### View

- `Ctrl + Mouse Wheel` Zoom in/out
- `Mouse Wheel` Scroll vertically
- `Shift + Mouse Wheel` Scroll horizontally
- `Ctrl + 0` Reset zoom
- `Middle Mouse Drag` Pan view

### Editing

- `Delete` Delete selected annotation
- `Ctrl + Z` Undo last change
- `Right Click` Show context menu
- Drag corners: Resize diagonally
- Drag edges: Resize horizontally/vertically
- Click bbox: Select annotation

### Audit

- `F1` Approve current image
- `F2` Reject current image

### File

- `Ctrl + O` Open workspace
- `Ctrl + S` Save workspace
- `Ctrl + Shift + S` Save workspace as

## Auto-Backup

The annotator automatically creates backups in case of crashes:

- Backup location: `~/.boxlab/backups/`
- Triggered on uncaught exceptions
- Includes all annotations and audit status
- Displayed in error dialog
- Load via File → Open Workspace

## File Formats

### Workspace Files (.cyw)

Workspace files preserve:

- Dataset structure and metadata
- All annotations (original + edits)
- Audit status and comments
- Image tags
- Current view state

### Audit Reports (JSON)

```json
{
  "generated_at": "2025-01-16T13:51:01Z",
  "total_images": 1000,
  "approved": 850,
  "rejected": 100,
  "pending": 50,
  "images": [
    {
      "image_id": "001",
      "file_name": "image1.jpg",
      "status": "approved",
      "comment": "Good quality",
      "audited_at": "2025-01-16T13:45:00Z"
    }
  ]
}
```


## See Also

- [Dataset Core](dataset/index.md) - Dataset management
- [I/O Operations](dataset/io.md) - Loading and exporting
- [Types](dataset/types.md) - Data structures
- [COCO Plugin](dataset/plugins/coco.md) - COCO format
- [YOLO Plugin](dataset/plugins/yolo.md) - YOLO format

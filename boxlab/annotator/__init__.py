from __future__ import annotations

import logging
import pathlib
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import types
import typing as t

from boxlab import __version__
from boxlab.annotator.canvas import AnnotationCanvas
from boxlab.annotator.controller import AnnotationController
from boxlab.annotator.dialogs import ExportDialog
from boxlab.annotator.dialogs import ImportDialog
from boxlab.annotator.panels import ControlPanel
from boxlab.annotator.panels import ImageListPanel
from boxlab.annotator.panels import InfoPanel

logger = logging.getLogger(__name__)


class AnnotatorApp:
    """Main application class for the Boxlab Annotator GUI.

    The AnnotatorApp provides a complete graphical interface for viewing, editing,
    and auditing object detection datasets. It supports COCO and YOLO formats,
    multi-split datasets, annotation editing, and audit workflows.

    The application consists of several key components:
    - Image list panel (left): Browse and filter images
    - Annotation canvas (center): View and edit annotations
    - Control panel (center bottom): Navigation and mode controls
    - Info panel (right): Display image metadata and statistics
    - Menu bar: File operations and settings
    - Status bar: Current operation status

    Features:
        - Import/export COCO and YOLO datasets
        - Visual annotation editing with drag-to-resize
        - Multi-split support (train/val/test)
        - Audit mode with approve/reject workflow
        - Workspace persistence (.cyw files)
        - Auto-backup on crashes
        - Image tagging system
        - Keyboard shortcuts for efficiency

    Args:
        root: Optional Tkinter root window. If None, creates a new Tk instance.

    Attributes:
        root: The main Tkinter window.
        controller: AnnotationController managing dataset state.
        canvas: AnnotationCanvas for displaying and editing images.
        control_panel: ControlPanel for navigation and mode controls.
        image_list_panel: ImageListPanel for browsing images.
        info_panel: InfoPanel for displaying metadata.
        backup_dir: Directory for auto-backup files.
        status_var: Tkinter StringVar for status bar text.

    Example:
        ```python
        from boxlab.annotator import AnnotatorApp

        # Create and run the application
        app = AnnotatorApp()
        app.run()
        ```

    Example:
        ```python
        # Create with custom root window
        import tkinter as tk

        root = tk.Tk()
        root.geometry("1920x1080")

        app = AnnotatorApp(root)
        app.run()
        ```
    """

    def __init__(self, root: tk.Tk | None = None):
        """Initialize the annotation application.

        Args:
            root: Optional Tkinter root window. If None, creates new instance.
        """
        self.root = root or tk.Tk()

        # Initialize controller
        self.controller = AnnotationController()

        self._update_window_title()

        # Start maximized
        self.root.state("zoomed")

        # Modern theme
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        # Setup auto-backup directory
        self.backup_dir = pathlib.Path.home() / ".boxlab" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._setup_exception_handler()
        self._setup_menu()
        self._setup_layout()
        self._setup_bindings()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("Annotator application initialized")

    def _update_window_title(self) -> None:
        """Update window title based on workspace state.

        Displays the workspace filename and a modified indicator (*) if
        there are unsaved changes.
        """
        base_title = "Boxlab Annotator"

        if self.controller.workspace_path:
            workspace_name = pathlib.Path(self.controller.workspace_path).name
            modified_marker = " *" if self.controller.workspace_modified else ""
            self.root.title(f"{base_title} - {workspace_name}{modified_marker}")
        else:
            self.root.title(base_title)

    def _setup_exception_handler(self) -> None:
        """Setup global exception handler for auto-backup.

        Catches uncaught exceptions, attempts to create an auto-backup
        of the current workspace, and displays an error dialog with
        backup location.
        """
        import sys

        def exception_handler(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: types.TracebackType | None,
        ) -> None:
            """Handle uncaught exceptions with auto-backup."""
            if isinstance(exc_value, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

            if self.controller.has_dataset():
                try:
                    backup_path = self.controller.auto_backup_workspace(self.backup_dir)
                    messagebox.showerror(
                        "Unexpected Error",
                        f"An unexpected error occurred:\n{exc_value}\n\n"
                        f"An automatic backup has been saved to:\n{backup_path}\n\n"
                        "You can load this backup to restore your work.",
                    )
                except Exception as backup_error:
                    logger.error(f"Failed to create auto-backup: {backup_error}")
                    messagebox.showerror(
                        "Critical Error",
                        f"An unexpected error occurred:\n{exc_value}\n\n"
                        f"Additionally, failed to create backup:\n{backup_error}",
                    )
            else:
                messagebox.showerror(
                    "Unexpected Error", f"An unexpected error occurred:\n{exc_value}"
                )

            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = exception_handler

    def _setup_menu(self) -> None:
        """Setup application menu bar.

        Creates File, View, Audit, and Help menus with keyboard
        shortcuts.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Workspace...", command=self.load_workspace, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Save Workspace", command=self.save_workspace, accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Save Workspace As...", command=self.save_workspace_as, accelerator="Ctrl+Shift+S"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Import Dataset...", command=self.import_dataset)
        file_menu.add_command(label="Export Dataset...", command=self.export_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Alt+F4")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Next Image", command=self.next_image, accelerator="→")
        view_menu.add_command(label="Previous Image", command=self.prev_image, accelerator="←")
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")

        # Audit menu
        audit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Audit", menu=audit_menu)
        audit_menu.add_command(
            label="Approve Current", command=self.approve_current, accelerator="F1"
        )
        audit_menu.add_command(
            label="Reject Current", command=self.reject_current, accelerator="F2"
        )
        audit_menu.add_separator()
        audit_menu.add_command(label="View Audit Report", command=self.show_audit_report)
        audit_menu.add_command(label="Export Audit Report...", command=self.export_audit_report)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

    def _setup_layout(self) -> None:
        """Setup main application layout.

        Creates three-column layout with image list (left), canvas and
        controls (center), and info panel (right), plus status bar at
        bottom.
        """
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)

        # Left panel: Image list
        self.image_list_panel = ImageListPanel(
            self.root,  # type: ignore
            on_select=self.on_image_selected,
            width=280,
        )
        self.image_list_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)

        # Center: Canvas and controls
        center_frame = ttk.Frame(self.root)
        center_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=5)
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        # Canvas
        self.canvas = AnnotationCanvas(
            center_frame,
            on_annotation_changed=self.on_annotations_changed,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Control panel
        self.control_panel = ControlPanel(
            center_frame,
            on_prev=self.prev_image,
            on_next=self.next_image,
            on_split_changed=self.on_split_changed,
            on_edit_mode_changed=self.on_edit_mode_changed,
            on_category_changed=self.on_category_changed,
            on_audit_mode_changed=self.on_audit_mode_changed,
            on_approve=self.approve_current,
            on_reject=self.reject_current,
            on_audit_comment_changed=self.on_audit_comment_changed,
        )
        self.control_panel.grid(row=1, column=0, sticky="ew")

        # Right panel: Info
        self.info_panel = InfoPanel(
            self.root,  # type: ignore
            width=320,
            on_tags_changed=self.on_tags_changed,
            on_new_tag=self.on_new_tag_created,
        )
        self.info_panel.grid(row=0, column=2, sticky="nsew", padx=(2, 5), pady=5)

        # Status bar
        status_frame = ttk.Frame(self.root, relief="sunken")
        status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.status_var = tk.StringVar(value="Ready - Import a dataset to begin")
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w", padding=(5, 2)).pack(
            side="left", fill="x", expand=True
        )

    def _setup_bindings(self) -> None:
        """Setup keyboard shortcuts.

        Binds common keyboard shortcuts for navigation, editing, saving,
        and audit operations.
        """
        self.root.bind("<Left>", lambda _: self.prev_image())
        self.root.bind("<Right>", lambda _: self.next_image())
        self.root.bind("<Control-o>", lambda _: self.load_workspace())
        self.root.bind("<Control-s>", lambda _: self.save_workspace())
        self.root.bind("<Control-Shift-S>", lambda _: self.save_workspace_as())
        self.root.bind("<Control-plus>", lambda _: self.zoom_in())
        self.root.bind("<Control-equal>", lambda _: self.zoom_in())
        self.root.bind("<Control-minus>", lambda _: self.zoom_out())
        self.root.bind("<Control-0>", lambda _: self.reset_zoom())

        # Delete and Undo
        self.root.bind("<Delete>", lambda _: self.delete_selected())
        self.root.bind("<Control-z>", lambda _: self.undo())

        # Audit shortcuts
        self.root.bind("<F1>", lambda _: self.approve_current())
        self.root.bind("<F2>", lambda _: self.reject_current())

    # File Operations ==========================================================

    def import_dataset(self) -> None:
        """Import dataset from COCO, YOLO, or raw image directory.

        Shows ImportDialog for format selection and path input, then
        loads the dataset in a background thread with progress
        indication.
        """
        if not self._check_unsaved_changes():
            return

        dialog = ImportDialog(self.root)  # type: ignore
        result = dialog.show()

        if result:
            format_type = result["format"]
            path = result["path"]
            splits = result["splits"]
            initial_tags = result.get("tags", [])

            # Show loading dialog
            import threading

            from boxlab.annotator.dialogs import LoadingDialog

            loading = LoadingDialog(self.root, "Importing Dataset")  # type: ignore
            loading.update_message(f"Loading {format_type.upper()} dataset...")
            loading.update_status(f"Reading from {path}")

            # Use threading to avoid blocking
            error_container: list[None | BaseException] = [None]

            def do_import_thread() -> None:
                try:
                    if format_type == "raw":
                        categories = result.get("categories", [])
                        self.controller.load_dataset(
                            path, format_type, splits, categories, initial_tags
                        )
                    else:
                        self.controller.load_dataset(
                            path, format_type, splits, initial_tags=initial_tags
                        )

                except Exception as e:
                    error_container[0] = e
                    logger.error(f"Failed to load dataset: {e}", exc_info=True)

            def check_thread_done(thread: threading.Thread, loading_dialog: LoadingDialog) -> None:
                def _check_thread_done(*_args: t.Any) -> None:
                    check_thread_done(thread, loading_dialog)

                if thread.is_alive():
                    self.root.after(100, _check_thread_done, "fuck")
                else:
                    loading_dialog.close()

                    if error_container[0]:
                        messagebox.showerror(
                            "Error", f"Failed to load dataset:\n{error_container[0]}"
                        )
                        self.status_var.set("Ready")
                    else:
                        loading_dialog.update_status("Updating UI...")

                        self.controller.workspace_path = None
                        self.controller.set_workspace_modified(True)
                        self._update_window_title()

                        self.update_after_load()

                        self.status_var.set(
                            f"✓ Loaded {self.controller.total_images()} images from {len(splits)} split(s)"
                        )

            load_thread = threading.Thread(target=do_import_thread, daemon=True)
            load_thread.start()

            def _check_thread_done(*_args: t.Any) -> None:
                check_thread_done(load_thread, loading)

            self.root.after(100, _check_thread_done, "fuck")

    def export_dataset(self) -> None:
        """Export dataset to COCO or YOLO format.

        Shows ExportDialog for format and naming options, then exports
        the dataset. Generates audit report JSON if in audit mode.
        """
        if not self.controller.has_dataset():
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        if self.controller.audit_mode:
            stats = self.controller.get_audit_statistics()
            if stats["pending"] > 0:
                response = messagebox.askyesno(
                    "Pending Audits",
                    f"There are {stats['pending']} images pending audit.\n\n"
                    "Do you want to continue exporting?",
                )
                if not response:
                    return

        dialog = ExportDialog(self.root)  # type: ignore
        result = dialog.show()

        if result:
            output_dir = result["output_dir"]
            format_type = result["format"]
            naming = result["naming"]

            self.status_var.set(f"Exporting to {format_type.upper()} format...")
            self.root.update()

            try:
                self.controller.export_dataset(output_dir, format_type, naming)

                self.status_var.set(f"✓ Exported to {output_dir}")

                if self.controller.audit_mode:
                    import pathlib

                    report_path = pathlib.Path(output_dir) / "audit_report.json"
                    self.controller.generate_audit_report_json(str(report_path))

                    messagebox.showinfo(
                        "Success",
                        f"Dataset exported successfully to:\n{output_dir}\n\n"
                        f"Audit report (JSON) saved to:\n{report_path}",
                    )
                else:
                    messagebox.showinfo(
                        "Success", f"Dataset exported successfully to:\n{output_dir}"
                    )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export dataset:\n{e}")
                self.status_var.set("Ready")
                logger.error(f"Failed to export dataset: {e}", exc_info=True)

    def load_workspace(self) -> None:
        """Load workspace from .cyw file.

        Prompts for workspace file, loads it in background thread, and
        restores the complete application state including annotations,
        tags, and audit status.
        """
        if not self._check_unsaved_changes():
            return

        filepath = filedialog.askopenfilename(
            filetypes=[("Boxlab Workspace", "*.cyw"), ("All Files", "*.*")],
            title="Open Workspace",
        )

        if filepath:
            import threading

            from boxlab.annotator.dialogs import LoadingDialog

            loading = LoadingDialog(self.root, "Loading Workspace")  # type: ignore
            loading.update_message("Loading workspace...")
            loading.update_status(f"Reading {filepath}")

            error_container: list[BaseException | None] = [None]

            def do_load_thread() -> None:
                try:
                    self.controller.load_workspace(filepath)
                except Exception as e:
                    error_container[0] = e
                    logger.error(f"Failed to load workspace: {e}", exc_info=True)

            def check_thread_done(thread: threading.Thread, loading_dialog: LoadingDialog) -> None:
                def _check_thread_done(*_args: t.Any) -> None:
                    check_thread_done(thread, loading_dialog)

                if thread.is_alive():
                    self.root.after(100, _check_thread_done, "fuck")
                else:
                    loading_dialog.close()

                    if error_container[0]:
                        messagebox.showerror(
                            "Error", f"Failed to load workspace:\n{error_container[0]}"
                        )
                    else:
                        self._update_window_title()
                        self.update_after_load()

                        current_id = self.controller.get_current_image_id()
                        if current_id:
                            self.load_image(current_id)

                        self.status_var.set("✓ Workspace loaded")
                        messagebox.showinfo("Success", f"Workspace loaded from:\n{filepath}")

            load_thread = threading.Thread(target=do_load_thread, daemon=True)
            load_thread.start()

            def _check_thread_done(*_args: t.Any) -> None:
                check_thread_done(load_thread, loading)

            self.root.after(100, _check_thread_done, "fuck")

    def save_workspace(self) -> None:
        """Save current workspace to associated .cyw file.

        If no workspace path is set, delegates to save_workspace_as().
        """
        if not self.controller.has_dataset():
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        if not self.controller.workspace_path:
            self.save_workspace_as()
            return

        try:
            self.controller.save_workspace()
            self._update_window_title()
            self.status_var.set("✓ Workspace saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save workspace:\n{e}")
            logger.error(f"Failed to save workspace: {e}", exc_info=True)

    def save_workspace_as(self) -> None:
        """Save workspace to a new .cyw file.

        Prompts for file location and saves complete workspace state.
        """
        if not self.controller.has_dataset():
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".cyw",
            filetypes=[("Boxlab Workspace", "*.cyw"), ("All Files", "*.*")],
            title="Save Workspace As",
        )

        if filepath:
            try:
                self.controller.save_workspace(filepath)
                self._update_window_title()
                self.status_var.set(f"✓ Workspace saved: {filepath}")
                messagebox.showinfo("Success", f"Workspace saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save workspace:\n{e}")
                logger.error(f"Failed to save workspace: {e}", exc_info=True)

    # Navigation ===============================================================

    def next_image(self) -> None:
        """Navigate to next image in current split."""
        next_id = self.controller.next_image()
        if next_id:
            self.load_image(next_id)
            self.image_list_panel.select_image(next_id)
            self.update_counter()

    def prev_image(self) -> None:
        """Navigate to previous image in current split."""
        prev_id = self.controller.prev_image()
        if prev_id:
            self.load_image(prev_id)
            self.image_list_panel.select_image(prev_id)
            self.update_counter()

    def load_image(self, image_id: str) -> None:
        """Load and display image with annotations.

        Args:
            image_id: ID of the image to load.
        """
        try:
            image_info = self.controller.get_image_info(image_id)
            annotations = self.controller.get_annotations(image_id)

            if image_info and image_info.path and image_info.path.exists():
                from PIL import Image

                img = Image.open(image_info.path)

                self.canvas.display_image(img, annotations, self.controller.get_categories())

                audit_status = None
                if self.controller.audit_mode:
                    audit_status = self.controller.get_audit_status(image_id)
                    comment = self.controller.get_audit_comment(image_id)
                    self.control_panel.update_audit_status(audit_status)
                    self.control_panel.set_audit_comment(comment)

                tags = self.controller.get_image_tags(image_id)
                self.info_panel.set_current_tags(tags)

                self.info_panel.update_image_info(
                    image_info=image_info,
                    annotations=annotations,
                    source=self.controller.get_image_source(image_id),
                    audit_status=audit_status,
                )

                self.status_var.set(f"🖼 {image_info.file_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            logger.error(f"Failed to load image {image_id}: {e}", exc_info=True)

    # Event Handlers ===========================================================

    def on_image_selected(self, image_id: str) -> None:
        """Handle image selection from list panel.

        Args:
            image_id: ID of the selected image.
        """
        if self.controller.current_split:
            images = self.controller.image_ids_by_split.get(self.controller.current_split, [])
            try:
                self.controller.current_index = images.index(image_id)
                self.load_image(image_id)
                self.update_counter()
            except ValueError:
                pass

    def on_split_changed(self, split: str) -> None:
        """Handle split selection change.

        Args:
            split: Name of the selected split (e.g., "train", "val", "test").
        """
        self.controller.set_current_split(split)

        images = self.controller.get_images_in_split(split)
        self.image_list_panel.set_images(images)

        if self.controller.audit_mode:
            split_audit_map = {}
            for img_id, _ in images:
                split_audit_map[img_id] = self.controller.get_audit_status(img_id)

            self.image_list_panel.set_audit_status_map(split_audit_map)

            audit_stats = self.controller.get_audit_statistics()
            self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

        if images:
            self.load_image(images[0][0])

        self.update_counter()

    def on_annotations_changed(self) -> None:
        """Handle annotation modifications on canvas."""
        current_id = self.controller.get_current_image_id()
        if current_id:
            annotations = self.canvas.get_annotations()
            self.controller.update_annotations(current_id, annotations)

            image_info = self.controller.get_image_info(current_id)
            if image_info:
                self.info_panel.update_image_info(
                    image_info=image_info,
                    annotations=annotations,
                    source=self.controller.get_image_source(current_id),
                )

        self._update_window_title()
        self.status_var.set("⚠️ Unsaved changes")

    def on_edit_mode_changed(self, enabled: bool) -> None:
        """Handle edit mode toggle.

        Args:
            enabled: Whether edit mode is enabled.
        """
        self.canvas.set_edit_mode(enabled)
        mode_text = "ON" if enabled else "OFF"
        self.status_var.set(f"Edit mode: {mode_text}")

    def on_category_changed(self, category_id: int | None) -> None:
        """Handle category selection change.

        Args:
            category_id: Selected category ID, or None.
        """
        self.canvas.set_current_category(category_id)

    def on_tags_changed(self, tags: list[str]) -> None:
        """Handle tags change for current image.

        Args:
            tags: List of tag strings.
        """
        current_id = self.controller.get_current_image_id()

        if current_id:
            self.controller.set_image_tags(current_id, tags)

            self.info_panel.set_current_tags(tags)
            self._update_window_title()
            self.status_var.set("⚠️ Unsaved changes")

    def on_new_tag_created(self, tag: str) -> None:
        """Handle new tag creation.

        Args:
            tag: New tag string.
        """
        self.controller.add_tag(tag)
        self.info_panel.set_available_tags(self.controller.available_tags)
        self._update_window_title()
        self.status_var.set(f"✓ New tag created: {tag}")

    # Audit Operations =========================================================

    def on_audit_mode_changed(self, enabled: bool) -> None:
        """Handle audit mode toggle.

        Args:
            enabled: Whether audit mode is enabled.
        """
        self.controller.enable_audit_mode(enabled)

        self.image_list_panel.show_audit_filter(enabled)

        if enabled:
            complete_audit_map = {}
            for split_images in self.controller.image_ids_by_split.values():
                for img_id in split_images:
                    complete_audit_map[img_id] = self.controller.get_audit_status(img_id)

            self.image_list_panel.set_audit_status_map(complete_audit_map)

            audit_stats = self.controller.get_audit_statistics()
            self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

            current_id = self.controller.get_current_image_id()
            if current_id:
                status = self.controller.get_audit_status(current_id)
                self.control_panel.update_audit_status(status)

        mode_text = "ON" if enabled else "OFF"
        self.status_var.set(f"Audit mode: {mode_text}")

    def approve_current(self) -> None:
        """Approve current image and move to next."""
        if not self.controller.audit_mode:
            return

        current_id = self.controller.get_current_image_id()
        if not current_id:
            return

        self.controller.set_audit_status(current_id, "approved")
        self.status_var.set("✓ Image approved")

        next_id = self.controller.next_image()

        if next_id:
            self.load_image(next_id)
            self.image_list_panel.select_image(next_id)
            self.image_list_panel.focus_set()
            self.update_counter()

            status = self.controller.get_audit_status(next_id)
            self.control_panel.update_audit_status(status)

        complete_audit_map = {}
        current_split_images = self.controller.get_images_in_split(
            self.controller.current_split or ""
        )
        for img_id, _ in current_split_images:
            complete_audit_map[img_id] = self.controller.get_audit_status(img_id)

        self.image_list_panel.set_audit_status_map(complete_audit_map)

        audit_stats = self.controller.get_audit_statistics()
        self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

    def reject_current(self) -> None:
        """Reject current image and move to next."""
        if not self.controller.audit_mode:
            return

        current_id = self.controller.get_current_image_id()
        if not current_id:
            return

        self.controller.set_audit_status(current_id, "rejected")
        self.status_var.set("✗ Image rejected")

        next_id = self.controller.next_image()

        if next_id:
            self.load_image(next_id)
            self.image_list_panel.select_image(next_id)
            self.image_list_panel.focus_set()
            self.update_counter()

            status = self.controller.get_audit_status(next_id)
            self.control_panel.update_audit_status(status)

        complete_audit_map = {}
        current_split_images = self.controller.get_images_in_split(
            self.controller.current_split or ""
        )
        for img_id, _ in current_split_images:
            complete_audit_map[img_id] = self.controller.get_audit_status(img_id)

        self.image_list_panel.set_audit_status_map(complete_audit_map)

        audit_stats = self.controller.get_audit_statistics()
        self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

    def on_audit_comment_changed(self, comment: str) -> None:
        """Handle audit comment change.

        Args:
            comment: Comment text.
        """
        current_id = self.controller.get_current_image_id()
        if current_id:
            self.controller.set_audit_comment(current_id, comment)
            self._update_window_title()

    def show_audit_report(self) -> None:
        """Show audit report in a text dialog."""
        if not self.controller.audit_mode:
            messagebox.showinfo("Info", "Audit mode is not enabled")
            return

        report = self.controller.generate_audit_report()

        dialog = tk.Toplevel(self.root)
        dialog.title("Audit Report")
        dialog.geometry("600x400")
        dialog.transient(self.root)  # type: ignore

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"+{x}+{y}")

        text_frame = ttk.Frame(dialog, padding=10)
        text_frame.pack(fill="both", expand=True)

        text = tk.Text(text_frame, wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
        text.config(yscrollcommand=scrollbar.set)

        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        text.insert("1.0", report)
        text.config(state="disabled")

        ttk.Button(dialog, text="Close", command=dialog.destroy, width=12).pack(pady=10)

    def export_audit_report(self) -> None:
        """Export audit report to JSON file."""
        if not self.controller.audit_mode:
            messagebox.showinfo("Info", "Audit mode is not enabled")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Export Audit Report (JSON)",
        )

        if filepath:
            try:
                self.controller.generate_audit_report_json(filepath)
                self.status_var.set(f"✓ Audit report exported: {filepath}")
                messagebox.showinfo("Success", f"Audit report (JSON) exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report:\n{e}")
                logger.error(f"Failed to export report: {e}", exc_info=True)

    # Utility Methods ==========================================================

    def update_after_load(self) -> None:
        """Update UI after loading dataset or workspace."""
        splits = self.controller.get_splits()
        current_split = splits[0] if splits else None

        if current_split:
            self.control_panel.set_splits(splits, current_split)
            self.on_split_changed(current_split)

        categories = self.controller.get_categories()
        self.control_panel.set_categories(categories)

        self.info_panel.set_available_tags(self.controller.available_tags)

        self.info_panel.update_dataset_info(self.controller.get_dataset_info())

        if self.controller.audit_mode:
            self.image_list_panel.show_audit_filter(True)
            self.image_list_panel.set_audit_status_map(self.controller.audit_status)
            self.control_panel.on_audit_mode_toggle()

    def update_counter(self) -> None:
        """Update image counter display."""
        current = self.controller.get_current_index()
        total = self.controller.get_split_size()
        self.control_panel.update_counter(current, total)

    def delete_selected(self) -> None:
        """Delete selected annotation from canvas."""
        if self.canvas.delete_selected():
            self.status_var.set("⚠️ Annotation deleted - Unsaved changes")

    def undo(self) -> None:
        """Undo last annotation change on canvas."""
        if self.canvas.undo():
            self.status_var.set("↶ Undo - Unsaved changes")

    def zoom_in(self) -> None:
        """Zoom in on canvas."""
        self.canvas.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out on canvas."""
        self.canvas.zoom_out()

    def reset_zoom(self) -> None:
        """Reset canvas zoom to 100%."""
        self.canvas.reset_zoom()

    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts help dialog."""
        shortcuts = """
Keyboard Shortcuts:

Navigation:
  ←\t\t\tPrevious image
  →\t\t\tNext image

View:
  Ctrl + MouseWheel\t\tZoom in/out
  MouseWheel\t\tScroll vertically
  Shift + MouseWheel\tScroll horizontally
  Ctrl + 0\t\t\tReset zoom
  Middle Button Drag\t\tPan view

Editing:
  Delete\t\t\tDelete selected annotation
  Ctrl + Z\t\t\tUndo last change
  Right Click\t\tShow delete menu
  Drag Corners\t\tResize (diagonal)
  Drag Edges\t\tResize (horizontal/vertical)
  Click BBox\t\tSelect annotation

Audit:
  F1\t\t\tApprove current image
  F2\t\t\tReject current image

File:
  Ctrl + O\t\t\tOpen workspace
  Ctrl + S\t\t\tSave workspace
  Ctrl + Shift + S\t\tSave workspace as
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def show_about(self) -> None:
        """Show about dialog with version information."""
        about_text = f"""
Boxlab Annotator
Version {__version__}

A simple tool for viewing and editing
object detection datasets in COCO and YOLO formats.

Features:
• Import COCO/YOLO datasets
• View and edit bounding boxes
• Multi-split support
• Export with flexible naming
"""
        messagebox.showinfo("About", about_text)

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user.

        Returns:
            True if should continue, False if cancelled.
        """
        if self.controller.has_unsaved_changes():
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes to the workspace.\n\n"
                "Do you want to save before continuing?",
            )
            if response is None:
                return False
            if response:
                self.save_workspace()
                if self.controller.has_unsaved_changes():
                    return False
        return True

    def on_closing(self) -> None:
        """Handle window close event with unsaved changes check."""
        if not self._check_unsaved_changes():
            return
        self.root.quit()

    def run(self) -> None:
        """Start the application main loop."""
        logger.info("Starting annotator application")
        self.root.mainloop()

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
import typing as t

if t.TYPE_CHECKING:
    from boxlab.dataset.types import Annotation
    from boxlab.dataset.types import ImageInfo


class ImageListPanel(ttk.Frame):
    """Panel for displaying list of images."""

    def __init__(self, parent: tk.Widget, on_select: t.Callable[[str], None], width: int = 250):
        """Initialize image list panel."""
        super().__init__(parent, width=width)

        self.on_select = on_select
        self.all_images: list[tuple[str, str]] = []  # (id, filename)
        self.audit_status_map: dict[str, str] = {}  # image_id -> status
        self.current_filter: str = "all"  # "all", "approved", "rejected", "pending"

        # Audit filter (initially hidden, will be shown when audit mode is on)
        self.filter_frame = ttk.LabelFrame(self, text="Audit Filter", padding=5)

        # Use vertical layout for filter buttons to save horizontal space
        self.filter_var = tk.StringVar(value="all")

        ttk.Radiobutton(
            self.filter_frame,
            text="All",
            variable=self.filter_var,
            value="all",
            command=self._on_filter_changed,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Radiobutton(
            self.filter_frame,
            text="✓ Approved",
            variable=self.filter_var,
            value="approved",
            command=self._on_filter_changed,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Radiobutton(
            self.filter_frame,
            text="✗ Rejected",
            variable=self.filter_var,
            value="rejected",
            command=self._on_filter_changed,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Radiobutton(
            self.filter_frame,
            text="⏳ Pending",
            variable=self.filter_var,
            value="pending",
            command=self._on_filter_changed,
        ).pack(anchor="w", padx=5, pady=2)

        # Count label
        self.count_label = ttk.Label(
            self.filter_frame, text="", font=("Segoe UI", 8), foreground="gray"
        )
        self.count_label.pack(pady=(5, 0))

        # Search box
        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True)

        # Listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, font=("Consolas", 9), selectmode=tk.SINGLE
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Add flag to prevent recursive selection
        self._selecting_programmatically = False

        # Track if filter is currently shown
        self._filter_shown = False

    def set_images(self, images: list[tuple[str, str]]) -> None:
        """Set images to display.

        Args:
            images: List of (image_id, filename) tuples
        """
        self.all_images = images
        self._update_display()

    def set_audit_status_map(self, status_map: dict[str, str]) -> None:
        """Set audit status mapping.

        Args:
            status_map: Mapping of image_id to audit status
        """
        self.audit_status_map = status_map.copy()
        self._update_display()

    def show_audit_filter(self, show: bool) -> None:
        """Show or hide audit filter.

        Args:
            show: True to show filter, False to hide
        """
        if show and not self._filter_shown:
            # Pack filter frame at the top
            self.filter_frame.pack(fill="x", padx=5, pady=(5, 5), before=self.children["!frame"])
            self._filter_shown = True
            self._update_display()
        elif not show and self._filter_shown:
            self.filter_frame.pack_forget()
            self._filter_shown = False
            self.filter_var.set("all")
            self.current_filter = "all"
            self._update_display()

    def _on_filter_changed(self) -> None:
        """Handle filter change."""
        self.current_filter = self.filter_var.get()
        self._update_display()

    def _update_display(self) -> None:
        """Update listbox display based on search and filter."""
        search_term = self.search_var.get().lower()

        self.listbox.delete(0, tk.END)

        # Count for each status
        counts = {"all": 0, "approved": 0, "rejected": 0, "pending": 0}

        filtered_images = []

        for img_id, filename in self.all_images:
            # Apply search filter
            if search_term and search_term not in filename.lower():
                continue

            # Get audit status
            status = self.audit_status_map.get(img_id, "pending")

            # Count
            counts["all"] += 1
            counts[status] += 1

            # Apply audit filter
            if self.current_filter == "all" or self.current_filter == status:
                filtered_images.append((img_id, filename, status))

        # Display filtered images
        for _, filename, status in filtered_images:
            # Add status indicator to filename only if filter is shown
            if self._filter_shown:
                status_icon = {"approved": "✓", "rejected": "✗", "pending": "⏳"}.get(status, "")
                display_name = f"{status_icon} {filename}"
            else:
                display_name = filename

            self.listbox.insert(tk.END, display_name)

        # Update count label (only if filter is shown)
        if self._filter_shown:
            if self.current_filter == "all":
                count_text = f"{len(filtered_images)} / {counts['all']} images"
            else:
                count_text = f"{len(filtered_images)} {self.current_filter}"

            # Add breakdown
            breakdown = f"(✓{counts['approved']} ✗{counts['rejected']} ⏳{counts['pending']})"
            self.count_label.config(text=f"{count_text}\n{breakdown}")

    def _on_search(self, *_args: t.Any) -> None:
        """Handle search text change."""
        self._update_display()

    def select_image(self, image_id: str) -> None:
        """Select image in list."""
        # Set flag to prevent triggering callback
        self._selecting_programmatically = True

        try:
            for _i, (img_id, filename) in enumerate(self.all_images):
                if img_id == image_id:
                    # Find in current filtered list
                    status = self.audit_status_map.get(image_id, "pending")

                    if self._filter_shown:
                        status_icon = {"approved": "✓", "rejected": "✗", "pending": "⏳"}.get(
                            status, ""
                        )
                        display_name = f"{status_icon} {filename}"
                    else:
                        display_name = filename

                    for j in range(self.listbox.size()):
                        if self.listbox.get(j) == display_name:
                            self.listbox.selection_clear(0, tk.END)
                            self.listbox.selection_set(j)
                            self.listbox.see(j)
                            break
                    break
        finally:
            # Always reset flag
            self._selecting_programmatically = False

    def _on_listbox_select(self, _event: tk.Event) -> None:
        """Handle listbox selection."""
        # Skip if we're selecting programmatically
        if self._selecting_programmatically:
            return

        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            display_name = self.listbox.get(index)

            # Remove status icon if present
            if display_name.startswith(("✓ ", "✗ ", "⏳ ")):
                filename = display_name[2:]
            else:
                filename = display_name

            # Find corresponding image ID
            for img_id, fname in self.all_images:
                if fname == filename:
                    self.on_select(img_id)
                    break


class ControlPanel(ttk.Frame):
    """Panel for navigation and editing controls."""

    def __init__(
        self,
        parent: tk.Widget,
        on_prev: t.Callable[[], None],
        on_next: t.Callable[[], None],
        on_split_changed: t.Callable[[str], None],
        on_edit_mode_changed: t.Callable[[bool], None],
        on_category_changed: t.Callable[[int | None], None],
        on_audit_mode_changed: t.Callable[[bool], None],
        on_approve: t.Callable[[], None],
        on_reject: t.Callable[[], None],
        on_audit_comment_changed: t.Callable[[str], None],
    ):
        """Initialize control panel."""
        super().__init__(parent)

        self.on_split_changed = on_split_changed
        self.on_edit_mode_changed = on_edit_mode_changed
        self.on_category_changed = on_category_changed
        self.on_audit_mode_changed = on_audit_mode_changed
        self.on_approve = on_approve
        self.on_reject = on_reject
        self.on_audit_comment_changed = on_audit_comment_changed

        # Navigation & Editing Frame
        self.nav_frame = ttk.Frame(self, padding=5)
        self.nav_frame.pack(fill="x")

        # Left side: Split and navigation
        left_frame = ttk.Frame(self.nav_frame)
        left_frame.pack(side="left", padx=5)

        # Split selector
        ttk.Label(left_frame, text="Split:").pack(side="left", padx=(0, 5))
        self.split_var = tk.StringVar()
        self.split_combo = ttk.Combobox(
            left_frame, textvariable=self.split_var, state="readonly", width=10
        )
        self.split_combo.pack(side="left", padx=(0, 15))
        self.split_combo.bind("<<ComboboxSelected>>", self._on_split_select)

        # Navigation with counter
        ttk.Button(left_frame, text="◀ Prev", command=on_prev, width=8).pack(side="left", padx=2)

        self.counter_label = ttk.Label(left_frame, text="0 / 0", font=("Segoe UI", 10, "bold"))
        self.counter_label.pack(side="left", padx=10)

        ttk.Button(left_frame, text="Next ▶", command=on_next, width=8).pack(side="left", padx=2)

        # Right side: Edit controls and Audit toggle
        right_frame = ttk.Frame(self.nav_frame)
        right_frame.pack(side="right", padx=5)

        # Category selector
        ttk.Label(right_frame, text="Category:").pack(side="left", padx=(0, 5))
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            right_frame, textvariable=self.category_var, state="readonly", width=15
        )
        self.category_combo.pack(side="left", padx=(0, 15))
        self.category_combo.bind("<<ComboboxSelected>>", self.on_category_select)

        # Edit mode toggle
        self.edit_mode_var = tk.BooleanVar(value=False)
        edit_btn = ttk.Checkbutton(
            right_frame,
            text="📝 Edit Mode",
            variable=self.edit_mode_var,
            command=self.on_edit_mode_toggle,
            style="Toolbutton",
        )
        edit_btn.pack(side="left", padx=(0, 10))

        # Audit mode toggle (always visible)
        self.audit_mode_var = tk.BooleanVar(value=False)
        audit_toggle_btn = ttk.Checkbutton(
            right_frame,
            text="🔍 Audit Mode",
            variable=self.audit_mode_var,
            command=self.on_audit_mode_toggle,
            style="Toolbutton",
        )
        audit_toggle_btn.pack(side="left")

        # Audit Controls Frame (conditionally displayed below nav frame)
        self.audit_frame = ttk.Frame(self, padding=(5, 0, 5, 5))

        # Audit buttons
        buttons_row = ttk.Frame(self.audit_frame)
        buttons_row.pack(fill="x", pady=(5, 5))

        ttk.Button(
            buttons_row,
            text="✓ Approve (Ctrl+A)",
            command=on_approve,
            width=18,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons_row,
            text="✗ Reject (Ctrl+R)",
            command=on_reject,
            width=18,
        ).pack(side="left", padx=5)

        # Audit status label
        self.audit_status_label = ttk.Label(buttons_row, text="", font=("Segoe UI", 10, "bold"))
        self.audit_status_label.pack(side="left", padx=10)

        # Audit comment
        comment_frame = ttk.LabelFrame(self.audit_frame, text="Comment", padding=5)
        comment_frame.pack(fill="x")

        comment_inner = ttk.Frame(comment_frame)
        comment_inner.pack(fill="x")

        self.audit_comment_text = tk.Text(
            comment_inner, height=2, width=70, font=("Consolas", 9), wrap="word"
        )
        comment_scrollbar = ttk.Scrollbar(comment_inner, command=self.audit_comment_text.yview)
        self.audit_comment_text.config(yscrollcommand=comment_scrollbar.set)

        self.audit_comment_text.pack(side="left", fill="x", expand=True)
        comment_scrollbar.pack(side="right", fill="y")

        # Bind comment change
        self.audit_comment_text.bind("<KeyRelease>", self.on_comment_changed)

        self.categories: dict[int, str] = {}

    def set_splits(self, splits: list[str], current: str) -> None:
        """Set available splits."""
        self.split_combo["values"] = splits
        self.split_var.set(current)

    def set_categories(self, categories: dict[int, str]) -> None:
        """Set available categories."""
        self.categories = categories
        category_names = list(categories.values())
        self.category_combo["values"] = category_names

        if category_names:
            self.category_var.set(category_names[0])
            self.on_category_select(None)

    def update_counter(self, current: int, total: int) -> None:
        """Update image counter display."""
        self.counter_label.config(text=f"{current} / {total}")

    def update_audit_status(self, status: str) -> None:
        """Update audit status display."""
        status_text = {
            "approved": "✓ Approved",
            "rejected": "✗ Rejected",
            "pending": "⏳ Pending",
        }.get(status, "")

        status_color = {
            "approved": "green",
            "rejected": "red",
            "pending": "orange",
        }.get(status, "black")

        self.audit_status_label.config(text=status_text, foreground=status_color)

    def set_audit_comment(self, comment: str) -> None:
        """Set audit comment text."""
        self.audit_comment_text.delete("1.0", tk.END)
        self.audit_comment_text.insert("1.0", comment)

    def get_audit_comment(self) -> str:
        """Get current audit comment."""
        return self.audit_comment_text.get("1.0", "end-1c")

    def _on_split_select(self, _event: tk.Event) -> None:
        """Handle split selection."""
        split = self.split_var.get()
        self.on_split_changed(split)

    def on_edit_mode_toggle(self) -> None:
        """Handle edit mode toggle."""
        enabled = self.edit_mode_var.get()
        self.on_edit_mode_changed(enabled)

    def on_audit_mode_toggle(self) -> None:
        """Handle audit mode toggle."""
        enabled = self.audit_mode_var.get()
        self.on_audit_mode_changed(enabled)

        # Show/hide audit controls frame
        if enabled:
            self.audit_frame.pack(fill="x", after=self.nav_frame)
        else:
            self.audit_frame.pack_forget()

    def on_category_select(self, _event: tk.Event | None) -> None:
        """Handle category selection."""
        cat_name = self.category_var.get()

        cat_id = None
        for cid, cname in self.categories.items():
            if cname == cat_name:
                cat_id = cid
                break

        self.on_category_changed(cat_id)

    def on_comment_changed(self, _event: tk.Event) -> None:
        """Handle audit comment change."""
        comment = self.get_audit_comment()
        self.on_audit_comment_changed(comment)


class InfoPanel(ttk.Frame):
    """Panel for displaying image and dataset information."""

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 300,
        on_tags_changed: t.Callable[[list[str]], None] | None = None,
        on_new_tag: t.Callable[[str], None] | None = None,
    ):
        """Initialize info panel."""
        super().__init__(parent, width=width)

        self.on_tags_changed = on_tags_changed
        self.on_new_tag = on_new_tag

        # Dataset info
        self.dataset_info_frame = ttk.LabelFrame(self, text="Dataset Overview", padding=5)
        self.dataset_info_frame.pack(fill="x", padx=5, pady=5)

        self.dataset_text = tk.Text(
            self.dataset_info_frame,
            height=6,
            width=30,
            state="disabled",
            font=("Consolas", 9),
            wrap="word",
        )
        self.dataset_text.pack(fill="x")

        # Image info
        self.image_info_frame = ttk.LabelFrame(self, text="Current Image", padding=5)
        self.image_info_frame.pack(fill="x", padx=5, pady=5)

        self.image_text = tk.Text(
            self.image_info_frame,
            height=6,
            width=30,
            state="disabled",
            font=("Consolas", 9),
            wrap="word",
        )
        self.image_text.pack(fill="x")

        # Tags section
        self.tags_frame = ttk.LabelFrame(self, text="Tags", padding=5)
        self.tags_frame.pack(fill="x", padx=5, pady=5)

        # Available tags dropdown
        tags_select_frame = ttk.Frame(self.tags_frame)
        tags_select_frame.pack(fill="x", pady=(0, 5))

        self.available_tags_var = tk.StringVar()
        self.available_tags_combo = ttk.Combobox(
            tags_select_frame,
            textvariable=self.available_tags_var,
            state="readonly",
            width=20,
        )
        self.available_tags_combo.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(tags_select_frame, text="Add", command=self._on_add_tag, width=8).pack(
            side="left"
        )

        # New tag input
        new_tag_frame = ttk.Frame(self.tags_frame)
        new_tag_frame.pack(fill="x", pady=(0, 5))

        self.new_tag_var = tk.StringVar()
        ttk.Entry(new_tag_frame, textvariable=self.new_tag_var, width=20).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(new_tag_frame, text="New", command=self._on_create_new_tag, width=8).pack(
            side="left"
        )

        # Current image tags
        ttk.Label(self.tags_frame, text="Current tags:", font=("Segoe UI", 9)).pack(
            anchor="w", pady=(5, 2)
        )

        self.tags_listbox = tk.Listbox(self.tags_frame, height=4, font=("Consolas", 9))
        tags_list_scrollbar = ttk.Scrollbar(self.tags_frame, command=self.tags_listbox.yview)
        self.tags_listbox.config(yscrollcommand=tags_list_scrollbar.set)

        tags_list_frame = ttk.Frame(self.tags_frame)
        tags_list_frame.pack(fill="x")

        self.tags_listbox.pack(in_=tags_list_frame, side="left", fill="both", expand=True)
        tags_list_scrollbar.pack(in_=tags_list_frame, side="right", fill="y")

        # Remove tag button
        ttk.Button(self.tags_frame, text="Remove Selected", command=self._on_remove_tag).pack(
            pady=(5, 0)
        )

        # Annotations
        self.annotations_frame = ttk.LabelFrame(self, text="Bounding Boxes", padding=5)
        self.annotations_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.annotations_text = tk.Text(
            self.annotations_frame, width=30, state="disabled", font=("Consolas", 9)
        )
        ann_scrollbar = ttk.Scrollbar(self.annotations_frame, command=self.annotations_text.yview)
        self.annotations_text.config(yscrollcommand=ann_scrollbar.set)

        self.annotations_text.pack(side="left", fill="both", expand=True)
        ann_scrollbar.pack(side="right", fill="y")

    def set_available_tags(self, tags: list[str]) -> None:
        """Set available tags for selection."""
        self.available_tags_combo["values"] = tags
        if tags:
            self.available_tags_var.set(tags[0])

    def set_current_tags(self, tags: list[str]) -> None:
        """Set current image tags."""
        self.tags_listbox.delete(0, tk.END)
        for tag in tags:
            self.tags_listbox.insert(tk.END, tag)

    def _on_add_tag(self) -> None:
        """Add selected tag to current image."""
        tag = self.available_tags_var.get()
        if tag and self.on_tags_changed:
            # Get current tags
            current_tags = [self.tags_listbox.get(i) for i in range(self.tags_listbox.size())]
            if tag not in current_tags:
                current_tags.append(tag)
                self.on_tags_changed(current_tags)

    def _on_create_new_tag(self) -> None:
        """Create a new tag."""
        tag = self.new_tag_var.get().strip()
        if tag:
            if self.on_new_tag:
                self.on_new_tag(tag)
            self.new_tag_var.set("")

    def _on_remove_tag(self) -> None:
        """Remove selected tag from current image."""
        selection = self.tags_listbox.curselection()
        if selection and self.on_tags_changed:
            current_tags = [self.tags_listbox.get(i) for i in range(self.tags_listbox.size())]
            # Remove selected tag
            tag_to_remove = self.tags_listbox.get(selection[0])
            current_tags.remove(tag_to_remove)
            self.on_tags_changed(current_tags)

    def update_dataset_info(
        self, info: dict[str, int], audit_stats: dict[str, int] | None = None
    ) -> None:
        """Update dataset information."""
        self.dataset_text.config(state="normal")
        self.dataset_text.delete("1.0", tk.END)

        text = ""
        for key, value in info.items():
            text += f"{key.capitalize()}: {value}\n"

        if audit_stats:
            text += "\nAudit Status:\n"
            text += f"Approved: {audit_stats['approved']}\n"
            text += f"Rejected: {audit_stats['rejected']}\n"
            text += f"Pending: {audit_stats['pending']}\n"

        self.dataset_text.insert("1.0", text)
        self.dataset_text.config(state="disabled")

    def update_image_info(
        self,
        image_info: ImageInfo,
        annotations: list[Annotation],
        source: str | None = None,
        audit_status: str | None = None,
    ) -> None:
        """Update image information."""
        # Image info
        self.image_text.config(state="normal")
        self.image_text.delete("1.0", tk.END)

        text = f"Filename: {image_info.file_name}\n"
        text += f"Size: {image_info.width} x {image_info.height}\n"
        text += f"Annotations: {len(annotations)}\n"
        if source:
            text += f"Source: {source}\n"
        if audit_status:
            status_text = {
                "approved": "✓ Approved",
                "rejected": "✗ Rejected",
                "pending": "⏳ Pending",
            }.get(audit_status, audit_status)
            text += f"Status: {status_text}\n"

        self.image_text.insert("1.0", text)
        self.image_text.config(state="disabled")

        # Annotations
        self.annotations_text.config(state="normal")
        self.annotations_text.delete("1.0", tk.END)

        for i, ann in enumerate(annotations, 1):
            self.annotations_text.insert(
                tk.END,
                f"{i}. {ann.category_name}\n   {ann.bbox.x_min:.0f}, {ann.bbox.y_min:.0f}, "
                f"{ann.bbox.x_max:.0f}, {ann.bbox.y_max:.0f}\n\n",
            )

        self.annotations_text.config(state="disabled")

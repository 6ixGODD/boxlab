from __future__ import annotations

import io
import json
import pathlib
import sys
import typing as t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if t.TYPE_CHECKING:
    from boxlab.annotator.types import AuditReport


def load_report(file: t.IO) -> AuditReport | None:
    # Load report
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"❌ Failed to load report: {e}")


def render_metric_card(title: str, value: str | int, delta: str = "", icon: str = "📊") -> None:
    """Render a custom metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_summary(report: AuditReport) -> None:
    """Display summary statistics with enhanced styling."""
    # Header
    metadata = report["metadata"]
    st.markdown(
        f"""
    <div class="dashboard-header">
        <h1>📊 Audit Dashboard</h1>
        <p>Dataset: {metadata["dataset_format"].upper()} |
           Generated: {metadata["generated_at"][:19].replace("T", " ")} UTC</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Statistics
    stats = report["statistics"]
    total = stats["total_images"]
    audit_status = stats["audit_status"]

    # Main metrics with custom cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Images", total, icon="📁")

    with col2:
        approved = audit_status["approved"]
        pct = f"{approved / total * 100:.1f}%" if total > 0 else "0%"
        render_metric_card("Approved", approved, pct, "✅")

    with col3:
        rejected = audit_status["rejected"]
        pct = f"{rejected / total * 100:.1f}%" if total > 0 else "0%"
        render_metric_card("Rejected", rejected, pct, "❌")

    with col4:
        pending = audit_status["pending"]
        pct = f"{pending / total * 100:.1f}%" if total > 0 else "0%"
        render_metric_card("Pending", pending, pct, "⏳")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tags info
    if stats.get("tags"):
        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Tagged Images", stats["tags"]["tagged_images"], icon="🏷️")
        with col2:
            render_metric_card("Available Tags", len(stats["tags"]["available_tags"]), icon="📝")

        if stats["tags"]["available_tags"]:
            tags_html = " ".join([
                f'<span class="status-badge status-approved">{tag}</span>'
                for tag in stats["tags"]["available_tags"]
            ])
            st.markdown(f"<div style='margin-top: 10px;'>{tags_html}</div>", unsafe_allow_html=True)


def show_charts(report: AuditReport) -> None:
    """Display enhanced charts."""
    st.markdown("## 📈 Visual Analysis")

    audit_status = report["statistics"]["audit_status"]

    # Color scheme
    colors = {"Approved": "#06D6A0", "Rejected": "#C73E1D", "Pending": "#F18F01"}

    # Row 1: Audit status charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Audit Status Distribution")
        status_df = pd.DataFrame({
            "Status": ["Approved", "Rejected", "Pending"],
            "Count": [audit_status["approved"], audit_status["rejected"], audit_status["pending"]],
        })

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=status_df["Status"],
                    values=status_df["Count"],
                    hole=0.4,
                    marker={"colors": [colors[s] for s in status_df["Status"]]},
                    textinfo="label+percent",
                    textfont={"size": 14, "color": "white"},
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=400,
            margin={"t": 0, "b": 0, "l": 0, "r": 0},
        )

        st.plotly_chart(fig, width=True)

    with col2:
        st.markdown("### Status Comparison")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=status_df["Status"],
                    y=status_df["Count"],
                    marker={
                        "color": [colors[s] for s in status_df["Status"]],
                        "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 2},
                    },
                    text=status_df["Count"],
                    textposition="outside",
                    textfont={"size": 14, "color": "white"},
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=400,
            yaxis={"gridcolor": "rgba(255, 255, 255, 0.1)", "showgrid": True},
            xaxis={"showgrid": False},
            margin={"t": 30, "b": 0, "l": 0, "r": 0},
        )

        st.plotly_chart(fig, width=True)

    # Tags distribution
    images = report["images"]
    tags_counter: dict[str, int] = {}
    for img in images:
        for tag in img.get("tags", []):
            tags_counter[tag] = tags_counter.get(tag, 0) + 1

    if tags_counter:
        st.markdown("---")
        st.markdown("### 🏷️ Tag Distribution")
        tags_df = pd.DataFrame({
            "Tag": list(tags_counter.keys()),
            "Count": list(tags_counter.values()),
        }).sort_values("Count", ascending=True)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=tags_df["Count"],
                    y=tags_df["Tag"],
                    orientation="h",
                    marker={
                        "color": tags_df["Count"],
                        "colorscale": "Blues",
                        "showscale": True,
                        "colorbar": {"title": "Count"},
                    },
                    text=tags_df["Count"],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=max(300, len(tags_counter) * 30),
            xaxis={"gridcolor": "rgba(255, 255, 255, 0.1)", "showgrid": True},
            yaxis={"showgrid": False},
            margin={"t": 30, "b": 0, "l": 0, "r": 0},
        )

        st.plotly_chart(fig, width=True)


def show_images_table(report: AuditReport) -> None:
    """Display images table with enhanced filters."""
    st.markdown("## 🖼️ Image Details")

    images = report["images"]
    df = pd.DataFrame(images)

    # Enhanced filters in expandable section
    with st.expander("🔍 Filters & Search", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            status_options = ["All", "approved", "rejected", "pending"]
            selected_status = st.selectbox("📋 Audit Status", status_options)

        with col2:
            changes_options = ["All", "With Changes", "No Changes"]
            selected_changes = st.selectbox("✏️ Modifications", changes_options)

        with col3:
            all_sources = sorted({img["source"] for img in images})
            selected_source = st.selectbox("📂 Source Split", ["All", *all_sources])

        # Tags filter
        all_tags = set()
        for img in images:
            all_tags.update(img.get("tags", []))

        selected_tags = st.multiselect("🏷️ Tags", sorted(all_tags)) if all_tags else []

        # Search box
        search_query = st.text_input("🔎 Search filename", placeholder="Type to search...")

    # Apply filters
    filtered_df = df.copy()

    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["audit_status"] == selected_status]

    if selected_changes == "With Changes":
        filtered_df = filtered_df[filtered_df["has_changes"] == True]  # noqa: E712
    elif selected_changes == "No Changes":
        filtered_df = filtered_df[filtered_df["has_changes"] == False]  # noqa: E712

    if selected_tags:
        filtered_df = filtered_df[
            filtered_df["tags"].apply(lambda tags: any(tag in tags for tag in selected_tags))
        ]

    if selected_source != "All":
        filtered_df = filtered_df[filtered_df["source"] == selected_source]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["filename"].str.contains(search_query, case=False, na=False)
        ]

    # Results info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** images")
    with col2:
        # Download button
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="filtered_audit_report.csv",
            mime="text/csv",
            width=True,
        )

    # Column selector
    cols_to_show = st.multiselect(
        "Select columns to display",
        options=[
            "filename",
            "source",
            "audit_status",
            "has_changes",
            "tags",
            "audit_comment",
            "timestamp",
        ],
        default=["filename", "source", "audit_status", "has_changes", "tags"],
    )

    if cols_to_show:
        display_df = filtered_df[cols_to_show].copy()

        # Format tags
        if "tags" in display_df.columns:
            display_df["tags"] = display_df["tags"].apply(lambda x: ", ".join(x) if x else "")

        # Add status emoji
        if "audit_status" in display_df.columns:
            status_map = {"approved": "✅", "rejected": "❌", "pending": "⏳"}
            display_df["audit_status"] = display_df["audit_status"].apply(
                lambda x: f"{status_map.get(x, '')} {x}"
            )

        # Display table
        st.dataframe(
            display_df,
            width=True,
            height=500,
            column_config={
                "filename": st.column_config.TextColumn("Filename", width="large"),
                "audit_status": st.column_config.TextColumn("Status", width="medium"),
            },
        )


def show_details(report: AuditReport) -> None:
    """Show detailed view for selected images."""
    st.markdown("## 🔍 Detailed View")

    images = report["images"]

    # Image selector
    filenames = [img["filename"] for img in images]
    selected_filename = st.selectbox("Select image to inspect", filenames)

    selected_img = next(img for img in images if img["filename"] == selected_filename)

    # Display in two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Basic Information")

        info_items = [
            ("📄 Filename", selected_img["filename"]),
            ("🆔 Image ID", selected_img["image_id"]),
            ("📂 Source", selected_img["source"]),
        ]

        status_emoji = {"approved": "✅", "rejected": "❌", "pending": "⏳"}
        status = selected_img["audit_status"]
        info_items.append(("📊 Status", f"{status_emoji.get(status, '')} {status}"))

        changes_icon = "✏️" if selected_img["has_changes"] else "📌"
        info_items.append((
            "🔄 Modified",
            f"{changes_icon} {'Yes' if selected_img['has_changes'] else 'No'}",
        ))

        for label, value in info_items:
            st.markdown(
                f"""
            <div class="info-card">
                <strong>{label}</strong><br>
                <span style="color: #8b92a7;">{value}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if selected_img.get("tags"):
            tags_html = " ".join([
                f'<span class="status-badge status-approved">{tag}</span>'
                for tag in selected_img["tags"]
            ])
            st.markdown(
                f"<div style='margin-top: 10px;'><strong>🏷️ Tags:</strong><br>{tags_html}</div>",
                unsafe_allow_html=True,
            )

        if selected_img.get("audit_comment"):
            st.markdown("### 💬 Audit Comment")
            st.info(selected_img["audit_comment"])

    with col2:
        st.markdown("### 📦 Annotations")

        original_count = len(selected_img.get("original_annotations", []))

        col_a, col_b = st.columns(2)
        with col_a:
            render_metric_card("Original", original_count, icon="📦")

        with col_b:
            if selected_img.get("modified_annotations"):
                modified_count = len(selected_img["modified_annotations"] or [])
                delta = f"{modified_count - original_count:+d}"
                render_metric_card("Modified", modified_count, delta, "✏️")
            else:
                render_metric_card("Modified", original_count, "No changes", "📌")

        # Annotation details
        if selected_img.get("original_annotations"):
            with st.expander("📦 View Original Annotations", expanded=False):
                st.json(selected_img["original_annotations"])

        if selected_img.get("modified_annotations"):
            with st.expander("✏️ View Modified Annotations", expanded=False):
                st.json(selected_img["modified_annotations"])


def show_statistics(report: AuditReport) -> None:
    """Show advanced statistics with enhanced visualizations."""
    st.markdown("## 📊 Advanced Statistics")

    images = report["images"]
    df = pd.DataFrame(images)

    # Modification analysis
    st.markdown("### ✏️ Modification Analysis")
    col1, col2 = st.columns(2)

    with col1:
        changes_by_status = (
            df.groupby(["audit_status", "has_changes"]).size().reset_index(name="count")
        )

        fig = px.bar(
            changes_by_status,
            x="audit_status",
            y="count",
            color="has_changes",
            barmode="group",
            labels={"has_changes": "Modified", "audit_status": "Status", "count": "Count"},
            color_discrete_map={True: "#F18F01", False: "#2E86AB"},
            title="Changes by Status",
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            yaxis={"gridcolor": "rgba(255, 255, 255, 0.1)"},
            xaxis={"showgrid": False},
        )

        st.plotly_chart(fig, width=True)

    with col2:
        source_by_status = df.groupby(["source", "audit_status"]).size().reset_index(name="count")

        fig = px.bar(
            source_by_status,
            x="source",
            y="count",
            color="audit_status",
            color_discrete_map={"approved": "#06D6A0", "rejected": "#C73E1D", "pending": "#F18F01"},
            title="Status by Source Split",
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            yaxis={"gridcolor": "rgba(255, 255, 255, 0.1)"},
            xaxis={"showgrid": False},
        )

        st.plotly_chart(fig, width=True)

    # Annotation statistics
    st.markdown("### 📦 Annotation Statistics")

    orig_counts = [len(img.get("original_annotations", [])) for img in images]
    mod_counts = [
        len(img.get("modified_annotations", []) or [])
        if img.get("modified_annotations")
        else len(img.get("original_annotations", []))
        for img in images
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_orig = sum(orig_counts) / len(orig_counts) if orig_counts else 0
        render_metric_card("Avg Original", f"{avg_orig:.2f}", icon="📦")

    with col2:
        avg_mod = sum(mod_counts) / len(mod_counts) if mod_counts else 0
        delta = f"{avg_mod - avg_orig:+.2f}"
        render_metric_card("Avg Modified", f"{avg_mod:.2f}", delta, "✏️")

    with col3:
        total_changes = sum(1 for img in images if img["has_changes"])
        pct = f"{total_changes / len(images) * 100:.1f}%"
        render_metric_card("Images Modified", total_changes, pct, "🔄")


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="Boxlab Audit Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # File uploader or command line arg
    report_file: io.IOBase | None = None
    report: AuditReport | None = None

    if len(sys.argv) > 1:
        report_path = sys.argv[1]
        with pathlib.Path(report_path).open("rb", encoding="utf-8") as f:
            report = load_report(f)

    else:
        st.sidebar.markdown("## 📂 Load Report")
        uploaded_file = st.file_uploader("Upload audit report JSON", type=["json"])
        if uploaded_file:
            report = load_report(uploaded_file)

    if report_file is None:
        st.markdown(
            """
        <div class="dashboard-header">
            <h1>📊 Boxlab Audit Dashboard</h1>
            <p>Interactive analysis of dataset audit results</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.info(
            "👆 Please upload an audit report JSON file or provide path as command line argument"
        )

        st.markdown("### 🚀 Usage")
        st.code("streamlit run audit_dashboard.py -- path/to/audit_report.json", language="bash")

        st.markdown("### 📋 Features")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("✅ **Summary Statistics**\nOverview of audit results")
        with cols[1]:
            st.markdown("📈 **Visual Charts**\nInteractive data visualizations")
        with cols[2]:
            st.markdown("🔍 **Detailed Analysis**\nDive into specific images")

        return

    if report is None:
        return

    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Summary", "📈 Charts", "🖼️ Images", "🔍 Details", "📊 Statistics"],
        label_visibility="collapsed",
    )

    # Page routing
    page_map = {
        "📊 Summary": show_summary,
        "📈 Charts": show_charts,
        "🖼️ Images": show_images_table,
        "🔍 Details": show_details,
        "📊 Statistics": show_statistics,
    }

    page_map[page](report)

    # Footer in sidebar
    st.sidebar.markdown(
        """
    <div id="sidebar-footer" class="sidebar-footer">
        <strong>Boxlab Audit Dashboard</strong>
        <small>v1.0</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

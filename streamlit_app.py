import logging
import os

import streamlit as st
from streamlit.components.v1 import html

from utils import (
    COLORS,
    download_meshes,
    download_semantic_gt,
    get_label_set as _get_label_set,
    MESHES_DIR,
    COLORED_MESHES_DIR,
    SEMANTIC_GT_DIR,
    download_colored_meshes,
)


@st.cache_resource
def get_label_set() -> dict[str, dict[str, list]]:
    return _get_label_set()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
MESHES_PATH = os.path.join(STATIC_DIR, MESHES_DIR)
COLORED_MESHES_PATH = os.path.join(STATIC_DIR, COLORED_MESHES_DIR)
SEMANTIC_GT_PATH = os.path.join(".", SEMANTIC_GT_DIR)


def main() -> None:
    download_meshes(STATIC_DIR)
    download_colored_meshes(STATIC_DIR)
    download_semantic_gt(".")
    label_set = get_label_set()
    uids = [
        uid for mesh_label_set in label_set.values() for uid in mesh_label_set.keys()
    ]

    # Streamlit app
    st.set_page_config(
        page_title="PartObjaverse-Tiny", page_icon=":robot:", layout="wide"
    )
    st.title("PartObjaverse-Tiny")
    st.write(
        "This app visualizes samples from the "
        "[PartObjaverse-Tiny](https://yhyang-myron.github.io/SAMPart3D-website/) dataset (Yang et al., 2024). "
        f"There are {len(uids)} sample meshes in total, across {len(label_set)} categories."
    )

    # Category and page selection
    select_cols = st.columns([4, 2, 1, 1], width=600)
    with select_cols[0]:
        category_select = st.selectbox(
            "Category",
            options=list(label_set.keys()),
            format_func=lambda category: f"{category} ({len(label_set[category])} samples)",
        )
    with select_cols[1]:
        max_pages = (len(label_set[category_select]) + 3) // 4
        page_select = st.selectbox(
            "Page",
            options=list(range(max_pages)),
            format_func=lambda page: f"{page + 1} of {max_pages}",
        )

    # Display samples for the selected category and page
    start_idx = page_select * 4
    end_idx = start_idx + 4
    uids_subset = list(label_set[category_select].keys())[start_idx:end_idx]
    part_labels = list(label_set[category_select].values())[start_idx:end_idx]
    for uid, part_labels in zip(uids_subset, part_labels):
        st.html("<div style='height: 8px;'></div>")
        display_sample_row(uid, part_labels)


def display_sample_row(uid: str, part_labels: list[str]) -> None:
    mesh_file = os.path.join(MESHES_PATH, f"{uid}.glb")
    colored_mesh_file = os.path.join(COLORED_MESHES_PATH, f"{uid}.glb")

    cols = st.columns(3, width=1230)
    with cols[0]:
        model_viewer(mesh_file)
    with cols[1]:
        model_viewer(colored_mesh_file)
    with cols[2]:
        st.write(f"**UID: {uid}**")
        legend_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for label_idx, part_label in enumerate(part_labels):
            color = COLORS[label_idx % len(COLORS)]
            legend_html += legend_entry(color, part_label)
        legend_html += "</div>"
        st.html(legend_html)


def legend_entry(color: str, label: str) -> str:
    return f"""
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 16px; height: 16px; background-color: {color}; margin-right: 8px; border-radius: 2px;"></div>
            <span>{label}</span>
        </div>
        """


def model_viewer(mesh_file: str) -> None:
    html(
        f"""
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
            <model-viewer src="app/{mesh_file}" alt="3D Model" auto-rotate camera-controls
                          style="width: 384px; height: 384px; background-color: #1f2937; border-radius: 8px;">
            </model-viewer>
            """,
        height=400,
    )


if __name__ == "__main__":
    main()

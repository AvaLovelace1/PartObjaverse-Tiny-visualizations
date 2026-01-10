import json
import logging
import multiprocessing as mp
import os
import zipfile

import numpy as np
import streamlit as st
import trimesh
from huggingface_hub import hf_hub_download
from rich.progress import track
from streamlit.components.v1 import html

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def hex2rgb(hex_str: str) -> tuple[int, int, int]:
    if hex_str[0] == "#":
        hex_str = hex_str[1:]
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


COLORS = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#ffffff",
    "#000000",
]
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
MESHES_DIR = "PartObjaverse-Tiny_mesh"
SEMANTIC_GT_DIR = "PartObjaverse-Tiny_semantic_gt"
MESHES_PATH = os.path.join(STATIC_DIR, MESHES_DIR)
COLORED_MESHES_PATH = os.path.join(STATIC_DIR, "PartObjaverse-Tiny_mesh_colored")
SEMANTIC_GT_PATH = os.path.join(".", SEMANTIC_GT_DIR)


def main() -> None:
    download_meshes(STATIC_DIR)
    download_semantic_gt(".")

    label_set = get_label_set()
    uids = [
        uid for mesh_label_set in label_set.values() for uid in mesh_label_set.keys()
    ]
    part_labels = [
        labels
        for mesh_label_set in label_set.values()
        for labels in mesh_label_set.values()
    ]

    # Color meshes
    if not os.path.exists(COLORED_MESHES_PATH):
        os.makedirs(COLORED_MESHES_PATH)
        with mp.Pool() as pool:
            for _ in track(
                pool.imap_unordered(color_mesh_file, uids),
                description="Processing meshes...",
                total=len(uids),
            ):
                pass

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


@st.cache_resource
def download_meshes(out_dir: str) -> None:
    if os.path.exists(os.path.join(out_dir, MESHES_DIR)):
        logger.info(
            f"Mesh directory {os.path.join(out_dir, MESHES_DIR)} already exists. Skipping mesh download."
        )
        return

    file_path = hf_hub_download(
        repo_id="yhyang-myron/PartObjaverse-Tiny",
        filename="PartObjaverse-Tiny_mesh.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(file_path) as zip_ref:
        zip_ref.extractall(out_dir)


@st.cache_resource
def download_semantic_gt(out_dir: str) -> None:
    if os.path.exists(os.path.join(out_dir, SEMANTIC_GT_DIR)):
        logger.info(
            f"Semantic GT directory {SEMANTIC_GT_DIR} already exists. Skipping semantic GT download."
        )
        return
    file_path = hf_hub_download(
        repo_id="yhyang-myron/PartObjaverse-Tiny",
        filename="PartObjaverse-Tiny_semantic_gt.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(file_path) as zip_ref:
        zip_ref.extractall(out_dir)


@st.cache_resource
def get_label_set() -> dict[str, dict[str, list[str]]]:
    file_path = hf_hub_download(
        repo_id="yhyang-myron/PartObjaverse-Tiny",
        filename="PartObjaverse-Tiny_semantic.json",
        repo_type="dataset",
    )
    with open(file_path) as f:
        return json.load(f)


def color_mesh_file(uid: str) -> None:
    mesh_file = os.path.join(MESHES_PATH, f"{uid}.glb")
    mesh = trimesh.load_scene(mesh_file).to_mesh()
    semantic_gt_file = os.path.join(SEMANTIC_GT_PATH, f"{uid}.npy")
    semantic_gt = np.load(semantic_gt_file)
    colored_mesh_file = os.path.join(COLORED_MESHES_PATH, f"{uid}.glb")
    colored_mesh = color_mesh(mesh, semantic_gt)
    colored_mesh.export(colored_mesh_file)


def color_mesh(mesh: trimesh.Trimesh, semantic_gt: np.ndarray) -> trimesh.Trimesh:
    """
    Color mesh based on semantic GT labels from the PartObjaverse-Tiny dataset.
    :param mesh: Mesh to be colored, with N faces.
    :param semantic_gt: Length-N array indicating the semantic label of each face.
    :return: Colored mesh.
    """
    assert len(semantic_gt) == len(mesh.faces)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    for face_idx, label in enumerate(semantic_gt):
        color = hex2rgb(COLORS[label % len(COLORS)]) + (255,)
        mesh.visual.face_colors[face_idx] = color
    return mesh


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

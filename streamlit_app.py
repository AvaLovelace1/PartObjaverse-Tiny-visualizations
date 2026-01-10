import json
import logging
import multiprocessing as mp
import os
import zipfile

import numpy as np
import trimesh
from huggingface_hub import hf_hub_download
from rich.progress import track

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
os.makedirs(COLORED_MESHES_PATH, exist_ok=True)
SEMANTIC_GT_PATH = os.path.join(".", SEMANTIC_GT_DIR)


def main() -> None:
    download_meshes(STATIC_DIR)
    download_semantic_gt(".")

    label_set = get_label_set()
    uids = [
        uid
        for mesh_label_set in label_set.values()
        for uid, part_labels in mesh_label_set.items()
    ]
    with mp.Pool() as pool:
        for _ in track(
            pool.imap_unordered(process_mesh_file, uids),
            description="Processing meshes...",
            total=len(uids),
        ):
            pass


def process_mesh_file(uid: str) -> None:
    mesh_file = os.path.join(MESHES_PATH, f"{uid}.glb")
    mesh = trimesh.load_scene(mesh_file).to_mesh()
    semantic_gt_file = os.path.join(SEMANTIC_GT_PATH, f"{uid}.npy")
    semantic_gt = np.load(semantic_gt_file)
    colored_mesh = color_mesh(mesh, semantic_gt)
    colored_mesh_file = os.path.join(COLORED_MESHES_PATH, f"{uid}.glb")
    colored_mesh.export(colored_mesh_file)


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


def get_label_set() -> dict[str, dict[str, list[str]]]:
    file_path = hf_hub_download(
        repo_id="yhyang-myron/PartObjaverse-Tiny",
        filename="PartObjaverse-Tiny_semantic.json",
        repo_type="dataset",
    )
    with open(file_path) as f:
        return json.load(f)


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


if __name__ == "__main__":
    main()

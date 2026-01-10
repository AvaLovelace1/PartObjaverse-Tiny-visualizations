"""
Color meshes from the PartObjaverse-Tiny dataset based on their semantic part labels, for visualization purposes.
"""

import multiprocessing as mp
import os

import numpy as np
import trimesh
from rich.progress import track

from utils import (
    get_label_set,
    hex2rgb,
    COLORS,
    download_meshes,
    download_semantic_gt,
)

MESHES_PATH = "PartObjaverse-Tiny_mesh"
SEMANTIC_GT_PATH = "PartObjaverse-Tiny_semantic_gt"
COLORED_MESHES_PATH = os.path.join("PartObjaverse-Tiny_mesh_colored")


def main() -> None:
    download_meshes(".")
    download_semantic_gt(".")
    label_set = get_label_set()
    uids = [
        uid for mesh_label_set in label_set.values() for uid in mesh_label_set.keys()
    ]

    os.makedirs(COLORED_MESHES_PATH, exist_ok=True)
    with mp.Pool() as pool:
        for _ in track(
            pool.imap_unordered(process_mesh, uids),
            description="Processing meshes...",
            total=len(uids),
        ):
            pass


def process_mesh(uid: str) -> None:
    mesh_file = os.path.join(MESHES_PATH, f"{uid}.glb")
    mesh = trimesh.load_scene(mesh_file).to_mesh()
    semantic_gt_file = os.path.join(SEMANTIC_GT_PATH, f"{uid}.npy")
    semantic_gt = np.load(semantic_gt_file)
    colored_mesh_file = os.path.join(COLORED_MESHES_PATH, f"{uid}.glb")
    colored_mesh = color_mesh_parts(mesh, semantic_gt)
    colored_mesh.export(colored_mesh_file)


def color_mesh_parts(mesh: trimesh.Trimesh, semantic_gt: np.ndarray) -> trimesh.Trimesh:
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

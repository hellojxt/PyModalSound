import meshio
import subprocess
import os
from glob import glob
import numpy as np


def update_triangle_normals(vertices, triangles):
    """
    vertices: (n, 3)
    triangles: (m, 3)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def tetra_from_mesh(input_mesh, log=False):
    result = subprocess.run(
        ["FloatTetwild_bin", "-i", input_mesh, "--max-threads", "8"],
        capture_output=True,
        text=True,
    )
    if log:
        print(result.stdout, result.stderr)

    tetra_file = input_mesh + "_.msh"
    surface_file = input_mesh + "__sf.obj"
    tetra_mesh = meshio.read(tetra_file)
    surface_mesh = meshio.read(surface_file)
    # remove input_mesh_*
    for f in glob(input_mesh + "_*"):
        os.remove(f)

    return (
        tetra_mesh.points,
        tetra_mesh.cells[0].data,
        surface_mesh.points,
        surface_mesh.cells[0].data,
    )

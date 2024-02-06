from src.model import ModalSoundObj, Material, MatSet, get_spherical_surface_points
from src.visualize import CombinedFig
import numpy as np

obj = ModalSoundObj("mesh.obj")
obj.normalize(0.2)
obj.modal_analysis(32, Material(MatSet.Plastic))

modes = (obj.modes**2).sum(axis=1)
print(obj.surf_vertices.shape, obj.surf_triangles.shape, modes.shape)

# solve the FFAT map of the first mode by BEM
FFAT_map_points = get_spherical_surface_points(obj.surf_vertices)
FFAT_map = np.abs(obj.solve_ffat_map(0, FFAT_map_points))

# visualize the first mode and its FFAT map
CombinedFig().add_mesh(obj.surf_vertices, obj.surf_triangles, modes[:, 0]).add_points(
    FFAT_map_points, FFAT_map
).show()

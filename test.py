from src.modalAnalysis.fem import FEMmodel
import numpy as np

def test_fem():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
    tets = np.array([[0, 1, 2, 3]]).astype(np.int32)
    fem = FEMmodel(vertices, tets)
    print(fem.mass_matrix.to_dense())
    print(fem.stiffness_matrix.to_dense())
    assert fem.mass_matrix.shape == (3*4, 3*4)
    assert fem.stiffness_matrix.shape == (3*4, 3*4)

test_fem()
import os
import numpy as np
from torch.utils.cpp_extension import load
import torch


def scipy2torch(M, device="cuda"):
    device = torch.device(device)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().to(device)
    values = torch.from_numpy(M.data).to(device)
    shape = torch.Size(M.shape)
    M_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return M_torch.coalesce()


def LOBPCG_solver(stiff_matrix, mass_matrix, k):
    vals, vecs = torch.lobpcg(
        stiff_matrix, 6 + k, mass_matrix, tracker=None, largest=False
    )
    return vals.cpu().numpy()[6:], vecs.cpu().numpy()[:, 6:]


class MatSet:
    Ceramic = 2700, 7.2e10, 0.19, 6, 1e-7
    Glass = 2600, 6.2e10, 0.20, 1, 1e-7
    Wood = 750, 1.1e10, 0.25, 60, 2e-6
    Plastic = 1070, 1.4e9, 0.35, 30, 1e-6
    Iron = 8000, 2.1e11, 0.28, 5, 1e-7
    Polycarbonate = 1190, 2.4e9, 0.37, 0.5, 4e-7
    Steel = 7850, 2.0e11, 0.29, 5, 3e-8
    Tin = 7265, 5e10, 0.325, 2, 3e-8

    def random_material():
        mat_lst = [
            MatSet.Ceramic,
            MatSet.Glass,
            MatSet.Wood,
            MatSet.Plastic,
            MatSet.Iron,
            MatSet.Polycarbonate,
            MatSet.Steel,
            MatSet.Tin,
        ]
        return Material(mat_lst[np.random.randint(0, len(mat_lst))])

    def print_relative_freq_k():
        mat_lst = [
            MatSet.Ceramic,
            MatSet.Glass,
            MatSet.Wood,
            MatSet.Plastic,
            MatSet.Iron,
            MatSet.Polycarbonate,
            MatSet.Steel,
            MatSet.Tin,
        ]
        lst = []
        base = mat_lst[0][1] / mat_lst[0][0]
        for mat in mat_lst:
            lst.append((mat[1] / mat[0] / base) ** 0.5)
        print("Ceramic:", lst[0])
        print("Glass:", lst[1])
        print("Wood:", lst[2])
        print("Plastic:", lst[3])
        print("Iron:", lst[4])
        print("Polycarbonate:", lst[5])
        print("Steel:", lst[6])
        print("Tin:", lst[7])


class Material(object):
    def __init__(self, material):
        self.density, self.youngs, self.poison, self.alpha, self.beta = material

    def relative_freq(self):
        return (
            self.youngs / self.density / (MatSet.Ceramic[1] / MatSet.Ceramic[0])
        ) ** 0.5


cuda_dir = os.path.dirname(__file__) + "/cuda"
cuda_include_dir = cuda_dir + "/include"
os.environ["TORCH_EXTENSIONS_DIR"] = cuda_dir + "/build"
src_file = cuda_dir + "/computeMatrix.cu"
cuda_module = load(
    name="matrixAssemble",
    sources=[src_file],
    extra_include_paths=[cuda_include_dir],
    # extra_cuda_cflags=['-O3'],
    # extra_cuda_cflags=['-G -g'],
    #    verbose=True,
)


class FEMmodel:
    def __init__(self, vertices, tets, material=Material(MatSet.Ceramic)):
        self.vertices = vertices.astype(np.float32)
        self.tets = tets.astype(np.int32)
        self.material = material
        self.stiffness_matrix_, self.mass_matrix_ = None, None

    @property
    def mass_matrix(self):
        if self.mass_matrix_ is None:
            values = torch.zeros(
                12 * 12 * self.tets.shape[0], dtype=torch.float32
            ).cuda()
            rows = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = (
                torch.from_numpy(self.vertices)
                .to(torch.float32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            tets_ = (
                torch.from_numpy(self.tets)
                .to(torch.int32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            cuda_module.assemble_mass_matrix(
                vertices_, tets_, values, rows, cols, self.material.density
            )
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size([3 * self.vertices.shape[0], 3 * self.vertices.shape[0]])
            self.mass_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.mass_matrix_

    @property
    def stiffness_matrix(self):
        if self.stiffness_matrix_ is None:
            values = torch.zeros(
                12 * 12 * self.tets.shape[0], dtype=torch.float32
            ).cuda()
            rows = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = (
                torch.from_numpy(self.vertices)
                .to(torch.float32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            tets_ = (
                torch.from_numpy(self.tets)
                .to(torch.int32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            cuda_module.assemble_stiffness_matrix(
                vertices_,
                tets_,
                values,
                rows,
                cols,
                self.material.youngs,
                self.material.poison,
            )
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size([3 * self.vertices.shape[0], 3 * self.vertices.shape[0]])
            self.stiffness_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.stiffness_matrix_

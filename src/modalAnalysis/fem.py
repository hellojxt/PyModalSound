import os
import numpy as np
from torch.utils.cpp_extension import load
import torch
from .material import MatSet, Material

cuda_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/cuda'
cuda_include_dir = cuda_dir + '/include'
os.environ['TORCH_EXTENSIONS_DIR'] = cuda_dir + '/build'
src_file = cuda_dir + '/computeMatrix.cu'
cuda_module = load(name="matrixAssemble",
                   sources=[src_file],
                   extra_include_paths=[cuda_include_dir],
                #    extra_cuda_cflags=['-O3'],
                   extra_cuda_cflags = ['-G -g'],
                   verbose=True,
                   )

class FEMmodel():
    def __init__(self, vertices, tets, material=Material(MatSet.Ceramic)):
        self.vertices = vertices.astype(np.float32)
        self.tets = tets.astype(np.int32)
        self.material = material
        self.stiffness_matrix_, self.mass_matrix_ = None, None

    @property
    def mass_matrix(self):
        if self.mass_matrix_ is None:
            values = torch.zeros(12*12*self.tets.shape[0], dtype=torch.float32).cuda()
            rows = torch.zeros(12*12*self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12*12*self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = torch.from_numpy(self.vertices).to(torch.float32).reshape(-1).contiguous().cuda()
            tets_ = torch.from_numpy(self.tets).to(torch.int32).reshape(-1).contiguous().cuda()
            cuda_module.assemble_mass_matrix(vertices_, tets_, values, rows, cols, self.material.density)
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size(
                [3*self.vertices.shape[0], 3*self.vertices.shape[0]])
            self.mass_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.mass_matrix_

    @property
    def stiffness_matrix(self):
        if self.stiffness_matrix_ is None:
            values = torch.zeros(12*12*self.tets.shape[0], dtype=torch.float32).cuda()
            rows = torch.zeros(12*12*self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12*12*self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = torch.from_numpy(self.vertices).to(torch.float32).reshape(-1).contiguous().cuda()
            tets_ = torch.from_numpy(self.tets).to(torch.int32).reshape(-1).contiguous().cuda()
            cuda_module.assemble_stiffness_matrix(vertices_, tets_, values, rows, cols, self.material.youngs, self.material.poison)
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size(
                [3*self.vertices.shape[0], 3*self.vertices.shape[0]])
            self.stiffness_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.stiffness_matrix_


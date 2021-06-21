import numpy as np
from sparselinear import SparseLinear
from torch import nn

from .base import BaseVectorField


class SparseVectorField(BaseVectorField):
    """Add sparsity to the tri-linear map.

    This is analogous to the Neural CDE vector field but the Linear HH -> H * I layer is replaced with a SparseLinear
    implementation.
    """

    def additional_network_initialisation(self):
        assert self.sparsity is not None, "sparse methods must have a sparsity!"
        assert (
            self.vector_field_type == "matmul"
        ), "Sparse method only work for the matmul vector field type."
        self.sparse_output = nn.Sequential(
            SparseLinear(
                self.hidden_hidden_dim,
                self.output_dim,
                sparsity=self.sparsity,
            ),
            nn.Tanh(),
        )

    def _forward(self, h):
        return self.sparse_output(self.net_to_hh(h))


class LowRankVectorField(BaseVectorField):
    """Low rank approximation to the traditional vector field.

    Rather than mapping HH -> H * I we instead map HH onto two matrices, one of shape H * R and the other R * I where
    R defines the rank. The output is given by the matmul of these two matrices.
    """

    def additional_network_initialisation(self):
        assert self.sparsity is not None, "sparse methods must have a sparsity!"
        assert (
            self.vector_field_type == "matmul"
        ), "Sparse method only work for the matmul vector field type."
        # Define the rank in terms of sparsity
        self.rank = int(np.ceil(self.input_dim * (1 - self.sparsity)))
        self.M_h = nn.Linear(self.hidden_hidden_dim, self.hidden_dim * self.rank)
        self.M_o = nn.Linear(self.hidden_hidden_dim, self.input_dim * self.rank)
        self.tanh = nn.Tanh()

    def _forward(self, h):
        inner = self.net_to_hh(h)
        M_h = self.M_h(inner).reshape(-1, self.hidden_dim, self.rank)
        M_o = self.M_o(inner).reshape(-1, self.rank, self.input_dim)
        return self.tanh(M_h @ M_o)

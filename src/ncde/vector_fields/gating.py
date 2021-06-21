""" Vector fields for the Neural CDE model. """
from torch import nn

from .base import BaseVectorField


class MinimalGatedVectorField(BaseVectorField):
    """Neural CDE vector field with minimal gating.

    Suppose we have an NCDE inner net that maps:
        INNER = H -> HH -> ... -> HH
    define
        Z = Sigmoid(Linear_z(INNER(HIDDEN)));   where Linear_z maps HH -> HH * I,
        R = Tanh(Linear_r_i(INNER(HIDDEN)));   where Linear_r_i maps HH -> HH * I
    then
        OUTPUT = Z(INNER(H)) * R(INNER(H))
    """

    def additional_network_initialisation(self):
        assert self.sparsity is None, "sparsity not implemented for gated methods"
        self.sigmoid_net = nn.Sequential(
            nn.Linear(self.hidden_hidden_dim, self.output_dim), nn.Sigmoid()
        )
        self.tanh_net = nn.Sequential(
            nn.Linear(self.hidden_hidden_dim, self.output_dim), nn.Tanh()
        )

    def _forward(self, h):
        hh = self.net_to_hh(h)
        return self.sigmoid_net(hh) * self.tanh_net(hh)


class GRUGatedVectorField(BaseVectorField):
    """Neural CDE vector field with GRU style gating.

    Suppose we have an NCDE inner net that maps:
        INNER = H -> HH -> ... -> HH
    we define forget, sigmoid, and reset nets
        F = Sigmoid(Linear_r(H));   where Linear_r maps H -> H
        Z = Sigmoid(Linear_z(INNER(HIDDEN)));   where Linear_z maps HH -> HH * I,
        R = Tanh(Linear_r_i(INNER(HIDDEN)));   where Linear_r_i maps HH -> HH * I
    then
        OUTPUT = Z(INNER(H)) * R(INNER(F(H) * H))
    """

    def additional_network_initialisation(self):
        assert self.sparsity is None, "sparsity not implemented for gated methods"
        self.reset_net = nn.Sequential(
            nn.Linear(self.initial_dim, self.initial_dim), nn.Sigmoid()
        )
        self.sigmoid_net = nn.Sequential(
            nn.Linear(self.hidden_hidden_dim, self.output_dim), nn.Sigmoid()
        )
        self.tanh_net = nn.Sequential(
            nn.Linear(self.hidden_hidden_dim, self.output_dim), nn.Tanh()
        )

    def _forward(self, h):
        inner = self.net_to_hh(h)
        reset = self.net_to_hh(self.reset_net(h) * h)
        return self.sigmoid_net(inner) * self.tanh_net(reset)

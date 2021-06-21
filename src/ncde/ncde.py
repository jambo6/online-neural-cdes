""" Simple subclassing of NeuralCDE overwriting the vector field. """
import abc

import torch
import torchcde
from torch import nn

from .interpolation import SmoothLinearInterpolation
from .vector_fields import base, gating
# from .vector_fields import sparsity as sparsity_

SPLINES = {
    "cubic": torchcde.NaturalCubicSpline,
    "linear": torchcde.LinearInterpolation,
    "rectilinear": torchcde.LinearInterpolation,
    "linear_cubic_smoothing": lambda x, eps: SmoothLinearInterpolation(
        x, gradient_matching_eps=eps, match_second_derivatives=False
    ),
    "linear_quintic_smoothing": lambda x, eps: SmoothLinearInterpolation(
        x, gradient_matching_eps=eps, match_second_derivatives=True
    ),
    "rectilinear_cubic_smoothing": NotImplemented,
}

VECTOR_FIELDS = {
    "original": base.OriginalVectorField,
    # "sparse": sparsity_.SparseVectorField,
    # "low-rank": sparsity_.LowRankVectorField,
    "gru": gating.GRUGatedVectorField,
    "minimal": gating.MinimalGatedVectorField,
}


class NeuralCDE(nn.Module, abc.ABC):
    """Meta class for Neural CDE modelling.

    Attributes:
        nfe (int): Number of function evaluations, this is inherited from the vector field if the vector field has this
            property.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=None,
        hidden_hidden_dim=15,
        num_layers=3,
        use_initial=True,
        interpolation="linear",
        interpolation_eps=None,
        sparsity=None,
        vector_field="original",
        vector_field_type="matmul",
        adjoint=True,
        solver="rk4",
        return_sequences=False,
        apply_final_linear=True,
        return_filtered_rectilinear=True,
    ):
        """
        Args:
            input_dim (int): The dimension of the path.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            static_dim (int): The dimension of any static values, these will be concatenated to the initial values and
                put through a network to build h0.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
                net with the given density. Hidden and hidden hidden dims must be multiples of 32.
            use_initial (bool): Set True to use the initial absolute values to generate h0.
            interpolation (str): Interpolation method from ('linear', 'cubic', 'rectilinear').
            interpolation_eps (float): Epsilon smoothing region if a linear_cubic/quintic_smoothing is selected.
            sparsity (float or None): Final matrix sparsity, only applies if the vector field can be sparse.
            vector_field (str): Any of ['original', 'sparse', 'low-rank', 'minimal', 'gru']
            vector_field_type (str): One of ('matmul', 'evaluate', 'derivative') determines whether the vector field
                will apply [f(h) dX/dt, f(h, X), f(h, dX/dt)].
            adjoint (bool): Set True to use odeint_adjoint.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
            apply_final_linear (bool): Set False for no final linear layer to be applied to the hidden state.
            return_filtered_rectilinear (bool): Set True to return every other output if the interpolation scheme chosen
                is rectilinear, this is because rectilinear doubles the input length. False will return the full output.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.static_dim = static_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.use_initial = use_initial
        self.interpolation = interpolation
        self.interpolation_eps = interpolation_eps
        self.sparsity = sparsity
        self.vector_field = vector_field
        self.vector_field_type = vector_field_type
        self.adjoint = adjoint
        self.solver = solver
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear
        self.return_filtered_rectilinear = return_filtered_rectilinear

        # Set initial linear layer
        if self.initial_dim > 0:
            self.initial_linear = nn.Linear(self.initial_dim, self.hidden_dim)

        # Spline
        assert (
            self.interpolation in SPLINES.keys()
        ), "Unrecognised interpolation scheme {}".format(self.interpolation)
        if interpolation in ("linear_cubic_smoothing", "linear_quintic_smoothing"):
            match_second = True if "quintic" in interpolation else False
            self.spline = lambda coeffs: SmoothLinearInterpolation(
                coeffs,
                gradient_matching_eps=interpolation_eps,
                match_second_derivatives=match_second,
            )
        else:
            if interpolation_eps == 1:
                interpolation_eps = None
            assert interpolation_eps is None
            self.spline = SPLINES.get(self.interpolation)

        # Set options
        assert self.solver in ["rk4", "dopri5"]
        self.atol = 1e-5
        self.rtol = 1e-3
        self.cdeint_options = (
            {"step_size": 1} if self.solver == "rk4" else {"min_step": 0.5}
        )

        # The net that is applied to h_{t-1}
        self.func = VECTOR_FIELDS[self.vector_field](
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            hidden_hidden_dim=self.hidden_hidden_dim,
            num_layers=self.num_layers,
            sparsity=self.sparsity,
            vector_field_type=vector_field_type,
        )

        # Linear classifier to apply to final layer
        self.final_linear = (
            nn.Linear(self.hidden_dim, self.output_dim)
            if apply_final_linear
            else lambda x: x
        )

    @property
    def initial_dim(self):
        # Setup initial dim dependent on `use_initial` and `static_dim` options
        initial_dim = 0
        if self.use_initial:
            initial_dim += self.input_dim
        if self.static_dim is not None:
            initial_dim += self.static_dim
        return initial_dim

    @property
    def nfe(self):
        nfe_ = None
        if hasattr(self.func, "nfe"):
            nfe_ = self.func.nfe
        return nfe_

    def _setup_h0(self, inputs):
        """Sets up the initial value of the hidden state.

        The hidden state depends on the options `use_initial` and `static_dim`. If either of these are specified the
        hidden state will be generated via a network applied to either a concatenation of the initial and static data,
        or a network applied to just initial/static depending on options. If neither are specified then a zero initial
        hidden state is used.
        """
        if not self.static_dim:
            spline = self.spline(inputs)
            if self.use_initial:
                h0 = self.initial_linear(spline.evaluate(0))
            else:
                h0 = torch.autograd.Variable(
                    torch.zeros(inputs.size(0), self.hidden_dim)
                ).to(inputs.device)
        else:
            assert (
                len(inputs) == 2
            ), "Inputs must be a 2-tuple of (static_data, temporal_data)"
            static, spline = inputs[0], self.spline(inputs[1])
            if self.use_initial:
                h0 = self.initial_linear(
                    torch.cat((static, spline.evaluate(0)), dim=-1)
                )
            else:
                h0 = self.initial_linear(static)

        return spline, h0

    def _make_outputs(self, hidden):
        """Hidden state to output format depending on `return_sequences` and rectilinear (return every other)."""
        if self.return_sequences:
            outputs = self.final_linear(hidden)

            # If rectilinear and return sequences, return every other value
            if (
                self.interpolation == "rectilinear"
            ) and self.return_filtered_rectilinear:
                outputs = outputs[:, ::2]
        else:
            outputs = self.final_linear(hidden[:, -1, :])
        return outputs

    def forward(self, inputs):
        # Handle h0 and inputs
        spline, h0 = self._setup_h0(inputs)

        # Only return sequences with a fixed grid solver
        if self.return_sequences:
            # assert (
            #     self.solver == "rk4"
            # ), "return_sequences is only allowed with a fixed grid solver (for now)"
            times = spline.grid_points
        else:
            times = spline.interval

        hidden = torchcde.cdeint(
            spline,
            self.func,
            h0,
            t=times,
            adjoint=self.adjoint,
            vector_field_type=self.vector_field_type,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
            options=self.cdeint_options,
        )

        # Convert to outputs
        outputs = self._make_outputs(hidden)

        return outputs

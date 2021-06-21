""" Implementation of a stacked Neural CDE. """
from torch import nn

from .ncde import NeuralCDE


class StackedNeuralCDE(nn.Module):
    """A stacked Neural CDE model.

    This is simply a chaining of n Neural CDEs:
        -> dZ_1 = f_1(Z_1) dX
        -> dZ_2 = f_2(Z_2) dZ_1
        ...
        -> dZ_n = f_2(Z_n) dZ_{n-1}
        -> Y = L(Z_n)

    Attributes:
        num_stacked (int): Total number of stacked Neural CDEs, will equal len(hidden_dims).
        ncdes (ModuleList): A list of the stacked Neural CDE modules.

    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        hidden_hidden_dim=15,
        static_dim=None,
        adjoint=True,
        return_sequences=False,
        static_in_all_layers=False,
    ):
        """
        Args:
            input_dim (int): The input dimension.
            hidden_dims (list of ints): A list of hidden sizes. The length determines the number of stacked Neural CDEs.
            output_dim (int): Dimension of the output.
            hidden_hidden_dim (int): hidden hidden size, currently only accepts int and we force the same size for all
                Neural CDEs.
            static_dim (int or None): Dimension of the static data, or None for no static inputs.
            adjoint (bool): Set True to use the adjoint method.
            return_sequences (bool): Set True to return values for all times.
            static_in_all_layers (bool): Set True to have static network fed to all layers, otherwise only in the first
                layer.
        """
        assert isinstance(
            hidden_dims, list
        ), "hidden_dims must be a list, got type {}".format(type(hidden_dims))
        super(StackedNeuralCDE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.static_dim = static_dim
        self.adjoint = adjoint
        self.return_sequences = return_sequences
        self.static_in_all_layers = static_in_all_layers

        # Some attributes
        self.num_stacked = len(hidden_dims)

        # Setup Neural CDEs
        # final linear is applied only for the final Neural CDE
        input_ = input_dim
        static_ = static_dim
        final_linear = False
        output_ = output_dim
        return_sequences = True
        self.ncdes = nn.ModuleList()
        for i, hidden_ in enumerate(hidden_dims):
            # Final linear on last net
            if i == self.num_stacked - 1:
                final_linear = True
                return_sequences = True if self.return_sequences else False
            # Nets are called ncde_i with i indexing from 0
            ncde = self._create_ncde(
                input_, hidden_, output_, static_, final_linear, return_sequences
            )
            self.ncdes.append(ncde)
            # New input is hidden dim
            input_ = hidden_
            if not self.static_in_all_layers:
                static_ = None

        # Output
        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)

    @property
    def ncde_list(self):
        return [exec("self.ncde_{}".format(i)) for i in range(self.num_stacked)]

    def _create_ncde(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        static_dim,
        apply_final_linear,
        return_sequences,
    ):
        return NeuralCDE(
            input_dim,
            hidden_dim,
            output_dim,
            static_dim,
            use_initial=True,
            interpolation="linear",
            adjoint=self.adjoint,
            num_layers=3,
            apply_final_linear=apply_final_linear,
            return_sequences=return_sequences,
        )

    def _handle_hidden_static_features(self, x, hidden_state):
        # Return a tuple (static, hidden) if we have static features.
        if any([self.static_dim is None, not self.static_in_all_layers]):
            return hidden_state
        else:
            return [x[0], hidden_state]

    def forward(self, x):
        # Run on initial then simply loop over the remaining
        hidden_state = self.ncdes[0](x)
        for ncde in self.ncdes[1:]:
            inputs = self._handle_hidden_static_features(x, hidden_state)
            hidden_state = ncde(inputs)
        # Final hidden state is either final hidden or linear(final hidden)
        output = hidden_state
        return output

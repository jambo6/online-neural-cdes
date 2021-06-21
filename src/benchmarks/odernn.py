import torch
from torch import nn
from torch.nn import GRUCell, RNNCell
from torchdiffeq import odeint, odeint_adjoint


class ODERNN(nn.Module):
    """ Implementation of the ODE_RNN method as used in LatentODEs. """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_hidden_dim=30,
        num_layers=1,
        solver="rk4",
        adjoint=False,
        gru=True,
        return_sequences=False,
        apply_final_linear=True,
    ):
        """
        Args:
            input_dim (int): The dimension of the log-signature.
            hidden_dim (int): The dimension of the hidden state in the RNN.
            output_dim (int): The dimension of the output.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            hidden_hidden_dim (int): The dimension of the hidden dim in the ODE forward solve.
            adjoint (bool): Set True to use the adjoint method for O(1) memory.
            gru (bool): Set True for a GRUCell.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
            apply_final_linear (bool): Set True to apply a linear map to the output.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.solver = solver
        self.adjoint = adjoint
        self.gru = gru
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear

        # The ODE update i.e. the new bit
        self.ode_cell = _ODERNNFunc(hidden_dim, hidden_hidden_dim)

        # The net applied to h_prev
        cell = GRUCell if gru else RNNCell
        self.cell = cell(input_dim, hidden_dim)

        # Linear classifier to apply to final layer
        self.final_linear = (
            nn.Linear(self.hidden_dim, self.output_dim)
            if self.apply_final_linear
            else lambda x: x
        )

    def forward(self, inputs, times=None):
        # Params
        batch_size, length = inputs.size()[:2]

        if times is None:
            times = torch.arange(0, length).to(inputs.device)

        # For storing all hidden states
        h_i = torch.zeros(batch_size, self.hidden_dim).to(inputs.device)
        hidden_states = []

        # Get the odeint function
        ode_func = odeint_adjoint if self.adjoint else odeint

        # Loop over time to get the final hidden state
        dts = [torch.Tensor([0, t]).to(inputs.device) for t in times[1:] - times[:-1]]
        for i in range(length):
            # Solve ODE then update with data
            h_i = ode_func(
                func=self.ode_cell, y0=h_i, t=dts[i - 1], method=self.solver
            )[-1]
            h_i = self.cell(inputs[:, i], h_i)
            hidden_states.append(h_i)

        # Stack hidden states
        hidden_states = torch.stack(hidden_states, dim=1)

        # Outputs
        outputs = (
            self.final_linear(h_i)
            if not self.return_sequences
            else self.final_linear(hidden_states)
        )

        return outputs


class _ODERNNFunc(nn.Module):
    """ The function for the ode solve on the hidden state. """

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(_ODERNNFunc, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]
        if num_layers > 1:
            layers += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1)

        self.net = nn.Sequential(*layers, nn.Tanh(), nn.Linear(hidden_dim, input_dim))

    def forward(self, t, x):
        return self.net(x)

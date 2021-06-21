""" Simple subclassing of NeuralCDE overwriting the vector field. """
import torchcde
from autots.preprocessing import ForwardFill, PadRaggedTensors, SimplePipeline
from torch import nn

from .ncde import NeuralCDE
from .sparsemax import Sparsemax


class AttentionNeuralCDE(nn.Module):
    """A Neural CDE attention model.

    The mechanism differs from regular attention modelling in two ways:
        1. Hidden states are kept if its corresponding attention weight > 1 / length.
        2. A second recurrent model is run over the states that are kept to produce the output.

    The model is as follows:
        1. Define hidden encoding via dZ = f(Z) dX
        2. Define weights A for each hidden state - dA = f(A) dZ - where A is 1D and we run backwards over Z.
        3. Apply a sparsemax to A and return the hidden states that had value > 1 / length.
        4. Run a final model over the remaining hidden states and apply a linear map to the output.

    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=None,
        adjoint=True,
        run_backwards=True,
        sparsemax=False,
    ):
        super(AttentionNeuralCDE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.static_dim = static_dim
        self.adjoint = adjoint
        self.run_backwards = run_backwards

        # Hidden state
        self.encoder = self._create_ncde(input_dim, hidden_dim, hidden_dim, static_dim)

        # For extraction of most relevant series information
        activation = Sparsemax(dim=1) if sparsemax else nn.Softmax(dim=1)
        self.attention = nn.Sequential(
            self._create_flipper(),
            self._create_ncde(hidden_dim, hidden_dim, 1, static_dim),
            self._create_flipper(),
            activation,
        )

        # Now extract features within a time-point
        self.final = nn.Sequential(
            self._create_ncde(
                hidden_dim, hidden_dim, hidden_dim, static_dim, return_sequences=False
            ),
        )

        # Output
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def _create_flipper(self):
        # Creates a flip module if run backwards is set
        if self.run_backwards:
            item_index = None
            if self.static_dim:
                item_index = 1
            return FlipTensor(dim=-2, item_index=item_index)
        else:
            return FlipTensor(dim=None)

    def _create_ncde(
        self, input_dim, hidden_dim, output_dim, static_dim, return_sequences=True
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
            apply_final_linear=True,
            return_sequences=return_sequences,
            return_filtered_rectilinear=False,
        )

    def _handle_hidden_static_features(self, x, hidden_state):
        # Return a tuple (static, hidden) if we have static features.
        if self.static_dim is None:
            return hidden_state
        else:
            return [x[0], hidden_state]

    def reduce_hidden_state(self, x, hidden_state, attention_weights):
        """ Reduce the number of states considered using the sparsemax. """
        # States that have attention value > 1 / length
        keep_bools = attention_weights > 1 / hidden_state.size(1)
        hold_states = PadRaggedTensors().transform(
            [h[k.view(-1)] for h, k in zip(hidden_state, keep_bools)]
        )

        # Pad back into standard format
        pipeline = SimplePipeline([PadRaggedTensors(), ForwardFill()])
        reduced_hidden_state = pipeline.transform(hold_states)

        # Spline it
        output = torchcde.linear_interpolation_coeffs(reduced_hidden_state)

        # Add static back on if specified
        output = self._handle_hidden_static_features(x, output)

        return output

    def forward(self, x):
        # Create the hidden embedding
        hidden_state = self.encoder(x)

        # If static dim is not none, concat to the hidden state
        attention_inputs = self._handle_hidden_static_features(x, hidden_state)

        # Get the time locations of important information
        attention_weights = self.attention(attention_inputs)

        # Only keep if attention weights are > 1 / length
        reduced_hidden_state = self.reduce_hidden_state(
            x, hidden_state, attention_weights
        )

        # Important features at said times
        final_ncde_out = self.final(reduced_hidden_state)

        #  Apply a net to the outputs
        output = self.fc_output(final_ncde_out)

        return output


class FlipTensor(nn.Module):
    """ Flip a tensor along a specified axis. """

    def __init__(self, dim=-2, item_index=None):
        super(FlipTensor, self).__init__()
        self.dim = dim
        self.item_index = item_index

    def flip(self, x):
        if self.dim is not None:
            if self.item_index is None:
                return x.flip(dims=[self.dim])
            else:
                x[self.item_index] = x[self.item_index].flip(dims=[self.dim])
                return x
        else:
            return x

    def forward(self, x):
        return self.flip(x)

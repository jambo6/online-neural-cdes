""" Base vector field class for the Neural CDE model. """
import abc

from torch import nn


class BaseVectorField(nn.Module, abc.ABC):
    """Base class for Neural CDE vector fields.

    Useful for building vector fields for Neural CDEs that perform slight variations upon the traditional trilinear map
    required to take HH -> H * I.

    This class initialises the map from H -> HH that is used in the standard Neural CDE, but allows for a different
    definition of the network to be defined from HH -> H * I. This behaviour is implemented by:
        1. The inner network to define the hidden hidden state is common to all models and denoted `net_to_hh`. This
        should be utilised in the overwritten _forward method for the given vector field.
        2. Overwrite the `additional_network_initialisation` method which builds additional network components required
        for computation of the vector field.
        3. Overwrite the _forward method to produce a vector of dimension H * I, all share a forward method which
        simply reshapes this output onto a 2d matrix [H, I] that is applied to dX which has shape [I, 1].

    For example, for the original Neural CDE formulation, initialise_network would simply define a linear map
    HH -> H * I, and the forward method would apply it to the output of net_to_hh.

    Attributes:
        nfe (int): Increments by 1 from zero each time the function is called
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_hidden_dim=15,
        num_layers=1,
        sparsity=None,
        vector_field_type="matmul",
    ):
        """
        Args:
            input_dim (int): The Neural CDE input dim, the dimension of the underlying path control.
            hidden_dim (int): The Neural CDE hidden dim, this is the anticipated output dim for the vector field.
            hidden_hidden_dim (int): The Neural CDE hidden hidden dim.
            num_layers (int): Number of internal hidden layers.
            sparsity (float or None): Amount of sparsity.
            vector_field_type (str): One of ('matmul', 'evaluate', 'derivative'), if matmul the network is from hidden
                to hidden hidden else from (hidden + input) to hidden hidden.
        """
        super(BaseVectorField, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.sparsity = sparsity
        self.vector_field_type = vector_field_type

        self.matmul = True if vector_field_type == "matmul" else False
        self.initial_dim = hidden_dim if self.matmul else hidden_dim + input_dim
        self.output_dim = (
            self.hidden_dim * self.input_dim if self.matmul else self.hidden_dim
        )
        self.nfe = 0

        # Build the net that maps H -> HH that is shared by all fields
        layers = [nn.Linear(self.initial_dim, hidden_hidden_dim), nn.ReLU()]
        if num_layers > 1:
            layers += [nn.Linear(hidden_hidden_dim, hidden_hidden_dim), nn.ReLU()] * (
                num_layers - 1
            )
        self.net_to_hh = nn.Sequential(*layers)

        # Additional network pieces
        self.additional_network_initialisation()

    @abc.abstractmethod
    def additional_network_initialisation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, h):
        # Forward method to overwrite
        raise NotImplementedError

    def forward(self, t, h):
        # Perform the required reshape if matmul method
        out = self._forward(h)
        if self.vector_field_type == "matmul":
            out = out.view(-1, self.hidden_dim, self.input_dim)

        # Update nfe
        self.nfe += 1

        return out


class OriginalVectorField(BaseVectorField):
    """ The standard Neural CDE vector field. """

    def additional_network_initialisation(self):
        self.tanh_output_layer = nn.Sequential(
            nn.Linear(self.hidden_hidden_dim, self.output_dim), nn.Tanh()
        )

    def _forward(self, h):
        return self.tanh_output_layer(self.net_to_hh(h))

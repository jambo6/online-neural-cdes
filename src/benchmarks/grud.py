"""
gru-d.py
===================================
GRU-D model taken and adapted from: https://github.com/zhiyongc/GRU-D
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from autots.preprocessing import ForwardFill
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class GRUD(nn.Module):
    """Implementation of the GRU-D model from:
        "Recurrent Neural Networks for Multivariate Times Series with Missing Values"

    Code adapted from: https://github.com/zhiyongc/GRU-D
    """

    def __init__(
        self, feature_means, input_dim, hidden_dim, output_dim, return_sequences=False
    ):
        super(GRUD, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        # Training data means
        # If no mean value exists, replace with 0
        feature_means[torch.isnan(feature_means)] = 0.0
        self.feature_means = feature_means

        self.delta_size = input_dim
        self.mask_size = input_dim

        # Hidden state dynamics
        self.cell = GRUDCell(input_dim, hidden_dim)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        """Performs the GRU-D forward pass.

        Args:
            input: Must be a tensor of shape [N, D, L, C] where D denotes the additional dimensions due to the GRU-D
                model. The D dimensions in order are (data, last data observation, mask, delta).

        Returns:
            torch.Tensor: A tensor of outputs of shape [N,] for classification or [N, L] for regression.
        """
        batch_size = input.size(0)
        length = input.size(2)

        # Get the different components of the data
        data, data_last_obs, mask, delta = [input[:, i, :, :] for i in range(4)]
        self.feature_means = self.feature_means.to(data.device).reshape(1, -1)

        # For storing all hidden states
        h_i = torch.zeros(batch_size, self.hidden_dim).to(data.device)
        hidden_states = []

        # GRU
        for i in range(length):
            h_i = self.cell(
                data[:, i, :],
                data_last_obs[:, i, :],
                self.feature_means.repeat(batch_size, 1),
                h_i,
                mask[:, i, :],
                delta[:, i, :],
            )
            hidden_states.append(h_i)
        hidden_states = torch.stack(hidden_states).transpose(0, 1)

        outputs = (
            self.final_linear(h_i)
            if not self.return_sequences
            else self.final_linear(hidden_states)
        )

        return outputs


class GRUDCell(nn.Module):
    """ This implements the hidden state dynamics for the GRU-D model. """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.mask_dim = input_dim
        self.hidden_dim = hidden_dim
        self.delta_size = input_dim

        # Needed for the Filter linear class
        self.identity = Variable(torch.eye(input_dim).cuda())

        # Needed for exp
        self.zeros_x = Variable(torch.zeros(input_dim).cuda())
        self.zeros_h = Variable(torch.zeros(hidden_dim).cuda())

        # Standard GRU weights
        self.zl = nn.Linear(input_dim + hidden_dim + self.mask_dim, hidden_dim)
        self.rl = nn.Linear(input_dim + hidden_dim + self.mask_dim, hidden_dim)
        self.hl = nn.Linear(input_dim + hidden_dim + self.mask_dim, hidden_dim)

        # Decay parameters
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_dim)

    def forward(self, x, x_last_obsv, x_mean, h, mask, delta):
        self.zeros_x = self.zeros_x.to(x.device)
        self.zeros_h = self.zeros_h.to(x.device)

        # The decay factors
        delta_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta)))

        # Impute missing values using decay and training mean.
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)

        # Hidden state decay
        h = delta_h * h

        # Concatenate
        combined = torch.cat((x, h, mask), 1)

        # GRU
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde

        return h


class FilterLinear(nn.Module):
    # TODO doc this up
    def __init__(self, input_dim, output_dim, filter_square_matrix, bias=True):
        """
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        self.filter_square_matrix = self.filter_square_matrix.to(input.device)
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


def _evaluate_time_since_last_observation(controls):
    """Given data of shape [N, L, C] with nan values, computes the time since last observation at each point.

    For input of shape [N, L, C] where the first index of the C-dimension is assumed to be time, this function first
    creates a [0, 1] nan-masking tensor over the data and then computes for each point in the data, the time since the
    last observation was made. This results in a `delta` tensor that denotes the time since last observation, and a
    `mask` tensor that notes whether an observation was made at that point.

    Args:
        controls (torch.Tensor): A tensor of shape [N, L, C].

    Returns:
        (torch.Tensor, torch.Tensor): The delta and masking tensor of the data.
    """
    # Positions where data is given
    mask = (~torch.isnan(controls)).float()

    # Times for each channel
    times = controls[:, :, [0]].repeat(1, 1, controls.shape[2])
    dts = times[:, 1:] - times[:, :-1]
    deltas = torch.zeros_like(times)
    for i in range(1, deltas.size(1)):
        deltas[:, i] += dts[:, i - 1] + deltas[:, i - 1] * (1 - mask[:, i - 1])

    return deltas, mask


def prepare_gru_variant_data(controls, variant, feature_means=None):
    """Given a tensor of controls with nan values, this will prepare the data for the GRU-d model.

    Input to the GRU-D model must be of shape [N, D, L, C] where D represents the additional dimension to store the
    GRU-D tensors. In the D dimension the information is the actual data (with nans), the last observed value of the
    data, a mask vector denoting whether a value was seen at that point, and a delta denoting how long it has been since
    an entry was last observed. Given times and controls, all these things are computed and put into a tensor.

    Args:
        controls (torch.Tensor): The input stream data of shape [N, L, C]. Must have times as first index.
        variant (str): One of ['gru-d', 'gru', 'gru-intensity', 'gru-dt', 'gru-dt-intensity'].
        intensity (bool): Only active for gru-dt, includes the intensity along with dt.
        feature_means (tensor): If specified will fill initial nan values with the feature means

    Returns:
        torch.Tensor: Shape [N, 4, L, C] where the additional dimension is described above.
    """
    assert variant in [
        "gru",
        "gru-intensity",
        "gru-dt",
        "gru-dt-intensity",
        "gru-d",
    ], "variant={} not implemented".format(variant)

    # Mask that denotes when the data exceeds final time
    end_mask = (
        torch.isnan(controls[..., 0]).repeat(controls.shape[-1], 1, 1).T.transpose(0, 1)
    )

    # Get the dt and mask tensors
    delta, mask = _evaluate_time_since_last_observation(controls)

    # prev_value is just the filled control
    prev_value = ForwardFill().transform(controls)

    # We still need to forward fill the controls (other fills can be applied here, but we use forward)
    controls = ForwardFill().transform(controls)

    # Fill if there are initial nans
    if feature_means is not None:
        feature_means = feature_means.repeat(delta.shape[0], delta.shape[1], 1)
        for tensor_data in (controls, prev_value, delta):
            nan_mask = torch.isnan(tensor_data)
            tensor_data[nan_mask] = feature_means[nan_mask]
            # Finally make everything zero after final time
            tensor_data[end_mask] = 0.0

    # Now concat into the correct type of tensor
    new_controls = torch.stack((controls, prev_value, mask, delta)).transpose(0, 1)

    # Now get a subset of the data if model type is not grud
    if ("gru" in variant) and (variant != "gru-d"):
        if variant == "gru":
            new_controls = new_controls[:, [0], :, :]
        elif variant == "gru-intensity":
            new_controls = new_controls[:, [0, -2], :, :]
        elif variant == "gru-dt":
            new_controls = new_controls[:, [0, -1], :, :]
        elif variant == "gru-dt-intensity":
            new_controls = new_controls[:, [0, -2, -1], :, :]
        else:
            raise NotImplementedError("Not implemented for {}".format(variant))

        # Open additional info into channels.
        new_controls = torch.cat(torch.unbind(new_controls, 1), -1)
        new_controls[torch.isnan(new_controls)] = 0.0

    return new_controls

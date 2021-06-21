""" Additional preprocessing transformers or modifications of transformers from autots. """
import torch
from sklearn.base import TransformerMixin
from torchcde import linear_interpolation_coeffs, natural_cubic_coeffs


class Interpolation(TransformerMixin):
    """ Linear, rectilinear, cubic, hybrid schemes. """

    def __init__(
        self,
        method="linear",
        channel_indices=None,
        initial_nan_to_zero=True,
        return_as_list=True,
    ):
        """
        Args:
            method (str): One of ("linear", "rectilinear", "cubic", "hybrid").
            channel_indices (list): List of channel indices for the hybrid method.
            initial_nan_to_zero (bool): Set True to mark the initial nan values to be zero.
            return_as_list (bool): Set True to return the data as a list up to final time rather than a padded tensor.
        """
        assert method in [
            "linear",
            "rectilinear",
            "cubic",
            "hybrid",
            "linear_forward_fill",
        ], "Got method {} which is not recognised".format(method)
        if method == "hybrid":
            assert (
                channel_indices is not None
            ), "Hybrid requires specification of the hybrid indices."
            raise NotImplementedError

        self.method = method
        self.channel_indices = channel_indices
        self.initial_nan_to_zero = initial_nan_to_zero
        self.return_as_list = return_as_list

        # Linear interpolation function requires the channel index of times
        self._rectilinear = 0 if self.method == "rectilinear" else None

    def __repr__(self):
        return "{} Interpolation".format(self.method.title())

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Causality
        if self.initial_nan_to_zero:
            for d in data:
                d[:1, :][torch.isnan(d[:1, :])] = 0.0

        # Build the coeffs
        if self.method == "cubic":

            def func(data):
                return natural_cubic_coeffs(data)

        else:

            def func(data):
                return linear_interpolation_coeffs(data, rectilinear=self._rectilinear)

        # Apply
        if isinstance(data, torch.Tensor):
            coeffs = func(data)
        else:
            coeffs = []
            for d in data:
                coeffs.append(func(d))

        return coeffs

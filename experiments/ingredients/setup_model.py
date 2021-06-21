# flake8: noqa
import numpy as np
import torch
from autots.models.rnn import RNN
from autots.preprocessing import ForwardFill
from ignite.utils import convert_tensor
from sacred import Ingredient

from src.benchmarks.grud import GRUD, prepare_gru_variant_data
from src.benchmarks.odernn import ODERNN
from src.ncde import NeuralCDE

model_ingredient = Ingredient("model")


@model_ingredient.config
def model_config():
    model_string = "ncde"
    static_dim = None
    hidden_dim = 15
    hidden_hidden_dim = 15
    vector_field = "original"
    vector_field_type = "matmul"
    sparsity = None
    solver = "rk4"
    adjoint = False
    interpolation_eps = None
    num_layers = 3
    return_sequences = False


@model_ingredient.capture
def setup_model(
    input_dim,
    output_dim,
    static_dim,
    interpolation,
    model_string=None,
    hidden_dim=None,
    hidden_hidden_dim=None,
    vector_field=None,
    vector_field_type=None,
    sparsity=None,
    solver=None,
    adjoint=None,
    interpolation_eps=None,
    num_layers=None,
    return_sequences=None,
    train_dl=None,  # Needed to compute GRU-D means
):
    # Needed for initial fills
    feature_means = torch.tensor(
        np.nanmean(torch.cat(train_dl.dataset.temporal_data, axis=0), axis=0)
    )

    assert model_string in [
        "ncde",
        "gru-dt",
        "gru-dt-intensity",
        "gru",
        "gru-intensity",
        "odernn",
        "gru-d",
    ]
    data_preparation_string = model_string

    if model_string == "ncde":
        model = NeuralCDE(
            input_dim,
            hidden_dim,
            output_dim,
            hidden_hidden_dim=hidden_hidden_dim,
            solver=solver,
            vector_field=vector_field,
            vector_field_type=vector_field_type,
            sparsity=sparsity,
            static_dim=static_dim,
            num_layers=num_layers,
            use_initial=True,
            adjoint=adjoint,
            interpolation=interpolation,
            interpolation_eps=interpolation_eps,
            return_sequences=return_sequences,
        )
    elif model_string == "gru-d":
        model = GRUD(
            feature_means,
            input_dim,
            hidden_dim,
            output_dim,
            return_sequences=return_sequences,
        )
    elif "gru" in model_string:
        if model_string in ["gru-dt", "gru-intensity"]:
            input_dim = 2 * input_dim
        elif model_string == "gru-dt-intensity":
            input_dim = 3 * input_dim
        model = RNN(
            input_dim,
            hidden_dim,
            output_dim,
            static_dim=None,
            num_layers=num_layers,
            model_string="gru",
            return_sequences=return_sequences,
        )
    elif "odernn" in model_string:
        model = ODERNN(
            input_dim * 2,
            hidden_dim,
            output_dim,
            hidden_hidden_dim,
            num_layers=num_layers,
            solver=solver,
            adjoint=adjoint,
            return_sequences=return_sequences,
        )
        # Change model string to gru-intensity so we get the same prepare batch
        data_preparation_string = "gru-intensity"
    else:
        raise NotImplementedError(
            "data_preparation_string {} not implemented".format(model_string)
        )

    def prepare_batch(batch, device, non_blocking=False):
        return _prepare_batch(
            data_preparation_string,
            batch,
            device,
            non_blocking,
            feature_means=feature_means,
        )

    return model, prepare_batch


def _prepare_batch(
    data_preparation_string, batch, device, non_blocking=False, feature_means=None
):
    """ Prepare batch for training: pass to a device with options. """
    x, y = batch
    if "gru" in data_preparation_string:
        x = prepare_gru_variant_data(
            x, data_preparation_string, feature_means=feature_means
        )
    elif data_preparation_string != "ncde":
        raise NotImplementedError("Not implemented {}".format(data_preparation_string))

    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )

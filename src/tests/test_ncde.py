import pytest
import torch
from autots.utils import make_time_series_problem

from src import ncde


@pytest.mark.parametrize(
    "vector_field, vector_field_type, sparsity",
    [
        ("original", "matmul", None),
        ("original", "evaluate", None),
        ("original", "derivative", None),
        ("sparse", "matmul", 0.99),
        ("low-rank", "matmul", 0.99),
        ("gru", "matmul", None),
        ("minimal", "matmul", None),
        ("gru", "derivative", None),
        ("minimal", "evaluate", None),
    ],
)
def test_all_vector_fields(vector_field, vector_field_type, sparsity):
    input_dim = 50
    static_dim = 5
    data, labels = make_time_series_problem(n_channels=input_dim, static_dim=static_dim)

    # Build and run model
    model = ncde.NeuralCDE(
        input_dim=input_dim,
        hidden_dim=50,
        static_dim=static_dim,
        output_dim=1,
        vector_field=vector_field,
        vector_field_type=vector_field_type,
        sparsity=sparsity,
    )
    from autots.models.utils import get_number_of_parameters

    print(vector_field, get_number_of_parameters(model))
    out = model(data)
    assert ~torch.any(torch.isnan(out))


@pytest.mark.parametrize("static_dim", [None, 5])
def test_attention(static_dim):
    input_dim = 50
    data, labels = make_time_series_problem(n_channels=input_dim, static_dim=static_dim)

    # Build and run
    model = ncde.AttentionNeuralCDE(
        input_dim=input_dim,
        hidden_dim=20,
        output_dim=1,
        static_dim=static_dim,
        run_backwards=True,
    )
    out = model(data)
    assert ~torch.any(torch.isnan(out))


@pytest.mark.parametrize(
    "hidden_dims, static_dim, static_in_all_layers",
    [
        ([10, 5, 20], False, False),
        ([10, 100, 10], True, True),
        ([10, 50, 30, 40], True, False),
    ],
)
def test_stacked(hidden_dims, static_dim, static_in_all_layers):
    input_dim = 50
    static_dim = 5
    data, labels = make_time_series_problem(n_channels=input_dim, static_dim=static_dim)

    # Build and run model
    model = ncde.StackedNeuralCDE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        static_dim=static_dim,
        output_dim=1,
        static_in_all_layers=static_in_all_layers,
    )
    out = model(data)
    assert ~torch.any(torch.isnan(out))

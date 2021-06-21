import torch

from src.ncde import interpolation


def test_reduced_rectilinear():
    # Explicit test
    nan = float("nan")
    times = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    fast_data = torch.tensor([3.0, 1.4, nan, 3.4, nan])
    sparse_1 = torch.tensor([nan, 1.5, nan, nan, nan])
    sparse_2 = torch.tensor([nan, nan, nan, nan, 1.2])
    sparse_3 = torch.tensor([nan, nan, nan, nan, nan])
    data = torch.stack([times, fast_data, sparse_1, sparse_2, sparse_3]).T.unsqueeze(0)
    interpolated_data = interpolation._prepare_linear_rectilinear_hybrid(
        data, rectilinear_indices=[2, 3, 4]
    )
    # Check only registers changes at times 1 and 4
    assert torch.equal(
        interpolated_data,
        torch.tensor(
            [
                [
                    [0.0000, 3.0000, 0.0000, 0.0000, 0.0000],
                    [1.0000, 1.4000, 0.0000, 0.0000, 0.0000],
                    [1.0000, 1.4000, 1.5000, 0.0000, 0.0000],
                    [2.0000, 2.4000, 1.5000, 0.0000, 0.0000],
                    [3.0000, 3.4000, 1.5000, 0.0000, 0.0000],
                    [4.0000, 3.4000, 1.5000, 0.0000, 0.0000],
                    [4.0000, 3.4000, 1.5000, 1.2000, 0.0000],
                ]
            ]
        ),
    )

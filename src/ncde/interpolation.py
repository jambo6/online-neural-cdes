import torch
from torchcde import interpolation_base, linear_interpolation_coeffs
from torchcde.misc import forward_fill


class SmoothLinearInterpolation(interpolation_base.InterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(
        self,
        coeffs,
        t=None,
        gradient_matching_eps=None,
        match_second_derivatives=False,
        **kwargs
    ):
        """
        Arguments:
            coeffs: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
            gradient_matching_eps: A float in (0, 1); if set, will match the gradients between each piecewise linear
                piece in the region (t, t + gradient_matching_eps) with a cubic spline.
            match_second_derivatives: A boolean active when gradient_matching_eps is set; True will use quintic splines
                which match the second derivative, False reverts to cubic matching first derivatives only.
        """
        super(SmoothLinearInterpolation, self).__init__(**kwargs)

        if t is None:
            t = torch.linspace(
                0,
                coeffs.size(-2) - 1,
                coeffs.size(-2),
                dtype=coeffs.dtype,
                device=coeffs.device,
            )
        else:
            assert (
                gradient_matching_eps is None
            ), "times not implemented for gradient_matching_eps"

        derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (
            t[1:] - t[:-1]
        ).unsqueeze(-1)

        # For gradient matching
        self.gradient_matching_eps = gradient_matching_eps
        self.match_second_derivatives = match_second_derivatives
        if self.gradient_matching_eps is not None:
            eps = gradient_matching_eps
            if match_second_derivatives:
                self.gradient_matching_coeffs = _setup_quintic_matching_coefficients(
                    coeffs, eps, device=coeffs.device
                )
            else:
                self.gradient_matching_coeffs = _setup_cubic_matching_coefficients(
                    coeffs, eps, device=coeffs.device
                )

        self.register_buffer("_t", t)
        self.register_buffer("_coeffs", coeffs)
        self.register_buffer("_derivs", derivs)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def __len__(self):
        return len(self.grid_points)

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._derivs.dtype, device=self._derivs.device)
        maxlen = self._derivs.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        # bool to denote whether we are in a matching region
        matching_region = False
        if self.gradient_matching_eps is not None:
            if all(
                [
                    0 < index,
                    index < len(self),
                    fractional_part < self.gradient_matching_eps,
                ]
            ):
                matching_region = True
        return fractional_part, index, matching_region

    def evaluate(self, t):
        fractional_part, index, matching_region = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        if matching_region:
            evaluation = _evaluate_matching_region(
                self.gradient_matching_coeffs[:, index - 1], fractional_part
            )
        else:
            prev_coeff = self._coeffs[..., index, :]
            next_coeff = self._coeffs[..., index + 1, :]
            prev_t = self._t[index]
            next_t = self._t[index + 1]
            diff_t = next_t - prev_t
            evaluation = prev_coeff + fractional_part * (
                next_coeff - prev_coeff
            ) / diff_t.unsqueeze(-1)
        return evaluation

    def derivative(self, t):
        fractional_part, index, matching_region = self._interpret_t(t)
        if matching_region:
            deriv = _evaluate_matching_region(
                self.gradient_matching_coeffs[:, index - 1],
                fractional_part,
                derivative=True,
            )
        else:
            deriv = self._derivs[..., index, :]
        return deriv


def _evaluate_matching_region(matching_coeffs, t, derivative=False):
    """ Evaluates the evaluation/derivative polynomials for the cubic or quintic matching regions. """
    device = matching_coeffs.device
    if derivative:
        t_powers = (
            torch.tensor([i * t ** (i - 1) for i in range(1, matching_coeffs.size(-1))])
            .flip(dims=[0])
            .to(device)
        )
        evaluation = (matching_coeffs[..., :-1] * t_powers).sum(dim=-1)
    else:
        t_powers = (
            torch.cat([t ** i for i in range(matching_coeffs.size(-1))])
            .flip(dims=[0])
            .to(device)
        )
        evaluation = (matching_coeffs * t_powers).sum(dim=-1)
    return evaluation


def _setup_cubic_matching_coefficients(coeffs, eps=0.1, device=None):
    """ Cubic spline coefficients to match linear gradients in a small matching region after each point. """
    assert 0 < eps <= 1
    x = coeffs[..., 1:-1, :]
    x_eps = x + eps * (coeffs[..., 2:, :] - x)
    delta_prev = coeffs[..., 1:-1, :] - coeffs[..., :-2, :]
    delta_next = coeffs[..., 2:, :] - coeffs[..., 1:-1, :]
    C = delta_prev
    D = x
    B = (1 / eps ** 2) * (3 * (x_eps - C * eps - D) - eps * (delta_next - C))
    A = (1 / (3 * eps ** 2)) * (delta_next - C - 2 * B * eps)
    matching_coeffs = torch.stack([A, B, C, D]).permute(1, 2, 3, 0)
    return matching_coeffs.to(device)


# def _setup_cubic_rectilinear_matching_coefficients(coeffs, eps=None):
#     """ Matching coefficients for rectilinear interpolation. """
#     # First find the coefficients for the non time variables
#     x = coeffs[..., 1:-1, :]
#     x1 = coeffs[..., 2:, :]
#     D = x
#     C = torch.zeros_like(x)
#     B = 3 * (x1 - D)
#     A = - (2 * B) / 3
#     matching_coeffs = torch.stack([A, B, C, D]).permute(1, 2, 3, 0)
#     return matching_coeffs


def _setup_quintic_matching_coefficients(coeffs, eps=0.1, device=None):
    """ Quintic spline coefficients to match linear gradients and second derivatives in the matching region. """
    assert 0 < eps <= 1
    x = coeffs[..., 1:-1, :]
    x_eps = x + eps * (coeffs[..., 2:, :] - x)
    delta_prev = coeffs[..., 1:-1, :] - coeffs[..., :-2, :]
    delta_next = coeffs[..., 2:, :] - coeffs[..., 1:-1, :]
    D = torch.zeros_like(x)
    E = delta_prev
    F = x
    C = (1 / eps ** 3) * (10 * (x_eps - E * eps - F) - 4 * eps * (delta_next - E))
    B = (1 / (2 * eps ** 3)) * (2 * (delta_next - E) - 3 * C * eps ** 2)
    A = -(1 / (10 * eps ** 2)) * (6 * B * eps + 3 * C)
    matching_coeffs = torch.stack([A, B, C, D, E, F]).permute(1, 2, 3, 0)
    return matching_coeffs.to(device)


def _prepare_linear_rectilinear_hybrid(data, rectilinear_indices, time_index=0):
    """A linear rectilinear hybrid scheme.

    This method performs linear interpolation on some channels, and rectilinear interpolation on others. The rationale
    for this comes from considering ICU data, often you have regularly sampled data that it is reasonable to approximate
    via linear interpolation alongside sparsely sampled data that is is unreasonable to perform linear interpolation on.
    This method enables a linear approximation to regular data, and utilised a rectilinear interpolation for the sparse
    data **updating only when a measurement is changed**. This can result in a much lower overall length than standard
    rectilinear interpolation for sparse measurements.

    Args:
        data (tensor): The data tensor.
        rectilinear_indices (list): The index locations of the sparsely measured variables to use rectilinear
            interpolation on. It is assumed all other variables will use linear interpolation.
        time_index (int): The index of the time variable.

    Returns:

    """
    assert isinstance(rectilinear_indices, list)

    # First linearly interpolate the non rectilinear indices
    time_and_rect_indices = [time_index] + rectilinear_indices
    non_rect_indices = [
        x for x in range(data.size(-1)) if x not in time_and_rect_indices
    ]
    data[..., non_rect_indices] = linear_interpolation_coeffs(
        data[..., non_rect_indices], initial_value_if_nan=0.0
    )

    # First rectilinear everything
    full_rectilinear = linear_interpolation_coeffs(
        data, rectilinear=0, initial_value_if_nan=0.0
    )

    # Now shift the slowly varying indices so the change occurs between the time points
    # This is opposed to an instantaneous change at (t, t + eps)
    if len(non_rect_indices) > 0:
        final_value = full_rectilinear[..., -1:, non_rect_indices]
        full_rectilinear[..., non_rect_indices] = torch.cat(
            [full_rectilinear[..., 1:, non_rect_indices], final_value], -2
        )

    # Finally remove rows where no element changed
    # Note these will only be in the non time changing rows where we have allowed only the rectilinear indicies to
    # change
    deltas = (
        full_rectilinear[..., :-1, time_and_rect_indices]
        - full_rectilinear[..., 1:, time_and_rect_indices]
    )
    change_locs = (deltas != 0).sum(dim=-1) > 0
    change_locs = torch.cat(
        [torch.ones_like(change_locs[:, :1], dtype=torch.bool), change_locs], dim=-1
    )
    hybrid_list = [r[..., c, :] for r, c, in zip(full_rectilinear, change_locs)]

    # Pad and forward fill
    hybrid_data = torch.nn.utils.rnn.pad_sequence(
        hybrid_list, batch_first=True, padding_value=float("nan")
    )
    hybrid_data = forward_fill(hybrid_data)

    return hybrid_data


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    data = torch.tensor(
        [[i, (-1) ** i] for i in range(10)], dtype=torch.float
    ).unsqueeze(0)
    data = torch.tensor(
        [[i, torch.randn(1).item() ** 2 + 1] for i in range(10)], dtype=torch.float
    ).unsqueeze(0)
    # data = torch.cat([torch.arange(10).unsqueeze(-1), data], -1).unsqueeze(0).to(torch.float)
    coeffs = linear_interpolation_coeffs(data, rectilinear=0)

    eps = 0.1
    interpolation = SmoothLinearInterpolation(
        coeffs, gradient_matching_eps=1, match_second_derivatives=False
    )
    # eval_times = torch.cat([torch.linspace(i + eps, i + 1) for i in range(5)])
    # plt.plot(eval_times, [interpolation.evaluate(t)[..., 1] for t in eval_times])
    # eval_times = torch.cat([torch.linspace(i, i + eps) for i in range(1, 5)])
    # plt.plot(eval_times, [interpolation.evaluate(t)[..., 1] for t in eval_times])
    # plt.show()

    eval_times = torch.linspace(0, 9)
    # plt.scatter(eval_times, [interpolation.evaluate(t)[..., 1] for t in eval_times])
    plt.plot(
        eval_times,
        [interpolation.evaluate(t)[..., 0] for t in eval_times],
        label="time",
    )
    plt.plot(
        eval_times,
        [interpolation.evaluate(t)[..., 1] for t in eval_times],
        label="variable",
    )
    # plt.scatter(eval_times, [interpolation.evaluate(t)[..., 0] for t in eval_times])
    plt.legend()
    plt.show()

import numpy as np
import torch
from torch import nn
from pfrlx.functions.lower_triangular_matrix import lower_triangular_matrix


class GaussianHeadWithStateIndependentCovariance(nn.Module):
    """Gaussian head with state-independent learned covariance.

    This link is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. The only learnable parameter this link has
    determines the variance in a state-independent way.

    State-independent parameterization of the variance of a Gaussian policy
    is often used with PPO and TRPO, e.g., in https://arxiv.org/abs/1709.06560.

    Args:
        action_size (int): Number of dimensions of the action space.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(
        self,
        action_size,
        var_type="spherical",
        var_func=nn.functional.softplus,
        var_param_init=0,
    ):
        super().__init__()

        self.var_func = var_func
        var_size = {"spherical": 1, "diagonal": action_size}[var_type]

        self.var_param = nn.Parameter(
            torch.tensor(np.broadcast_to(var_param_init, var_size), dtype=torch.float,)
        )

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor or ndarray): Mean of Gaussian.

        Returns:
            torch.distributions.Distribution: Gaussian whose mean is the
                mean argument and whose variance is computed from the parameter
                of this link.
        """
        var = self.var_func(self.var_param)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )


class GaussianHeadWithDiagonalCovariance(nn.Module):
    """Gaussian head with diagonal covariance.

    This module is intended to be attached to a neural network that outputs
    a vector that is twice the size of an action vector. The vector is split
    and interpreted as the mean and diagonal covariance of a Gaussian policy.

    Args:
        var_func (callable): Callable that computes the variance
            from the second input. It should always return positive values.
    """

    def __init__(self, var_func=nn.functional.softplus, beta=1.0, init_scale=0.3, min_scale=1e-6):
        super().__init__()
        self.var_func = var_func
        self.beta = beta
        self.init_scale = init_scale
        self._output_scale = 0.6931  # nn.functional.softplus(0.)
        self.min_scale = min_scale

    def forward(self, mean_and_var):
        """Return a Gaussian with given mean and diagonal covariance.

        Args:
            mean_and_var (torch.Tensor): Vector that is twice the size of an
                action vector.

        Returns:
            torch.distributions.Distribution: Gaussian distribution with given
                mean and diagonal covariance.
        """
        assert mean_and_var.ndim == 2
        mean, pre_var = mean_and_var.chunk(2, dim=1)
        if self.var_func == nn.functional.softplus:
            scale = self.var_func(pre_var, beta=self.beta) * self.init_scale / self._output_scale + self.min_scale
        else:
            scale = self.var_func(pre_var).sqrt()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=scale), 1
        )


class GaussianHeadWithFixedCovariance(nn.Module):
    """Gaussian head with fixed covariance.

    This module is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. Its covariance is fixed to a diagonal matrix
    with a given scale.

    Args:
        scale (float): Scale parameter.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor): Batch of mean vectors.

        Returns:
            torch.distributions.Distribution: Multivariate Gaussian whose mean
                is the mean argument and whose scale is fixed.
        """
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=self.scale), 1
        )

class MultivariateGaussianHeadWithFullCovariance(nn.Module):
    """
    Multivariate Gaussian head with full covariance matrix.

    Args:
        var_func (callable): Callable that computes the variance
            from the second input. It should always return positive values.
    """

    def __init__(
        self,
        action_size,
        hidden_size=100,
        init_scale=0.01,
        beta=2.0,
    ):
        super().__init__()

        self.var_func = nn.functional.softplus
        self.hidden_size = hidden_size
        self.non_diag_size = action_size * (action_size - 1) // 2
        self.mean_and_diag = nn.Linear(self.hidden_size, action_size * 2)
        self.non_diag = nn.Linear(self.hidden_size, self.non_diag_size)
        self.beta = beta  # scaling variance to sufficently small values

        # NOTE: original function
        self._init_weights(init_scale=init_scale)

    def _init_weights(self, init_scale=0.01):
        # init mean_and_diag
        torch.nn.init.uniform_(self.mean_and_diag.weight, a=-init_scale, b=init_scale)
        torch.nn.init.zeros_(self.mean_and_diag.bias)
        # init non_diag
        torch.nn.init.uniform_(self.non_diag.weight, a=-init_scale, b=init_scale)
        torch.nn.init.zeros_(self.non_diag.bias)

    def forward(self, hidden):
        """Return a Gaussian with given mean and Full covariance.

        Args:
            hidden (torch.Tensor): Vector

        Returns:
            torch.distributions.Distribution: Gaussian distribution with given
                mean and full covariance.
        """

        assert hidden.ndim == 2
        assert hidden.shape[1] == self.hidden_size

        mean_and_diag = self.mean_and_diag(hidden)
        non_diag = self.non_diag(hidden)

        mean, pre_diag = mean_and_diag.chunk(2, dim=1)
        diag = self.var_func(pre_diag, beta=self.beta)  # scaling

        tril = lower_triangular_matrix(diag, non_diag)

        return torch.distributions.MultivariateNormal(loc=mean, scale_tril=tril)

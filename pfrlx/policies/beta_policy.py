import torch
from torch import nn


class BetaHead(nn.Module):
    """Head module for a beta-distribution policy."""

    def __init__(self):
        super().__init__()
        self.activation = nn.functional.softplus

    def forward(self, alpha_and_beta):
        assert alpha_and_beta.ndim == 2
        alpha, beta = alpha_and_beta.chunk(2, dim=1)
        # TODO: check which one is stable
        print(alpha)
        print(beta)
        alpha = self.activation(alpha) + 1
        beta = self.activation(beta) + 1
        # alpha = torch.clamp(self.activation(alpha), min=1.0)
        # beta = torch.clamp(self.activation(beta), min=1.0)
        return torch.distributions.Independent(
            torch.distributions.Beta(concentration1=alpha, concentration0=beta), 1
            )

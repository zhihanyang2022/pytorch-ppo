import torch
import torch.nn as nn
from torch.distributions import Distribution, Beta, Independent, MultivariateNormal


class AddOne(nn.Module):

    def forward(self, x):
        return x + 1


class MLPBetaPolicy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.alphas_layer = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus(),
            AddOne()
        )
        self.betas_layer = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus(),
            AddOne()
        )

    def forward(self, states: torch.tensor) -> Distribution:
        out = self.shared_net(states)
        alphas, betas = self.alphas_layer(out), self.betas_layer(out)
        return Independent(Beta(concentration0=alphas, concentration1=betas), reinterpreted_batch_ndims=1)


class MLPGaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.means_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.fixed_var = torch.tensor([0.1])

    def forward(self, states: torch.tensor) -> Distribution:
        means = self.means_net(states)
        vars = self.fixed_var.expand_as(self.fixed_var)
        cov_matrix = torch.diag_embed(vars)
        return MultivariateNormal(means, cov_matrix)

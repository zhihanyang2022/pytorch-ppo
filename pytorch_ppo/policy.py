import torch
import torch.nn as nn
from torch.distributions import Distribution, Beta, Independent


class AddOne(nn.Module):

    def forward(self, x):
        return x + 1


class MLPBetaPolicy(nn.Module):

    def __init__(self, input_dim, action_dim):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
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

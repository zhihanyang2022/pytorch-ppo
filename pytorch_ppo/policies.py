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

    """
    MLPGaussianPolicy follows from https://github.com/DLR-RM/stable-baselines3/blob/6daf82bf7439675fa8595812ddaa1bc00473da26/stable_baselines3/common/policies.py#L377.
    - activation function: Tanh
    - network architecture: [64, 64]
    - log_std_init: 0 => std_init = e^(log_std_init) = e^(0) = 1
    """

    def __init__(self, state_dim, action_dim, log_std_init=1):

        super().__init__()

        # mean and log_std vector

        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # self.apply(self.init_weights)

        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)

    # @staticmethod
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.orthogonal_(m.weight, gain=0.01)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)

    def forward(self, states: torch.tensor) -> Distribution:
        means = self.mean_net(states)
        shared_cov_matrix = torch.diag_embed(torch.exp(self.log_std))
        return MultivariateNormal(means, shared_cov_matrix)

    def forward_determ(self, states: torch.tensor) -> torch.tensor:
        means = self.mean_net(states)
        return means

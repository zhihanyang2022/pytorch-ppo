from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, Beta, Independent, MultivariateNormal, Categorical

import gin

from utils import init_weights


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# we use these functions to set hyper-parameters that otherwise need to be set several times
# (since they are shared between policy and value function)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


@gin.configurable(module=__name__)
def make_backbone(num_in, net_arch=[64, 64], activation_str="tanh"):
    if activation_str == "tanh":
        activation_klass = torch.nn.Tanh
    elif activation_str == "relu":
        activation_klass = torch.nn.ReLU
    else:
        raise NotImplementedError
    temp = [num_in] + net_arch
    input_dims, output_dims = temp[:-1], temp[1:]
    layers = []
    for input_dim, output_dim in zip(input_dims, output_dims):
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(activation_klass())
    return nn.Sequential(*layers), temp[-1]


@gin.configurable(module=__name__)
def do_ortho_init(boolean=True):
    return boolean


K_BACKBONE_GAIN = np.sqrt(2)
K_ACTION_NET_GAIN = 0.01
K_VALUE_NET_GAIN = 1


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# below are policies and value functions
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


@gin.configurable(module=__name__)
class MLPGaussianPolicy(nn.Module):

    """
    MLPGaussianPolicy follows from https://github.com/DLR-RM/stable-baselines3/blob/6daf82bf7439675fa8595812ddaa1bc00473da26/stable_baselines3/common/policies.py#L377.
    - log_std_init: 0 => std_init = e^(log_std_init) = e^(0) = 1
    """

    def __init__(self, state_dim, action_dim, log_std_init=0):

        super().__init__()

        # mean and log_std vector

        self.backbone, backbone_output_dim = make_backbone(state_dim)
        self.mean_net = nn.Linear(backbone_output_dim, action_dim)

        if do_ortho_init():
            self.backbone.apply(partial(init_weights, gain=K_BACKBONE_GAIN))
            self.mean_net.apply(partial(init_weights, gain=K_ACTION_NET_GAIN))

        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)

    def forward(self, states: torch.tensor) -> Distribution:
        features = self.backbone(states)
        means = self.mean_net(features)
        shared_cov_matrix = torch.diag_embed(torch.exp(self.log_std))
        return MultivariateNormal(means, shared_cov_matrix)

    def forward_determ(self, states: torch.tensor) -> torch.tensor:
        features = self.backbone(states)
        means = self.mean_net(features)
        return means


class MLPCategorialPolicy(nn.Module):

    def __init__(self, state_dim, num_actions):

        super().__init__()

        self.backbone, backbone_output_dim = make_backbone(state_dim)
        self.logit_net = nn.Linear(backbone_output_dim, num_actions)  # outputs a vector of logits, one for each action

        if do_ortho_init():
            self.backbone.apply(partial(init_weights, gain=K_BACKBONE_GAIN))
            self.logit_net.apply(partial(init_weights, gain=K_ACTION_NET_GAIN))

    def forward(self, states: torch.tensor) -> Distribution:
        features = self.backbone(states)
        logits = self.logit_net(features)
        return Categorical(logits=logits)

    def forward_determ(self, states: torch.tensor) -> torch.tensor:
        features = self.backbone(states)
        logits = self.logit_net(features)
        return torch.argmax(logits, dim=1)


class MLPValueFunction(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.backbone, backbone_output_dim = make_backbone(state_dim)

        self.value_net = nn.Linear(backbone_output_dim, 1)

        if do_ortho_init():
            self.backbone.apply(partial(init_weights, gain=K_BACKBONE_GAIN))
            self.value_net.apply(partial(init_weights, gain=K_VALUE_NET_GAIN))

    def forward(self, states: torch.tensor) -> torch.tensor:
        features = self.backbone(states)
        return self.value_net(features)


# class AddOne(nn.Module):
#
#     def forward(self, x):
#         return x + 1
#
#
# class MLPBetaPolicy(nn.Module):
#
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#
#         self.shared_net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU()
#         )
#         self.alphas_layer = nn.Sequential(
#             nn.Linear(256, action_dim),
#             nn.Softplus(),
#             AddOne()
#         )
#         self.betas_layer = nn.Sequential(
#             nn.Linear(256, action_dim),
#             nn.Softplus(),
#             AddOne()
#         )
#
#     def forward(self, states: torch.tensor) -> Distribution:
#         out = self.shared_net(states)
#         alphas, betas = self.alphas_layer(out), self.betas_layer(out)
#         return Independent(Beta(concentration0=alphas, concentration1=betas), reinterpreted_batch_ndims=1)

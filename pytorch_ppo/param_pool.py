from typing import Dict
import numpy as np
import torch

from utils import get_device, save_net, load_net
from policy import MLPBetaPolicy
from episodic_buffer import Data


class ParamPool:

    """Class containing parameters and ways by which they interact with env / data."""

    def __init__(
            self,
            input_dim: int,
            action_dim: int,
            num_epochs_for_policy: int,
            num_epochs_for_value_fn: int,
            eps: float = 0.2
    ):
        self.num_epochs_for_policy = num_epochs_for_policy
        self.num_epochs_for_value_fn = num_epochs_for_value_fn
        self.eps = eps
        self.policy = MLPBetaPolicy(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.policy_optimizer = None
        self.value_fn = None
        self.value_fn_optimizer = None

    def act(self, state: np.array) -> np.array:
        """Output action to be performed in the environment, together with value of state and log p(action|state)"""
        state = torch.from_numpy(state).unsqueeze(0)
        value = self.value_fn(state)
        dist = self.policy(state)  # check shae
        action = dist.sample()  # check sahpe
        log_prob = dist.log_prob(action)  # check shape TODO
        return action, log_prob, value

    def update_networks(self, data: Data) -> Dict[str, float]:
        """Perform gradient steps on policy and value_fn based on data collected under the current policy."""

        init_policy_loss, init_value_fn_loss = None, None

        for i in range(self.num_epochs_for_policy):
            log_prob = self.policy(data.states).log_prob(data.actions)
            ratio = torch.exp(log_prob - data.old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            policy_loss = - (torch.min(ratio * data.adv, clipped_ratio * data.adv)).mean()
            if i == 0: init_policy_loss = float(policy_loss)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        for j in range(self.num_epochs_for_value_fn):
            value_fn_loss = ((self.value_fn(data.states) - data.returns) ** 2).mean()
            if i == 0: init_value_fn_loss = float(value_fn_loss)
            self.value_fn_optimizer.zero_grad()
            value_fn_loss.backward()
            self.value_fn_optimizer.step()

        assert init_policy_loss is not None and init_value_fn_loss is not None

        return {"policy loss": init_policy_loss, "value_fn loss": init_value_fn_loss}

    def save_policy(self, save_dir) -> None:
        save_net(self.policy, save_dir, "policy.pth")

    def load_policy(self, save_dir) -> None:
        load_net(self.policy, save_dir, "policy.pth")

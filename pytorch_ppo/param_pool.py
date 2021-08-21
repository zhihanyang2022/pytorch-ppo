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
            num_iters_for_policy: int,
            num_iters_for_value_fn: int,
            eps: float = 0.2
    ):
        self.num_iters_for_policy = num_iters_for_policy
        self.num_iters_for_value_fn = num_iters_for_value_fn
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

        # Ideally (1), we would train the value function first so that it is aligned with
        # the policy that collected the data. However, as GAE paper (page 8) has pointed
        # out, if value function overfits, then r + V(s') - V(s) would be close to zero,
        # which makes advantage estimation biased. Therefore, it is safer to update the value
        # function after updating the policy (2).

        # (1) Training the value function before the policy is the right thing to do theoretically;
        #     see slides for Lecture 6: Actor-Critic Algorithms of CS 285 by UC Berkeley.
        # (2) Also done in OpenAI SpinningUp's PPO implementation.

        # A major advantage of PPO is the ability to train to convergence on existing data (
        # by maximizing within a trust region around the parameters of the policy used to collect the
        # data). This is why PPO is more "sample-efficient" than Vanilla Policy Gradient algorithms.

        for i in range(self.num_iters_for_policy):
            log_prob = self.policy(data.s).log_prob(data.a)
            ratio = torch.exp(log_prob - data.old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            policy_loss = - (torch.min(ratio * data.adv, clipped_ratio * data.adv)).mean()
            if i == 0: init_policy_loss = float(policy_loss)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        for i in range(self.num_iters_for_value_fn):
            value_fn_loss = ((self.value_fn(data.s) - data.ret) ** 2).mean()
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

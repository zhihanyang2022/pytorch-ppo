from typing import Dict, Tuple
import numpy as np
import torch
import torch.optim as optim

from utils import get_device, save_net, load_net
from policies import MLPBetaPolicy, MLPGaussianPolicy
from value_function import MLPValueFunction

from mpi_utils import mpi_avg_grads



class ParamPool:

    """Class containing parameters and ways by which they interact with env / data."""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            num_iters_for_policy: int,
            num_iters_for_vf: int,
            eps: float = 0.2,
            policy_lr: float = 3e-4,
            vf_lr: float = 1e-3
    ):

        # hyperparameters
        self.num_iters_for_policy = num_iters_for_policy
        self.num_iters_for_vf = num_iters_for_vf
        self.eps = eps

        # important objects
        self.policy = MLPGaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(get_device())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.vf = MLPValueFunction(state_dim=state_dim).to(get_device())
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=vf_lr)

    def act(self, state: np.array) -> Tuple[np.array, float, float]:
        """Output action to be performed in the environment, together with value of state and log p(action|state)"""
        state = torch.from_numpy(state).unsqueeze(0).float()  # (1, state_dim)
        value = self.vf(state)  # (1, 1)
        dist = self.policy(state)
        action = dist.sample()  # (1, action_dim)
        log_prob = dist.log_prob(action)  # (1, )
        return np.clip(np.array(action)[0], -1, 1), float(log_prob), float(value)

    def update_networks(self, data: dict) -> Dict[str, float]:
        """Perform gradient steps on policy and vf based on data collected under the current policy."""

        # Implements PPO-Clip

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

        init_policy_loss, init_vf_loss = None, None

        for i in range(self.num_iters_for_policy):
            log_prob = self.policy(data["obs"]).log_prob(data["act"])
            ratio = torch.exp(log_prob - data["logp"])
            clipped_ratio = torch.clamp(ratio, min=1 - self.eps, max=1 + self.eps)
            policy_loss = - (torch.min(ratio * data["adv"], clipped_ratio * data["adv"])).mean()
            if i == 0: init_policy_loss = float(policy_loss)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            mpi_avg_grads(self.policy)
            self.policy_optimizer.step()

        for i in range(self.num_iters_for_vf):
            vf_loss = ((self.vf(data["obs"]) - data["ret"]) ** 2).mean()
            if i == 0: init_vf_loss = float(vf_loss)
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            mpi_avg_grads(self.vf)
            self.vf_optimizer.step()

        assert init_policy_loss is not None and init_vf_loss is not None

        return {"policy loss": init_policy_loss, "vf loss": init_vf_loss}

    def save_policy(self, save_dir) -> None:
        save_net(self.policy, save_dir, "policy.pth")

    def load_policy(self, save_dir) -> None:
        load_net(self.policy, save_dir, "policy.pth")

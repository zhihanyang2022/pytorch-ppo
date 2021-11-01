from typing import Dict, Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

    def act(self, state: np.array) -> np.array:
        """Output action to be performed in the environment"""
        state = torch.from_numpy(state).unsqueeze(0).float()  # (1, state_dim)
        dist = self.policy(state)
        action = dist.sample()  # (1, action_dim)
        return np.clip(np.array(action)[0], -1, 1)

    def _compute_logp(self, s, a):
        return ""

    def _compute_policy_entropy(self, s):
        return

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

        self.num_epochs = None
        self.batch_size = None  # 64

        # compute old log prob
        # compute values
        # compute advantages (this is the trickiest amongst all, but we can do it in a for-loop)
        # unlike other implementations, this is actually easier to understand, hopefully

        ds = TensorDataset(data["obs"], data["act"], data["old_logp"], data["adv"], data["ret"])
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.num_epochs):

            for obs, act, old_logp, adv, ret in dl:

                # compute ratio

                logp = self._compute_logp(obs, act)
                ratio = torch.exp(logp - old_logp)

                # normalize advantages

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # compute policy loss

                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                policy_loss = torch.min(policy_loss_1, policy_loss_2)

                # computer value function loss

                predicted_values = self.vf(obs)
                value_fn_loss = F.mse_loss(predicted_values, ret)

                # compute entropy loss

                entropy_loss = - torch.mean(entropy)

                # overall loss

                loss = policy_loss + vf_coef * vf_loss + entropy_coef * entropy_loss

                self.policy_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss.backward()
                # clip grad norm
                self.policy_optimizer.step()
                self.vf_optimizer.step()

                # after each batch, check for condition
                # break if condition is unmet

            # break again if condition s unmet

            # TOOD:

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

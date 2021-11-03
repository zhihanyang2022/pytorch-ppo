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



class PPO:

    """Class containing parameters and ways by which they interact with env / data."""

    def __init__(
            self,

            state_dim: int,
            action_dim: int,

            num_epochs: int = 10,  # SB3
            batch_size: int = 64,  # SB3

            eps: float = 0.1,  # openai spinup

            vf_loss_weight: float = 0.5,  # SB3
            entropy_loss_weight: float = 0.0,  # SB3
            max_grad_norm: float = 0.5,  # SB3

            lr: float = 3e-4,  # SB3

            target_kl: float = 1e-3  # openai spinup
    ):

        # hyperparameters

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.eps = eps

        self.vf_loss_weight = vf_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.max_grad_norm = max_grad_norm

        self.target_kl = target_kl

        # important objects

        self.policy = MLPGaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(get_device())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.vf = MLPValueFunction(state_dim=state_dim).to(get_device())
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=lr)

    def act(self, state: np.array) -> np.array:
        """Output action to be performed in the environment, along with other useful stats"""

        state = torch.from_numpy(state).unsqueeze(0).float()  # (1, state_dim)

        dist = self.policy(state)
        action = dist.sample()  # (1, action_dim)
        logp = dist.log_prob(action)  # (1, )
        value = self.vf(state)  # (1, 1)

        # action: for now, we test on continuous control domains
        # logp: required for update_networks
        # value: required for bootstraping in computing returns
        return np.clip(np.array(action)[0], -1, 1), float(logp), float(value)

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

        # for logging

        policy_losses = []
        vf_losses = []
        entropy_losses = []
        losses = []
        approx_kls = []

        # datasets

        ds = TensorDataset(data["states"], data["actions"], data["old_logps"], data["advs"], data["rets"])
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.num_epochs):

            for states, actions, old_logps, advs, rets in dl:

                # compute ratio

                dists = self.policy(states)
                logps = dists.log_prob(actions)
                log_ratios = logps - old_logps
                ratios = torch.exp(log_ratios)

                # normalize advantages

                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                # compute policy loss

                policy_objective_1 = advs * ratios
                policy_objective_2 = advs * torch.clamp(ratios, 1 - self.eps, 1 + self.eps)
                policy_objective = torch.min(policy_objective_1, policy_objective_2).mean()
                policy_loss = - policy_objective

                policy_losses.append(float(policy_loss))

                # computer value function loss

                predicted_values = self.vf(states)
                vf_loss = F.mse_loss(predicted_values, rets)

                vf_losses.append(float(vf_loss))

                # compute entropy loss

                entropies = dists.entropy()
                entropy_objective = torch.mean(entropies)
                entropy_loss = - entropy_objective

                entropy_losses.append(float(entropy_loss))

                # overall loss

                loss = policy_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entropy_loss

                losses.append(float(loss))

                self.policy_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy_optimizer.step()
                self.vf_optimizer.step()

                # after each batch, check for condition
                # break if condition is unmet

                with torch.no_grad():
                    approx_kl = torch.mean((torch.exp(log_ratios) - 1) - log_ratios).cpu().numpy()

                approx_kls.append(float(approx_kl))

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

            # break again if condition s unmet

            if not continue_training:
                break

        # compute stat values, and explained variance

        # for i in range(self.num_iters_for_vf):
        #     vf_loss = ((self.vf(data["obs"]) - data["ret"]) ** 2).mean()
        #     if i == 0: init_vf_loss = float(vf_loss)
        #     self.vf_optimizer.zero_grad()
        #     vf_loss.backward()
        #     mpi_avg_grads(self.vf)
        #     self.vf_optimizer.step()

        return {
            "policy_loss": np.mean(policy_losses),
            "vf_loss": np.mean(vf_losses),
            "entropy_loss": np.mean(entropy_losses),
            "loss": np.mean(losses),
            "approx_kls": np.mean(approx_kls)
        }

    def save_policy(self, save_dir) -> None:
        save_net(self.policy, save_dir, "policy.pth")

    def load_policy(self, save_dir) -> None:
        load_net(self.policy, save_dir, "policy.pth")

from typing import Dict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import gin

from utils import get_device, save_net, load_net, explained_variance
from policies_and_vfs import MLPGaussianPolicy, MLPCategorialPolicy, MLPValueFunction


@gin.configurable(module=__name__)
class PPOClip:

    """
    Class containing neural networks and methods by which they interact with env / data.

    All default hyperparameter values are copied from SB3, and can be overrode by configs.
    """

    def __init__(
        self,

        # env specifications
        state_dim: int,
        action_dim: int,
        num_actions: int,

        # training loop
        num_epochs: int = 10,
        batch_size: int = 64,

        # clipping (alternative to trust region in TRPO)
        eps: float = 0.2,

        # loss
        vf_loss_weight: float = 0.5,
        entropy_loss_weight: float = 0.0,

        # gradient
        max_grad_norm: float = 0.5,
        lr: float = 3e-4,

        # early stopping
        target_kl: float = None
    ):

        assert ((action_dim is None) or (num_actions is None)) and (not (action_dim is None and num_actions is None))

        if action_dim is not None:
            self.action_type = "continuous"
        else:
            self.action_type = "discrete"

        # hyperparameters

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.eps = eps

        self.vf_loss_weight = vf_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.max_grad_norm = max_grad_norm

        self.target_kl = target_kl

        # important objects

        if self.action_type == "continuous":
            self.policy = MLPGaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(get_device())
        else:
            self.policy = MLPCategorialPolicy(state_dim=state_dim, num_actions=num_actions).to(get_device())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.vf = MLPValueFunction(state_dim=state_dim).to(get_device())
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=lr)

    def act(self, state: np.array) -> np.array:

        with torch.no_grad():

            state = torch.from_numpy(state).unsqueeze(0).float()  # (1, state_dim)

            dist = self.policy(state)
            action = dist.sample()  # (1, action_dim) for continuous; (1, ) for discrete

            logp = dist.log_prob(action)  # (1, )
            value = self.vf(state)  # (1, 1)

            # action: for now, we test on continuous control domains
            # logp: required for update_networks
            # value: required for bootstraping in computing returns
            if self.action_type == 'continuous':
                return np.array(action)[0], float(logp), float(value)
            else:
                return int(action), float(logp), float(value)

    def act_determ(self, state: np.array) -> np.array:
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).float()  # (1, state_dim)
            action = self.policy.forward_determ(state)
            if self.action_type == 'continuous':
                return np.array(action)[0]
            else:
                return int(action)

    def update_networks(self, data: dict) -> Dict[str, float]:

        # This method is very similar to SB3's PPO's learn method; some implementation details may be exactly the same.
        # Unlike SB3's PPO, we do not offer the option to do value function clipping (default is false in SB3 anyway).

        # for logging

        policy_losses = []
        vf_losses = []
        entropy_losses = []
        losses = []
        approx_kls = []
        clip_fractions = []

        # datasets

        ds = TensorDataset(*list(map(torch.as_tensor, [data["states"], data["actions"], data["old_logps"], data["advs"], data["rets"]])))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        continue_training = True

        # A major advantage of PPO is the ability to train to convergence on existing data by
        # maximizing within a trust region around the parameters of the policy used to collect the
        # This is why PPO is more "sample-efficient" than Vanilla Policy Gradient algorithms.

        for _ in range(self.num_epochs):

            for states, actions, old_logps, advs, rets in dl:

                # compute ratio

                dists = self.policy(states)
                logps = dists.log_prob(actions)
                log_ratios = logps - old_logps
                ratios = torch.exp(log_ratios)

                clip_fraction = (torch.abs(ratios - 1) > self.eps).float().mean()
                clip_fractions.append(float(clip_fraction))

                # normalize advantages

                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                # compute policy loss

                policy_objective_1 = advs * ratios
                policy_objective_2 = advs * torch.clamp(ratios, 1 - self.eps, 1 + self.eps)
                policy_objective = torch.min(policy_objective_1, policy_objective_2).mean()
                policy_loss = - policy_objective

                policy_losses.append(float(policy_loss))

                # computer value function loss

                # A word about value function training

                # Ideally (1), we would train the value function first so that it is aligned with
                # the policy that collected the data. However, as GAE paper (page 8) has pointed
                # out, if value function overfits, then r + V(s') - V(s) would be close to zero,
                # which makes advantage estimation biased. Therefore, it is safer to update the value
                # function after updating the policy (2).

                # (1) Training the value function before the policy is the right thing to do theoretically;
                #     see slides for Lecture 6: Actor-Critic Algorithms of CS 285 by UC Berkeley.
                # (2) Also done in OpenAI SpinningUp's PPO implementation.

                predicted_values = self.vf(states)
                vf_loss = F.mse_loss(input=predicted_values.flatten(), target=rets)

                vf_losses.append(float(vf_loss))

                # compute entropy loss

                entropies = dists.entropy()
                entropy_objective = torch.mean(entropies)
                entropy_loss = - entropy_objective

                entropy_losses.append(float(entropy_loss))

                # overall loss

                loss = policy_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entropy_loss

                losses.append(float(loss))

                # after each batch, check for condition
                # break if condition is unmet

                with torch.no_grad():
                    approx_kl = torch.mean((ratios - 1) - log_ratios).cpu().numpy()

                approx_kls.append(float(approx_kl))

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    print("Training cut off")
                    break

                # update parameters

                self.policy_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)

                self.policy_optimizer.step()
                self.vf_optimizer.step()

            # break again if condition is unmet

            if not continue_training:
                break

        # compute stat values, and explained variance

        explained_var = explained_variance(y_pred=data["values"], y_true=data["rets"])

        print({
            "policy_loss": np.mean(policy_losses),
            "vf_loss": np.mean(vf_losses),
            "entropy_loss": np.mean(entropy_losses),
            "loss": np.mean(losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
            "explained_var": explained_var
        })

    def save(self, save_dir) -> None:
        save_net(self.vf, save_dir, "vf.pth")
        save_net(self.policy, save_dir, "policy.pth")

    def load(self, save_dir) -> None:
        load_net(self.vf, save_dir, "vf.pth")
        load_net(self.policy, save_dir, "policy.pth")

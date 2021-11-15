import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class EpisodicBuffer:

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            num_envs: int,
            gamma: float = 0.99,
            lam: float = 0.95
    ):
        # hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.gamma, self.lam = gamma, lam
        self.ptr = 0

        # filled on the go
        self.states = np.zeros((self.buffer_size, self.num_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)  # also denote the start of a new episode
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)  # for computing returns and explained variance
        self.log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)  # for update step

        # filled in the end of each step
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)

    def push(
            self,
            states_batch: np.array,
            actions_batch: np.array,
            rewards_batch: np.array,
            dones_batch: np.array,
            values_batch: np.array,
            log_probs_batch: np.array
    ):

        assert self.ptr < self.buffer_size, "Buffer is filled up!"

        self.states[self.ptr] = states_batch
        self.actions[self.ptr] = actions_batch
        self.rewards[self.ptr] = rewards_batch
        self.dones[self.ptr] = dones_batch
        self.values[self.ptr] = values_batch
        self.log_probs[self.ptr] = log_probs_batch

        self.ptr += 1

    def make_data_dict_and_reset(self, values_of_last_next_states, dones):

        # compute advantages and returns
        # Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        # both are truncated

        advantage = 0
        for step in reversed(range(self.buffer_size)):

            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = values_of_last_next_states
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

            advantage = delta + self.gamma * self.lam * next_non_terminal * advantage

            self.advantages[step] = advantage

        self.returns = self.advantages + self.values

        # make dataset

        data = dict(
            states=self.states,
            actions=self.actions,
            values=self.values,
            old_log_probs=self.log_probs,
            advantages=self.advantages,
            returns=self.returns
        )

        # reset buffer

        self.states = np.zeros((self.buffer_size, self.num_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr = 0

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

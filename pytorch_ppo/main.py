import gym
from gym.wrappers import RescaleAction
import numpy as np
from episodic_buffer import EpisodicBuffer
from param_pool import PPOClip
import torch


env = RescaleAction(gym.make("Pendulum-v0"), -1, 1)

param_pool = PPOClip(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    lr=1e-3
)

buffer = EpisodicBuffer(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    size=4096,
    gamma=0.9
)

for e in range(25):

    state = env.reset()  # every epoch should start with a fresh episode

    train_ret = 0
    train_rets = []

    for t in range(4096):

        action, log_prob, value = param_pool.act(state)
        next_state, reward, done, info = env.step(np.clip(action, -1, 1)); train_ret += reward
        buffer.store(state, action, reward, value, log_prob)

        if done:
            buffer.finish_path(last_val=float(param_pool.vf(torch.from_numpy(next_state).float())))  # pendulum is timeout only
            state = env.reset()
            train_rets.append(train_ret)
            train_ret = 0
        else:
            state = next_state

    param_pool.update_networks(buffer.get())

    print(e, np.mean(train_rets), len(train_rets))

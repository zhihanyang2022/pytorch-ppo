import gym
from gym.wrappers import RescaleAction
import numpy as np
from episodic_buffer import EpisodicBuffer
from param_pool import ParamPool
from trash.mpi_utils import setup_pytorch_for_mpi, sync_params, proc_id
from trash.mpi_tools import mpi_fork


num_cpus = 10

setup_pytorch_for_mpi()
mpi_fork(num_cpus)  # we want this number to be as big as possible

env = RescaleAction(gym.make("Pendulum-v0"), -1, 1)

num_epochs = 100
num_steps = 4000 // num_cpus

# should be integer multiple of environment episode length

param_pool = ParamPool(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    num_iters_for_policy=80,
    num_iters_for_vf=80,
)

sync_params(param_pool.policy)
sync_params(param_pool.vf)

buffer = EpisodicBuffer(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    size=num_steps
)

for e in range(num_epochs):

    state = env.reset()  # every epoch should start with a fresh episode

    train_ret = 0
    train_rets = []

    for t in range(num_steps):

        action, log_prob, value = param_pool.act(state)
        next_state, reward, done, info = env.step(action); train_ret += reward
        buffer.store(state, action, reward, value, log_prob)

        if done:
            buffer.finish_path(last_val=value)  # pendulum is timeout only
            state = env.reset()
            train_rets.append(train_ret)
            train_ret = 0
        else:
            state = next_state

    param_pool.update_networks(buffer.get())  # gradient should be the same, so no problem here

    if proc_id() == 0:
        print(e, np.mean(train_rets))

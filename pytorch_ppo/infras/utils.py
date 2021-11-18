import os
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import gym
gym.logger.set_level(40)


def gym_make_advanced(env_name):

    env_raw = gym.make(env_name)

    if isinstance(env_raw.action_space, gym.spaces.Box):

        env = gym.wrappers.RescaleAction(env_raw, -1, 1)

        action_type = "continuous"

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        num_actions = None

    elif isinstance(env_raw.action_space, gym.spaces.Discrete):

        env = env_raw

        action_type = "discrete"

        state_dim = env.observation_space.shape[0]
        action_dim = None
        num_actions = env.action_space.n

    else:

        raise NotImplementedError

    return env, state_dim, action_dim, num_actions, action_type


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    torch.save(net.state_dict(), os.path.join(save_dir, save_name))


def load_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    net.load_state_dict(
        torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device()))
    )


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# from stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# copied from SB3
def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer:
    :param learning_rate:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def remove_dir_from_path(path: str) -> str:
    return path.split('/')[-1]


def remove_jsons_from_dir(directory):
    for fname in os.listdir(directory):
        if fname.endswith('.json'):
            os.remove(os.path.join(directory, fname))

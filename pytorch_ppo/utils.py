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


def train_and_test(env, algo, buffer, num_alters, num_steps_per_alter, action_type):

    for a in range(num_alters):

        # data collection

        state = env.reset()  # every epoch should start with a fresh episode

        train_ret = 0
        train_rets = []
        episode_len = 0

        for t in range(num_steps_per_alter):

            action, log_prob, value = algo.act(state)

            if action_type == "continuous":
                next_state, reward, done, info = env.step(np.clip(action, -1, 1))
            else:
                next_state, reward, done, info = env.step(action)

            train_ret += reward
            episode_len += 1

            buffer.store(state, action, reward, value, log_prob)

            if done:

                if episode_len == env.spec.max_episode_steps:
                    cutoff = info.get('TimeLimit.truncated')
                else:
                    cutoff = False

                if cutoff:
                    last_val = float(algo.vf(torch.from_numpy(next_state).float()))
                else:
                    last_val = 0

                buffer.finish_path(last_val=last_val)
                state = env.reset()
                train_rets.append(train_ret)
                train_ret = 0
                episode_len = 0

            else:

                state = next_state

        # updating parameters

        algo.update_networks(buffer.get())

        # testing

        test_rets = []
        for _ in range(10):
            test_ret = 0
            state = env.reset()
            while True:
                action = algo.act_determ(state)
                if action_type == "continuous":
                    next_state, reward, done, info = env.step(np.clip(action, -1, 1))
                else:
                    next_state, reward, done, info = env.step(action)
                test_ret += reward
                if done:
                    break
                state = next_state
            test_rets.append(test_ret)

        print(a, np.mean(train_rets), np.mean(test_rets))


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

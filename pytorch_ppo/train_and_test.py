import numpy as np
import torch

import gin


@gin.configurable(module=__name__)
def train_and_test(
    env,
    algo,
    buffer,
    num_alters=gin.REQUIRED,
    num_steps_per_alter=gin.REQUIRED,
    action_type=gin.REQUIRED
):

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

            # WARNING: do not store clipped action
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

import numpy as np
import torch
from collections import deque
from gym.wrappers import Monitor

import gin
import wandb
import time
import json


def test_for_one_episode(env, action_type, algo, render):

    test_ret = 0
    test_eplen = 0
    state = env.reset()
    while True:
        action = algo.act_determ(state)
        if action_type == "continuous":
            next_state, reward, done, info = env.step(np.clip(action, -1, 1))
        else:
            next_state, reward, done, info = env.step(action)
        if render:
            env.render()
        test_ret += reward
        test_eplen += 1
        if done:
            break
        state = next_state

    return test_ret, test_eplen


def load_and_visualize_policy(env, action_type, algo, policy_dir, num_episodes, save_videos):

    algo.load(policy_dir)

    if save_videos:  # save videos as well as rendering

        env = Monitor(
            env,
            directory=f'{policy_dir}/videos/',
            video_callable=lambda episode_id: True,  # record every single episode
            force=True
        )

    ep_lens, ep_rets = [], []
    for i in range(num_episodes):
        ep_len, ep_ret = test_for_one_episode(env, action_type, algo, render=True)
        ep_lens.append(ep_len)
        ep_rets.append(ep_ret)

    print('Stats for sanity check:')
    print('Episode Returns:', [round(ret, 2) for ret in ep_rets])
    print('Episode Lengths:', ep_lens)


@gin.configurable(module=__name__)
def train_and_test(
    env_fn,  # we pass env_fn instead of env here so that a separate env can be instantiated for testing
    algo,
    buffer,
    num_alters=gin.REQUIRED,
    num_steps_per_alter=gin.REQUIRED,
    action_type=gin.REQUIRED,
    num_test_episodes=10
):

    start = time.perf_counter()

    # do no ever reset this env, unless when an episode is done
    # do not ever pass this env to testing directly
    env = env_fn()
    test_env = env_fn()

    train_rets_deque = deque(maxlen=20)
    train_eplens_deque = deque(maxlen=20)

    state = env.reset()

    train_ret = 0
    train_eplen = 0

    for a in range(num_alters):

        # data collection

        for t in range(num_steps_per_alter):

            action, log_prob, value = algo.act(state)

            if action_type == "continuous":
                next_state, reward, done, info = env.step(np.clip(action, -1, 1))
            else:
                next_state, reward, done, info = env.step(action)

            train_ret += reward
            train_eplen += 1

            # WARNING: do not store clipped action because log_prob is computed
            # using the unclipped action; if clipped action is stored, learning suffers

            buffer.store(state, action, reward, value, log_prob)

            # compute advantages and do bootstrapping

            # the second condition is crucial when the final episode is cut off
            # otherwise advantages would remain zeros in buffer

            if done or (t == num_steps_per_alter - 1):

                if train_eplen == env.spec.max_episode_steps:
                    cutoff = info.get('TimeLimit.truncated')
                else:
                    cutoff = False

                if cutoff:
                    last_val = float(algo.vf(torch.from_numpy(next_state).float()))
                else:
                    last_val = 0

                buffer.finish_path(last_val=last_val)

            # end of episode handling

            if done:

                train_rets_deque.append(train_ret)
                train_eplens_deque.append(train_eplen)

                train_ret = 0
                train_eplen = 0

                state = env.reset()

            else:

                state = next_state

        # due to numerical imprecision, no advantage should be exactly zero
        # this would throw an error when the 2nd condition is removed from "if done or (t == num_steps_per_alter - 1):"
        # for CartPole-v0, for example

        assert not np.any(buffer.adv_buf == 0), "Part of the advantage buffer is not filled."

        # updating parameters

        dict_for_stats = algo.update_networks(buffer.get(), progress=a/num_alters)

        # testing

        test_rets = []
        test_eplens = []
        for _ in range(num_test_episodes):
            test_ret, test_eplen = test_for_one_episode(test_env, action_type, algo, render=False)
            test_rets.append(test_ret)
            test_eplens.append(test_eplen)

        # reporting stats to wandb

        dict_for_wandb = {}

        dict_for_wandb.update({
            'Episode Return (Train)': np.nan if len(train_rets_deque) == 0 else np.mean(train_rets_deque),
            'Episode Length (Train)': np.nan if len(train_eplens_deque) == 0 else np.mean(train_eplens_deque),
            'Episode Return (Test)': np.mean(test_rets),
            'Episode Length (Test)': np.mean(test_eplens),
            'Hours': (time.perf_counter() - start) / 3600
        })

        dict_for_wandb.update(dict_for_stats)

        num_alters_elapsed = a + 1

        wandb.log(dict_for_wandb, step=num_alters_elapsed * num_steps_per_alter)

        # reporting stats to console

        dict_for_printing = {
            'Progress': num_alters_elapsed / num_alters,
            'Step': num_alters_elapsed * num_steps_per_alter,
        }

        dict_for_printing.update(dict_for_wandb)

        print(json.dumps(dict_for_printing, sort_keys=False, indent=4))

    algo.save(wandb.run.dir)  # need to manually download model later, but more organized

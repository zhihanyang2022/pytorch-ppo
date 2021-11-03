import time
import numpy as np
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


if __name__ == '__main__':

    example_env = gym.make("HalfCheetah-v2")

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    def make_env():
        return gym.make("HalfCheetah-v2")


    env_fns = [make_env for _ in range(16)]
    env = SubprocVecEnv(env_fns)

    start = time.perf_counter()

    for _ in range(1000):

        env.step_async(np.random.normal(size=(16, example_env.action_space.shape[0])))
        observation, reward, done, information = env.step_wait()

    end = time.perf_counter()

    duration = end - start

    print("SubprocVecEnv Duration: ", duration)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env_fns = [make_env for _ in range(16)]
    env = DummyVecEnv(env_fns)

    start = time.perf_counter()

    for _ in range(1000):

        env.step_async(np.random.normal(size=(16, example_env.action_space.shape[0])))
        observation, reward, done, information = env.step_wait()

    end = time.perf_counter()

    duration = end - start

    print("DummyVecEnv Duration: ", duration)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env = make_env()

    start = time.perf_counter()

    for _ in range(16000):

        env.step(env.action_space.sample())
        obs, reward, done, info = env.step()

        if done:
            env = make_env()

    end = time.perf_counter()

    duration = end - start

    print("Standard Duration: ", duration)

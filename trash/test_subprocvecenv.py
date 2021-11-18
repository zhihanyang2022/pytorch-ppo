import time
import numpy as np
import gym
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


if __name__ == '__main__':

    example_env = gym.make("HalfCheetah-v2")

    def make_env():
        return gym.make("HalfCheetah-v2")

    env_fns = [make_env for _ in range(16)]

    def measure_performance(env):

        start = time.perf_counter()

        env.reset()
        for _ in range(1000):
            env.step_async(np.random.normal(size=(16, example_env.action_space.shape[0])))
            observation, reward, done, information = env.step_wait()
        env.close()

        end = time.perf_counter()

        return end - start

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env = ShmemVecEnv(env_fns, context='spawn')
    print("ShmemVecEnv (context='spawn') Duration: ", measure_performance(env))

    env = ShmemVecEnv(env_fns, context='fork')
    print("ShmemVecEnv (context='fork') Duration: ", measure_performance(env))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env = SubprocVecEnv(env_fns)
    print("SubprocVecEnv Duration: ", measure_performance(env))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env = DummyVecEnv(env_fns)
    print("DummyVecEnv Duration: ", measure_performance(env))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    env = make_env()
    start = time.perf_counter()
    env.reset()
    for _ in range(16000):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env = make_env()
            env.reset()
    end = time.perf_counter()
    duration = end - start
    print("Standard Duration: ", duration)

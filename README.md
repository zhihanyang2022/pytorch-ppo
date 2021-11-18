# PPO to RPPO

## First step: running benchmarked PPO code

`rl-baselines3-zoo` is an excellent way to train RL agents (including PPO agents) with `stable-baselines3` and the best hyperparameters. Be careful not to install `rl-baselines-zoo`, a repo whose name closely resemble `rl-baselines3-zoo`; their difference is that `rl-baselines-zoo` uses `stable-baselines` while `rl-baselines3-zoo` uses `stable-baselines3`. In general,`stable-baselines3` is better documented and its maintainers respond faster to questions and issues.

To start, go to:

-   https://github.com/DLR-RM/rl-baselines3-zoo#installation

-   https://github.com/DLR-RM/rl-baselines3-zoo#train-an-agent

You do not need GPUs for PPO, but it’s ideal to have a relatively large number of CPUs.

As a first step, we train PPO on one continuous-control domain (Pendulum-v0) and one discrete-control domain (CartPole-v0). Later on, we will reproduce these performances using our custom implementation.



Waiting on the interpretation of monitor files and plot_train.py

Potential to help a little bit with plotting functionality

## Our PPO

Features that it have and not have, vs stable-baseliens current implementation

-   Policy and value networks
    -   Same: same architecture, same activation function, we do not share part of policy and value networks, we do weight initializatio
    -   We do not use feature_extractor class, mlp extractor

-   Hyperparameters
    -   Our config files encode the exact hyperparameters tuned by `rl-baselines3-zoo`

Questions:

-   Do SB3’s data collection start from new episode each time?
-   How does the tuning process work? How are the hyperparameter ranges chosen?

Sharing is not used by default in PPO?

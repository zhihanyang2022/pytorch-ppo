# PPO-Clip

Minimal PyTorch implementation of PPO-Clip, without parallelism. Converge very stably within a
minute on CartPole-v0 and Pendulum-v0. 

## Requirements

```bash
pip install wandb==0.12.2 numpy==1.19.5 torch==1.9.1 scipy==1.6.2 gym==0.18.3
```

Don't use a gym version that's newer because `Pendulum-v0` or `CartPole-v0` might be unavailable (i.e., only newer version is available).

## Scripts

Setup wandb.

Train:

```python
python launch.py --expdir experiments/CartPole-v0_ppo_sb3 --run_id 1
```

Visualize policy (require you to download the trained models from wandb):

```python
python launch.py --expdir experiments/CartPole-v0_ppo_sb3 --run_id 1 --enjoy
```

## Plots

You can view plots on wandb. Here's an example screenshot:

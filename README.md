I'm considering reproducing SB3's PPO's performance on some hard domains, so stay tuned!

# PPO-Clip

In this repo, I minimally but very carefully implemented PPO-Clip in PyTorch, without parallelism. It has been very time-consuming since there are many details to get right, even though I had stable-baselines3 as a reference.

Now, it's working very well and converges very stably within a minute on CartPole-v0 and Pendulum-v0. 

CartPole-v0             |  Pendulum-v0 | HalfCheetah-v3
:-------------------------:|:-------------------------:|:-------------------------:|
![ezgif com-gif-maker](https://user-images.githubusercontent.com/43589364/148905684-8e313857-868d-489f-aba0-1abbce30c3de.gif) | ![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/43589364/148905756-1450b124-c499-4eb3-aadc-db0a4049a44d.gif) | (see below for video)
![image](https://user-images.githubusercontent.com/43589364/148906014-8d69927f-3c0b-4a09-9dc8-537d0b8be754.png) | ![image](https://user-images.githubusercontent.com/43589364/148906065-11107c86-16c3-46c4-9eff-601d18d442c6.png) | ![image](https://user-images.githubusercontent.com/43589364/149087421-0018acda-e870-4c2d-9db2-52eafe6df7e2.png)

https://user-images.githubusercontent.com/43589364/149088284-ec9a5759-6383-4b02-a907-348d394f7d40.mp4

## Requirements

```bash
pip install wandb==0.12.2 numpy==1.19.5 torch==1.9.1 scipy==1.6.2 gym==0.18.3
```

Don't use a gym version that's newer because `Pendulum-v0` or `CartPole-v0` might be unavailable (i.e., only newer version is available).

## Scripts

Setup wandb.

Train:

```bash
python launch.py --expdir experiments/CartPole-v0_ppo_sb3 --run_id 1
```

Visualize policy (require you to download the trained models from wandb):

```bash
python launch.py --expdir experiments/CartPole-v0_ppo_sb3 --run_id 1 --enjoy
```

## Plots

You can view plots on wandb. Here's an example screenshot:

![image](https://user-images.githubusercontent.com/43589364/148901571-e91203df-6ce4-41d4-a876-d3f3de288c22.png)

## FAQs

*Why doesn't parallel data collection bring you as much speedup as you think?*

Sure, parallel data collection brings some speedup, but PPO is fast primarily because it's taking far fewer gradient steps than, e.g., DDPG, given the same number of environment timesteps. See https://github.com/DLR-RM/stable-baselines3/issues/643 for a thorough discussion.

*How is this implementation different from SB3's PPO?*

They are supposed to be exactly the same, except that this repo doesn't have certain arguments, which are default to None or False in SB3 anyways. The hyper-parameters in config files were copied from rl-baselines3-zoo. These hyper-parameter values are good because they are tuned.

*What hyper-parameter values are you using?*

Checkout the config files in the two subdirs of `experiments`, one for the simplest discrete control domain and the other for the simplest continuous control domain. 

*How to extend this codebase for research purposes?*

You can create a new file in `algorithms`, add it to the `algo_name2class` dictionary inside `launch.py`. Then, you can simply specify that you want to run that algorithm in config files, which means you need to create a folder in `experiments` to contain that config file. 

## Thanks

I'd like to thank the maintainers of SB3 for answering my questions during this process.

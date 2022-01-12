import os
import argparse
import gin
import wandb

from algorithms.ppo import PPO
from infras.episodic_buffer import EpisodicBuffer
from infras.utils import gym_make_advanced, remove_dir_from_path
from infras.run_fns import train_and_test, load_and_visualize_policy
from launch_utils import pick


algo_name2class = {
    'PPO': PPO
}

parser = argparse.ArgumentParser()
parser.add_argument("--expdir", type=str, required=True)
parser.add_argument('--run_id', nargs='+', type=int, required=True)
parser.add_argument("--enjoy", action='store_true')
args = parser.parse_args()

gin.parse_config_file(os.path.join(args.expdir, "config.gin"))

env, algo = pick()
algo_klass = algo_name2class[algo]

for run_id in args.run_id:

    if args.enjoy:

        env_fn, state_dim, action_dim, num_actions, action_type = gym_make_advanced(env)

        algo = algo_klass(state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)

        load_and_visualize_policy(
            env=env_fn(),
            action_type=action_type,
            algo=algo,
            policy_dir=f"{args.expdir}/{run_id}",
            num_episodes=5,
            save_videos=True  # do no render on screen, but you can watch later for many times
        )

    else:

        run = wandb.init(
            project="pytorch_ppo",  # os.getenv('OFFPCC_WANDB_PROJECT'),
            entity="yangz2",  # os.getenv('OFFPCC_WANDB_ENTITY'),
            group=args.expdir,
            settings=wandb.Settings(_disable_stats=True),
            name=f'run_id={run_id}',
            reinit=True
        )

        env_fn, state_dim, action_dim, num_actions, action_type = gym_make_advanced(env)

        print(f"Env name: {env}")
        print(f"Detected action type: {action_type}")
        print(f"Exp dir: {args.expdir}")

        algo = algo_klass(state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)
        buffer = EpisodicBuffer(obs_dim=state_dim, act_dim=action_dim)

        train_and_test(env_fn=env_fn, action_type=action_type, algo=algo, buffer=buffer)

        run.finish()

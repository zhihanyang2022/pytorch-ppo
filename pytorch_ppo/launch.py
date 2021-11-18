import os
import argparse
import gin
import wandb

from algorithms.ppo import PPO
from infras.episodic_buffer import EpisodicBuffer
from infras.utils import gym_make_advanced, remove_dir_from_path
from infras.run_fns import train_and_test, load_and_visualize_policy


algo_name2class = {
    'ppo': PPO
}

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--algo", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument('--run_id', nargs='+', type=int, required=True)
parser.add_argument("--enjoy", action='store_true')
args = parser.parse_args()

gin.parse_config_file(args.config)

for run_id in args.run_id:

    if args.enjoy:

        env, state_dim, action_dim, num_actions, action_type = gym_make_advanced(args.env)

        algo = algo_name2class[args.algo](state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)

        load_and_visualize_policy(
            env=env,
            action_type=action_type,
            algo=algo,
            policy_dir=f"pytorch-ppo-pretrained/{args.algo}_{args.env}_{remove_dir_from_path(args.config).split('.')[0]}/{run_id}",
            num_episodes=5,
            save_videos=True  # do no render on screen, but you can watch later for many times
        )

    else:

        run = wandb.init(
            project=os.getenv('OFFPCC_WANDB_PROJECT'),
            entity=os.getenv('OFFPCC_WANDB_ENTITY'),
            group=f"{args.env} {args.algo} {remove_dir_from_path(args.config)}",
            settings=wandb.Settings(_disable_stats=True),
            name=f'run_id={run_id}',
            reinit=True
        )

        env, state_dim, action_dim, num_actions, action_type = gym_make_advanced(args.env)

        print(f"Env name: {args.env}")
        print(f"Detected action type: {action_type}")
        print(f"Config: {remove_dir_from_path(args.config)}")

        algo = algo_name2class[args.algo](state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)
        buffer = EpisodicBuffer(obs_dim=state_dim, act_dim=action_dim)

        train_and_test(env=env, action_type=action_type, algo=algo, buffer=buffer)

        run.finish()

import os
import argparse
import gin
import wandb

from algorithms.ppo import PPO
from infra.episodic_buffer import EpisodicBuffer
from infra.utils import gym_make_advanced
from infra.train_and_test import train_and_test


algo_name2class = {
    'ppo': PPO
}

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--algo", type=str, required=True)
parser.add_argument('--run_id', nargs='+', type=int, required=True)
args = parser.parse_args()

for run_id in args.run_id:

    config_path = f"configs/{args.algo}/{args.env}.gin"

    run = wandb.init(
        project=os.getenv('OFFPCC_WANDB_PROJECT'),
        entity=os.getenv('OFFPCC_WANDB_ENTITY'),
        group=f"{args.env} {args.algo}",
        settings=wandb.Settings(_disable_stats=True),
        name=f'run_id={run_id}',
        reinit=True
    )

    gin.parse_config_file(config_path)

    env, state_dim, action_dim, num_actions, action_type = gym_make_advanced(args.env)

    print(f"Env name: {args.env}")
    print(f"Detected action type: {action_type}")
    print(f"Config: {config_path}")

    algo = algo_name2class[args.algo](state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)
    buffer = EpisodicBuffer(obs_dim=state_dim, act_dim=action_dim)

    train_and_test(env=env, action_type=action_type, algo=algo, buffer=buffer)

    run.finish()

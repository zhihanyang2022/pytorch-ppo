import os
import argparse
import gin
import wandb

from param_pool import PPOClip
from episodic_buffer import EpisodicBuffer
from utils import gym_make_advanced
from train_and_test import train_and_test


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--env", type=str, required=True)
args = parser.parse_args()

# TODO: specify run id as part of args
# TODO: allow automatical re-run

run = wandb.init(
    project=os.getenv('OFFPCC_WANDB_PROJECT'),
    entity=os.getenv('OFFPCC_WANDB_ENTITY'),
    group=f"{args.env} {args.algo} {args.config.split('configs/')[-1]} (ours)",
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={run_id}',
    reinit=True
)

# config should be automatically selected by algorithm and task (at this point)
# use algoname to specify the algorithm class

gin.parse_config_file(f"configs/{args.env}.gin")

env, state_dim, action_dim, num_actions, action_type = gym_make_advanced(args.env)

print(f"Env name: {args.env}")
print(f"Detected action type: {action_type}")
print(f"Config: {f'configs/{args.env}.gin'}")

ppo = PPOClip(state_dim=state_dim, action_dim=action_dim, num_actions=num_actions)
buffer = EpisodicBuffer(obs_dim=state_dim, act_dim=action_dim)

train_and_test(env=env, action_type=action_type, algo=ppo, buffer=buffer)

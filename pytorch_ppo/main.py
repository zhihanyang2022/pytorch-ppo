import argparse
import gin

from param_pool import PPOClip
from episodic_buffer import EpisodicBuffer
from utils import gym_make_advanced, train_and_test


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
args = parser.parse_args()

# gin.parse_config_file(f"configs/{args.env}.gin")

env, state_dim, action_dim, num_actions, action_type = gym_make_advanced(args.env)

print(f"Detected {action_type}-action environment ...")

ppo = PPOClip(state_dim=state_dim, action_dim=action_dim, num_actions=num_actions,
              num_epochs=10, lr=1e-3)
buffer = EpisodicBuffer(obs_dim=state_dim, act_dim=action_dim,
                        size=4096, gamma=0.9, lam=0.95)

train_and_test(env=env, action_type=action_type, algo=ppo, buffer=buffer,
               num_alters=25, num_steps_per_alter=4096)

#!/usr/bin/env python
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from gym_simulations.envs.discrete_2dof import Discrete2DoF

if __name__ == '__main__':
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    gym.envs.register(
        id='Discrete2DoF-v0',
        entry_point='gym_simulations.envs.discrete_2dof:Discrete2DoF',
        max_episode_steps=4000.0, # Limit steps to avoid getting stuck in training and eval
        reward_threshold=-1000.0,
    )

    # Create environment
    env = gym.make('Discrete2DoF-v0')

    print('loading...')
    # Instantiate the agent
    model = DQN.load(BASE_PATH + 'sim_outputs/models/dqn_default_2dof_arm_try3')
    # Evaluate the agent
    print('evaluating...')
    n_episodes = 10
    mean_reward, std_reward = evaluate_policy(model, env,
                                                n_eval_episodes=n_episodes)
    print("[Mean reward over last {} eval episodes is {}".format(n_episodes,
                                                                mean_reward))

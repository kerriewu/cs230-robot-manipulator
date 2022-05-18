#!/usr/bin/env python

# Code adapted from examples on Stable Baselines3 docs
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

# Reference: A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus,
# N. Dormann. Stable Baselines3: Reliable Reinforcement Learning
# Implementations. In:Journal of Machine Learning Research 22(268), 1-8 (2015).
# http://jmlr.org/papers/v22/20-1364.html.

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from gym_simulations.envs.discrete_3dof import Discrete3DoF

if __name__ == '__main__':
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    gym.envs.register(
        id='Box2DoF-v0',
        entry_point='gym_simulations.envs.box_2dof:Box2DoF',
        max_episode_steps=4000.0, # Limit steps to avoid getting stuck in training and eval
        reward_threshold=-1000.0,
    )


    # Create environment
    env = gym.make('Box2DoF-v0')
    random_env = gym.make('Box2DoF-v0')
    n_episodes = 100

    print('loading...')
    # # Instantiate a random baseline
    # random_model = DQN('MlpPolicy', random_env)
    # print('evaluating random model')
    # rewards, lengths = evaluate_policy(random_model, random_env,
                                                # n_eval_episodes=n_episodes,
                                                # return_episode_rewards=True)
    # rewards = np.array(rewards)
    # lengths = np.array(lengths)
    # num_successes = np.sum(np.array(lengths < 4000))
    # avg_reward = np.mean(rewards)
    # print("[Mean reward for random baseline over last {} eval episodes is {},"
          # " number of successes is {}".format(n_episodes,
                                              # avg_reward,
                                              # num_successes))

    # Instantiate the trained agent
    model = DDPG.load(BASE_PATH + 'sim_outputs/models/ddpg_box_2dof_01')
    # # Evaluate the agent
    # print('evaluating...')
    # rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_episodes,
                                                # return_episode_rewards=True)
    # rewards = np.array(rewards)
    # lengths = np.array(lengths)
    # num_successes = np.sum(np.array(lengths < 4000))
    # avg_reward = np.mean(rewards)
    # print("[Mean reward for trained baseline over last {} eval episodes is {},"
          # " number of successes is {}".format(n_episodes,
                                              # avg_reward,
                                              # num_successes))
    print("Visual simulation")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

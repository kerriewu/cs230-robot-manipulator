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
import supersuit as ss
import numpy as np

from gym_simulations.envs.passing_game import PassingGame

if __name__ == '__main__':
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    random_env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    trained_env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    random_env = ss.concat_vec_envs_v1(random_env, 1, num_cpus=1, base_class="stable_baselines3")
    trained_env = ss.concat_vec_envs_v1(trained_env, 1, num_cpus=1, base_class="stable_baselines3")

    n_episodes = 10

    # print('loading...')
    # # Instantiate the trained agent
    model = DQN.load(BASE_PATH + 'sim_outputs/models/dqn_passing_game_01')
    # # Evaluate the agent
    # print('evaluating...')
    # rewards, lengths = evaluate_policy(model, trained_env, n_eval_episodes=n_episodes,
    #                                             return_episode_rewards=True)
    # rewards = np.array(rewards)
    # lengths = np.array(lengths)
    # num_successes = np.sum(np.array(lengths < 10000))
    # avg_reward = np.mean(rewards)
    # print("[Mean reward for trained baseline over last {} eval episodes is {},"
    #       " number of successes is {}".format(n_episodes,
    #                                           avg_reward,
    #                                           num_successes))
    #
    # # Instantiate a random baseline
    # random_model = DQN('MlpPolicy', random_env)
    # print('evaluating random model')
    # rewards, lengths = evaluate_policy(random_model, random_env,
    #                                             n_eval_episodes=n_episodes,
    #                                             return_episode_rewards=True)
    # rewards = np.array(rewards)
    # lengths = np.array(lengths)
    # num_successes = np.sum(np.array(lengths < 10000))
    # avg_reward = np.mean(rewards)
    # print("[Mean reward for random baseline over last {} eval episodes is {},"
    #       " number of successes is {}".format(n_episodes,
    #                                           avg_reward,
    #                                           num_successes))

    obs = trained_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = trained_env.step(action)
        trained_env.render()
        if done.all():
            obs = trained_env.reset()

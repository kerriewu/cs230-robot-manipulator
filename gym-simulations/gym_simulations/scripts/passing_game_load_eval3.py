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

import os
import gym
import pettingzoo
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import supersuit as ss

from gym_simulations.envs.passing_game3 import PassingGame

if __name__ == '__main__':
    model_desc = "147_entry_or_bin_n5_5e5"
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    # Create environment
    env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    eval_env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")

    # print('loading...')
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
    model = DQN.load(BASE_PATH + 'sim_outputs/models/' + model_desc)
    # Evaluate the agent
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

    print("Rendering with learned policy")
    obs = env.reset()
    # Measure performance without render for first few episodes, then render
    n_success = 0
    n_episodes = 0
    while True:
        actions, _states = model.predict(obs)
        obs, reward, done, info = env.step(actions)
        if n_episodes >= 0:
            env.render()
        if True in done:
            n_episodes += 1
            for info_dict in info:
                if info_dict["is_success"]:
                    n_success += 1
            obs = env.reset()
            print("Success rate: %.2f" % (n_success / n_episodes))

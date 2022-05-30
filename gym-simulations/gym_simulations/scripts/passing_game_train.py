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
from stable_baselines3 import DQN, PPO, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import supersuit as ss
import tensorflow as tf

from gym_simulations.envs.passing_game import PassingGame

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.

    Plots length of the episode and the reward per episode.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # print(self.locals)
        # print(self.locals['infos'][0]['done'])
        if self.locals['infos'][0]['done']:
            # print('logged')
            self.logger.record("rollout/episode_length_timesteps", self.locals['infos'][0]['current_steps'])
            self.logger.record("rollout/episode_reward", self.locals['infos'][0]['current_episode_reward'])
            self.logger.record("rollout/episode_reward", self.locals['infos'][1]['current_episode_reward'])
        return True

if __name__ == '__main__':
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    # Create environment
    log_dir = BASE_PATH+'sim_outputs/passing_game_logs_02/'
    os.makedirs(log_dir, exist_ok=True)
    env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    eval_env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")

    # env = Monitor(env, log_dir)
    # eval_env = Monitor(env, log_dir + "eval/")
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1e8, verbose=1)
    eval_callback = EvalCallback(
         eval_env,
         best_model_save_path=log_dir,
         log_path=log_dir,
         eval_freq=1e5,
         deterministic=True,
         render=True,
         callback_on_new_best=callback_on_best)

    callback=CallbackList([TensorboardCallback(), eval_callback])

    # Instantiate the agent
    model = DQN('MlpPolicy',
                env,
                learning_starts=1e5,
                exploration_fraction=0.95,
                exploration_final_eps=0.50,
                batch_size=256,
                learning_rate=1e-3,
                gamma=0.999,
                verbose=1,
                tensorboard_log=log_dir)

    print('training...')
    # model.learn(total_timesteps=int(0.6e6), callback=eval_callback)
    model.learn(total_timesteps=int(1.0e6), callback=callback)
    print('trained for some timesteps')
    print('saving')
    model.save(BASE_PATH + 'sim_outputs/models/dqn_passing_game_02')
    # Evaluate the agent
    print('evaluating...')
    mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                n_eval_episodes=10, render=True)
    print('evaluated')
    print("[Mean reward over last 10 eval episodes is {}".format(mean_reward))

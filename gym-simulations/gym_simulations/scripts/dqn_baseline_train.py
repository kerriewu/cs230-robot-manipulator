#!/usr/bin/env python

# Code adapted from examples on
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from gym_simulations.envs.discrete_2dof import Discrete2DoF

if __name__ == '__main__':
    gym.envs.register(
        id='Discrete2DoF-v0',
        entry_point='gym_simulations.envs.discrete_2dof:Discrete2DoF',
        max_episode_steps=4000.0, # Limit steps to avoid getting stuck in training and eval
        reward_threshold=-1000.0,
    )

    # base path
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    # Create environment
    log_dir = BASE_PATH+'sim_outputs/dqn_default_2dof_arm_try4_logs/'
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make('Discrete2DoF-v0')
    eval_env = gym.make('Discrete2DoF-v0')
    env = Monitor(env, log_dir)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1000, verbose=1)
    eval_callback = EvalCallback(
         eval_env,
         best_model_save_path=log_dir,
         log_path=log_dir,
         eval_freq=1e5,
         deterministic=True,
         render=False,
         callback_on_new_best=callback_on_best)

    # Instantiate the agent
    model = DQN('MlpPolicy',
                env,
                learning_starts=1e5,
                exploration_fraction=0.95,
                exploration_final_eps=0.0,
                learning_rate=1e-3,
                verbose=1)

    print('training...')
    model.learn(total_timesteps=int(1e6), callback=eval_callback)
    print('trained for some timesteps')
    print('saving')
    model.save(BASE_PATH + 'sim_outputs/models/dqn_default_2dof_arm_try4')
    # Evaluate the agent
    print('evaluating...')
    mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                n_eval_episodes=10)
    print('evaluated')
    print("[Mean reward over last 10 eval episodes is {}".format(mean_reward))

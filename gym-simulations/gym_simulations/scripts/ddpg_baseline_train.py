#!/usr/bin/env python

# Code adapted from examples on Stable Baselines3 docs
# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html

# Reference: A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus,
# N. Dormann. Stable Baselines3: Reliable Reinforcement Learning
# Implementations. In:Journal of Machine Learning Research 22(268), 1-8 (2015).
# http://jmlr.org/papers/v22/20-1364.html.


from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
from gym_simulations.envs.box_3dof import Box3DoF

import os
import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (NormalActionNoise,
        OrnsteinUhlenbeckActionNoise)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (CallbackList, EvalCallback, 
        StopTrainingOnRewardThreshold)

if __name__ == '__main__':
    gym.envs.register(
            id='Box2DoF-v0',
            entry_point='gym_simulations.envs.box_2dof:Box2DoF',
            max_episode_steps=4000.0, # Limit steps to avoid getting stuck in training and eval
            reward_threshold=10000.0,
    )
        
     # base path
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    # Create environment
    log_dir = BASE_PATH+'sim_outputs/ddpg_box_2dof_01_logs/'
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make('Box2DoF-v0')
    eval_env = gym.make('Box2DoF-v0')
    env = Monitor(env, log_dir)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10000, 
            verbose=1)
    eval_callback = EvalCallback(
         eval_env,
         best_model_save_path=log_dir,
         log_path=log_dir,
         eval_freq=1e5,
         deterministic=True,
         render=False,
         callback_on_new_best=callback_on_best)


    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
            sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", 
                 env, 
                 learning_starts=1e4,
                 learning_rate=.005,
                 verbose=1,
                 action_noise=action_noise)
    print('training...')
    model.learn(total_timesteps=1e6, callback=eval_callback)
    print('trained for some timesteps')
    print('saving')
    model.save(BASE_PATH + 'sim_outputs/models/ddpg_box_2dof_01')
    env = model.get_env()

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
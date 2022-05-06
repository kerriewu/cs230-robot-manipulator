#!/usr/bin/env python
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold

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
    env = gym.make('Discrete2DoF-v0')
    eval_env = gym.make('Discrete2DoF-v0')

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1000, verbose=1)
    eval_callback = EvalCallback(
         eval_env,
         best_model_save_path=BASE_PATH+'sim_outputs/dqn_default_2dof_arm_try3_logs/',
         log_path=BASE_PATH+'sim_outputs/dqn_default_2dof_arm_try3_logs/',
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
    model.save(BASE_PATH + 'sim_outputs/models/dqn_default_2dof_arm_try3')
    # Evaluate the agent
    print('evaluating...')
    mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                n_eval_episodes=10)
    print('evaluated')
    print("[Mean reward over last 10 eval episodes is {}".format(mean_reward))

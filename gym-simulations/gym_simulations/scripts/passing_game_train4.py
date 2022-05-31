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

from gym_simulations.envs.passing_game4 import PassingGame

if __name__ == '__main__':
    model_desc = '159_4_phase_observed_1e6'
    BASE_PATH='cs230-robot-manipulator/gym-simulations/gym_simulations/'

    # Create environment
    log_dir = BASE_PATH+'sim_outputs/logs/' + model_desc
    os.makedirs(log_dir, exist_ok=True)
    env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    eval_env = ss.pettingzoo_env_to_vec_env_v1(PassingGame())
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
    eval_callback = EvalCallback(
         eval_env,
         best_model_save_path=log_dir,
         log_path=log_dir,
         eval_freq=3.5e5,
         deterministic=True,
         render=False,
         callback_on_new_best=callback_on_best)

    # Instantiate the agent
    model = DQN('MlpPolicy',
                env,
                learning_starts=1e5,
                exploration_fraction=0.95,
                exploration_final_eps=0.5,
                learning_rate=1e-3,
                gamma=0.999,
                verbose=1,
                batch_size=256,
                policy_kwargs={"net_arch": [64, 64]}
    )

    print("\nModel architecture:") 
    print(model.policy)

    print('\ntraining...')
    model.learn(total_timesteps=int(1e6))#, callback=eval_callback)
    print('trained for some timesteps')
    print('saving')
    model.save(BASE_PATH + 'sim_outputs/models/' + model_desc)
    
    # # Evaluate the agent
    # print('evaluating...')
    # mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                # n_eval_episodes=10)
    # print('evaluated')
    # print("[Mean reward over last 10 eval episodes is {}".format(mean_reward))
    
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
    

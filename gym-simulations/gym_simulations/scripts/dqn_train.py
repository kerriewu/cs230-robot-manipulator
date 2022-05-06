#!/usr/bin/env python
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

# Demo, code mostly from this tutorial and modified to fit our custom environment:
# https://www.theconstructsim.com/testing-different-openai-rl-algorithms-with-ros-and-gazebo/

# Inspired by https://keon.io/deep-q-learning/
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2

from gym_simulations.envs.discrete_2dof import Discrete2DoF

class DQNRobotSolver():
    def __init__(self,
     n_observations,
     n_actions,
     n_episodes=1000,
     n_win_steps=195,
     min_episodes= 100,
     max_env_steps=None,
     gamma=1.0,
     epsilon=1.0,
     epsilon_min=0.01,
     epsilon_log_decay=0.995,
     alpha=0.01,
     alpha_decay=0.01,
     batch_size=64,
     quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = Discrete2DoF()
        # if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)

        self.input_dim = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_steps = n_win_steps # reward at which we consider ourselves done with the task
        self.min_episodes = min_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()

        self.model.add(Dense(24, input_dim=self.input_dim, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.alpha, decay=self.alpha_decay))


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        flattened_state = [
                            state['joint_angs'][0],
                            state['joint_angs'][1],
                            state['joint_locs'][0][0],
                            state['joint_locs'][0][1],
                            state['joint_locs'][1][0],
                            state['joint_locs'][1][1],
                            state['reward_loc'][0],
                            state['reward_loc'][1]
                            ]
        return np.reshape(flattened_state, [1, self.input_dim])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):

        scores = deque(maxlen=100)
        successes = deque(maxlen=100)
        max_rewards = deque(maxlen=100)

        for e in range(self.n_episodes):

            init_state = self.env.reset()

            state = self.preprocess_state(init_state)
            done = False
            i = 0
            max_reward = -np.Inf

            while not (done or i > 1000):
                action = self.choose_action(state, self.get_epsilon(e))
                observation, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(observation)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                max_reward = max(max_reward, reward)
                i += 1
                if i % 25 == 0:
                    print("step {}:".format(i))
                    print("end_effector_location: {}, {}".format(observation['joint_locs'][1][0], observation['joint_locs'][1][1]))
                    print("reward location: {}, {}".format(observation['reward_loc'][0],observation['reward_loc'][1]))

            scores.append(i)
            mean_score = np.mean(scores)
            max_rewards.append(max_reward)
            successes.append(done)
            median_max_reward = np.median(max_rewards)
            num_successes = np.count_nonzero(successes)
            if mean_score <= self.n_win_steps and e >= min_episodes:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - min_episodes))
                return e - min_episodes
            if e % 1 == 0 and not self.quiet:
                print('[Episode {}] - Mean steps before termination over last {} episodes was {} steps. Number of successes was {}. Median max reward was {}'.format(e, min_episodes, mean_score, num_successes, median_max_reward))

            for num_batches in range(int(i /  self.batch_size)):
                self.replay(self.batch_size)


        if not self.quiet: print('Did not solve after {} episodes'.format(e))
        return e

if __name__ == '__main__':

    n_observations = 8
    n_actions = 4

    n_episodes = 1000
    n_win_steps = 250
    min_episodes = 100
    max_env_steps = None
    gamma =  1.0
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_log_decay = 0.995
    alpha = 0.01
    alpha_decay = 0.01
    batch_size = 64
    quiet = False


    agent = DQNRobotSolver(
                                n_observations,
                                n_actions,
                                n_episodes,
                                n_win_steps,
                                min_episodes,
                                max_env_steps,
                                gamma,
                                epsilon,
                                epsilon_min,
                                epsilon_log_decay,
                                alpha,
                                alpha_decay,
                                batch_size,
                                quiet)
    agent.run()

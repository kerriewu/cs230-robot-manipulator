#
# passing_game.py
#
# Environment for 4-dof robotic arms playing a passing game with discrete
# action space.
# CS 230 Final Project
# 26 May 2022
#

import gym
import functools
from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import api_test, parallel_api_test, test_save_obs
from gym_simulations.envs.game_token import Token
from gym_simulations.envs.arm import Arm
# from game_token import Token
# from arm import Arm


class PassingGame(ParallelEnv):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 300}

    def __init__(self,
                 angle_deltas=[.01745, .01745, .01745, .01745],
                 max_episode_steps = 10000):
        """
        Creates a new environment for 2 4DOF arms in a 3D space for a game.

        The game setup is as follows:
            Each agent controls one arm, and the agent's goal is to place all
        tokens within bin_radius of the token's respective bin_location. Tokens
        start at a token_entry_location which is within the arm's reach (a
        different location for each arm). However, the desired bin_location is
        out of its reach and only in the reach of the other arm in the game.
        Therefore the arm needs to coordinate with the other agent in order to
        achieve the goal.

        For attributes that are held in a length two list, the index corresponds
        to the agent.
        Eg: Agent 0 corresponds to
                self.arms[0]
                self.bin_locations[0],
                self.token_entry_locations[0],
                self.tokens[0],

        Args:
            angle_deltas: the change in angle for each action.
            max_episode_steps: the max number of timesteps per episode.
        """

        # Two possible arms / agents, which are each identical
        self.possible_agents = ["arm_" + str(r) for r in range(2)]

        self.agent_name_mapping = dict(zip(self.possible_agents,
                                        list(range(len(self.possible_agents)))))

        # Angle deltas are the same for both arms.
        self.angle_deltas = np.array(angle_deltas)

        # Location for each arm's tokens to be dropped at for reward.
        self.bin_locations = [np.array([20.0, 0.0, 0.0]),
                              np.array([-20.0, 0.0, 0.0])]

        # Drop tokens this close to the bin location for reward
        self.bin_radius = 1.0

        # Where each token is picked up
        self.token_entry_locations = [np.array([-16.0, +8.0, 0.0]),
                                      np.array([+16.0, -8.0, 0.0])]

        # Number of tokens to hand to bin before completing the game
        self.num_tokens_per_arm = 5

        # Maps action space to angular delta on each joint. See action_space()
        self._action_to_direction = {
                0: np.array([  self.angle_deltas[0],  0.,  0.,  0.]),
                1: np.array([ -self.angle_deltas[0],  0.,  0.,  0.]),
                2: np.array([  0.,  self.angle_deltas[1],  0.,  0.]),
                3: np.array([  0., -self.angle_deltas[1],  0.,  0.]),
                4: np.array([  0.,  0.,  self.angle_deltas[2],  0.]),
                5: np.array([  0.,  0., -self.angle_deltas[2],  0.]),
                6: np.array([  0.,  0.,  0.,  self.angle_deltas[3]]),
                7: np.array([  0.,  0.,  0., -self.angle_deltas[3]]),
                8: np.zeros((4,)),
                9: np.zeros((4,)),
        }

        self.max_episode_steps = max_episode_steps # Limit max number of timesteps

        self.reset()

    """ See _get_obs for documentation on observations."""
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-50, high=50, shape=(74,))

    """ Discrete actions: +base_angle,
                          -base_angle,
                          +arm1,
                          -arm1,
                          +arm2,
                          -arm2,
                          +arm3,
                          -arm3,
                          get_token,
                          drop_token """
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(10)

    def is_token_at_bin(self, token):
        """ Return true if token is at the correct bin"""
        distance_to_bin = np.linalg.norm(token.location -
                                         self.bin_locations[token.arm_id])
        if token.state == "dropped" and distance_to_bin < self.bin_radius:
            return True
        return False

    def _get_obs(self, agent_index):
        """ Returns an 74x1 array with observations from current state.


            Each agent observes:
              Self arm Angles            4
              Self base location         3
              Self joint locations       3 * 3
              Self bin location          3
              Self token entry location  3
              Self token location        3 * self.num_tokens_per_arm
              Other arm Angles           4
              Other base location        3
              Other joint locations      3 * 3
              Other bin location         3
              Other token entry location 3
              Other token location       3 * self.num_tokens_per_arm
              Total                     74

              The agent observes its own information first, then the other
              agent's information (so the observation is symmetric per agent).
        """
        agent_arm = self.arms[agent_index]
        other_arm = self.arms[agent_index - 1]

        state = np.concatenate((agent_arm.angles,
                        agent_arm.origin,
                        agent_arm.joint_locs.flatten(),
                        self.bin_locations[agent_index],
                        self.token_entry_locations[agent_index],
                        np.array([token.location for token in
                                    self.tokens[agent_index]]).flatten(),
                        other_arm.angles,
                        other_arm.origin,
                        other_arm.joint_locs.flatten(),
                        self.bin_locations[agent_index-1],
                        self.token_entry_locations[agent_index-1],
                        np.array([token.location for token in
                                    self.tokens[agent_index-1]]).flatten()
                        ))

        return np.array(state).reshape(74,)

    def _success(self):
        """ Returns true if all tokens have been dropped at the appropriate
            bin locations. """
        success = True
        for arm_tokens in self.tokens:
            for token in arm_tokens:
                success = success and self.is_token_at_bin(token)
        return success

    def _get_info(self, agent_index):
        """ Returns a dictionary with information about the current state. """
        return {"is_success": self._success()}

    def reset(self, seed=None, return_info=False, options=None):
        """ Resets simulation environment. See gym.env.reset(). """
        self.agents = self.possible_agents[:]

        # Reset Arms and Tokens
        self.arms = [Arm(link_lengths=[6., 4., 2.5],
                         origin=[-10.0, 0.0, 0.0]),
                     Arm(link_lengths=[6., 4., 2.5],
                         origin=[10.0, 0.0, 0.0])]

        self.tokens = [
            [Token(self.token_entry_locations[0], 0)
                for i in range(self.num_tokens_per_arm)],
            [Token(self.token_entry_locations[1], 1)
                for i in range(self.num_tokens_per_arm)]
            ]

        observation = {self.agents[0]: self._get_obs(0),
                       self.agents[1]: self._get_obs(1)}
        info = {self.agents[0]: self._get_info(0),
                self.agents[1]: self._get_info(1)}

        self.current_steps = 0

        return (observation, info) if return_info else observation

    def step(self, actions):
        """ Applies one step in the simulation.

        Args:
            actions: dictionary of (int) discrete actions, one for each agent

        Returns:
            dicts for each of the following for each agent:
                observation: Observations from current state
                reward (float): Any reward accumulated from the timestep
                done (bool): Whether simulation has terminated
                info (dict): Auxiliary information about environment
            where each dict looks like: {agent_0: item1, agent_1: item2}
        """
        # If a user passes in actions with no agents, then just return empty
        # observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}


        reward = {self.agents[0]: 0., self.agents[1]: 0.}
        for i in range(len(self.agents)):
            arm = self.arms[i]
            action = actions[self.agents[i]]
            arm.update_angles_and_joint_locs(
                            self._action_to_direction[action])
            if action == 8:
                # Only pick up a token if:
                #   -Arm is not already holding a token
                #   -Token is dropped
                #   -Token is not already at its bin location
                #   -Arm is close to token
                for arm_tokens in self.tokens:
                    if arm.held_token is not None:
                        break
                    for token in arm_tokens:
                        if token.state != "dropped":
                            continue
                        if self.is_token_at_bin(token):
                            continue
                        distance_to_arm = np.linalg.norm(token.location -
                                                         arm.joint_locs[-1])
                        if distance_to_arm <= 0.25:
                            arm.get_token(token)
                            break
            if action == 9:
                token = arm.drop_token()
                if token:
                    if self.is_token_at_bin(token):
                        reward[self.agents[i]] = 100
                        reward[self.agents[i-1]] = 100

        self.current_steps += 1
        env_done = self.current_steps > self.max_episode_steps

        done = {self.agents[0]: self._success() or env_done,
                self.agents[1]: self._success() or env_done}
        observation = {self.agents[0]: self._get_obs(0),
                       self.agents[1]: self._get_obs(1)}
        info = {self.agents[0]: self._get_info(0),
                self.agents[1]: self._get_info(1)}
        return observation, reward, done, info

    def render(self, mode="human"):
        """ Renders simulation for viewing. """
        pass

    def close(self):
        """ Closes any open resources used by the environment. """
        pass

if __name__ == '__main__':
    env = PassingGame()
    env.reset()
    env.step({agent: 5 for agent in env.agents})
    env.step({agent: 2 for agent in env.agents})
    parallel_api_test(env)

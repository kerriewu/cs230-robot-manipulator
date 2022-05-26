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
from pettingzoo.test import api_test, parallel_api_test
from game_token import Token
from arm import Arm


class PassingGame(ParallelEnv):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 300}

    def __init__(self,
                 angle_deltas=[.01745, .01745, .01745, .01745]):
        """
        Creates a new environment for 2 4DOF arms in a 3D space for a game.
         TODO(kerwu) add more thorough description.

        For attributes that are held in a length two list, the index corresponds
        to the agent.
        Eg: Agent 0 corresponds to
                self.arms[0]
                self.bin_locations[0],
                self.token_entry_locations[0],
                self.tokens[0],

        Args:
            angle_deltas: the change in angle for each action.
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

        # Maps action space to angular delta on each joint.
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

        self._action_spaces = {agent: spaces.Discrete(10)
                                for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Box(low=-50, high=50, shape=(74,))
             for agent in self.possible_agents
        }

        self.reset()



    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-50, high=50, shape=(74,))

    """Discrete actions: +base_angle,
                          -base_angle,
                          +arm1,
                          -arm1,
                          +arm2,
                          -arm2,
                          +arm3,
                          -arm3,
                          get_token,
                          drop_token"""
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

            TODO(kerwu) clarify ordering in comment.

            Each agent observes:
              Self arm Angles            4
              Self base location         3
              Self joint locations       3 * 3
              Other arm Angles           4
              Other base location        3
              Other joint locations      3 * 3
              All token entry locations  2 * 3
              All bin locations          2 * 3
              All token locations        self.num_tokens_per_arm * 2 * 3
              Total = 44 + 6 * self.num_tokens_per_arm (5) = 74

              The agent observes its own information first, then the other
              agent's information
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

    # TODO(kerwu): fill this out.
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

        observation = {self.agents[i]: self._get_obs(i)
                        for i in range(len(self.agents))}
        info = {self.agents[i]: self._get_info(i)
                    for i in range(len(self.agents))}
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
            where each dict looks like: {agent_1: item1, agent_2: item2}
        """
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

        done = {self.agents[0]: self._success(),
                self.agents[1]: self._success()}
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
    # api_test(env, num_cycles=100)
    # env = env()
    # env.reset()
    #
    # print(env.action_space(env.agents[1]))
    # env.step([2,2])

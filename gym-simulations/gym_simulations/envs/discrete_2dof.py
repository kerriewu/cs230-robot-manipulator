#
# discrete_2dof.py
#
# Environment for 2-dof robotic arm with discrete action space.
# CS 230 Final Project
# 1 May 2022
#

import gym
from gym import spaces
import numpy as np

class Discrete2DoF(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 4}

    def __init__(self, reward_loc=[5., 8.], angle_deltas=[.01745, .01745],
            link_lengths=[6., 4.]):
        """
        Creates a new environment for a 2-dof arm with discrete action space.

        Args:
            reward_loc (2,): (x, y) location of reward_loc (m)
            angle_deltas (2,): angle update when actioning joints 1 and 2 (rad)
            link_lengths (2,): Lengths of links 1 and 2 (m)
        """
        self.angle_deltas = np.array(angle_deltas)
        self.reward_loc = np.array(reward_loc)
        self.link_lengths = np.array(link_lengths)
        self.arm_angles = np.array([0., 0.])  # Joint 1 and joint 2, radians
        self.update_joint_locs()
        self.update_reward()

        # 3 observations:
        #   (1) Angles of each of the (2) joints
        #   (2) End points (x, y) of each of the (2) joints
        #   (3) Location (x, y) of the reward
        self.observation_space = spaces.Dict({
                "joint_angs" : spaces.Box(low=-np.pi, high=np.pi, shape=(2,),
                        dtype=np.float32),
                "joint_locs" : spaces.Box(low=-np.inf, high=np.inf,
                        shape=(2, 2), dtype=np.float32),
                "reward_loc" : spaces.Box(low=-np.inf, high=np.inf,
                        shape=(1,), dtype=np.float32)
        })

        # Discrete actions: +joint 1, -joint1, +joint2, -joint2
        self.action_space = spaces.Discrete(4)

        # Maps action space to angular delta on each of the two joints.
        self._action_to_direction = {
                0: np.array([ self.angle_deltas[0],                    0.]),
                1: np.array([-self.angle_deltas[0],                    0.]),
                2: np.array([                   0.,  self.angle_deltas[1]]),
                3: np.array([                   0., -self.angle_deltas[1]])
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """ Returns a dictionary with observations from current state. """
        return {
                "joint_angs" : self.arm_angles,
                "joint_locs" : self.joint_locs,
                "reward_loc" : self.reward_loc}

    def _get_info(self):
        """ Returns a dictionary with information about the current state. """
        return {}

    def update_arm_angles(self, delta_angles):
        """ Adds delta_angles (2,) to self.arm_angles and wraps to [-pi, pi] """
        a = self.arm_angles + delta_angles
        self.arm_angles = np.array([(x + np.pi) % (2*np.pi) - np.pi for x in a])

    def calculate_end_effector_location(self, t1, t2):
        l1, l2 = self.link_lengths[0], self.link_lengths[1]
        end_effector_location = np.array(
            [l1 * np.cos(t1) + l2 * np.cos(t1 + t2),
             l1 * np.sin(t1) + l2 * np.sin(t1 + t2)]
             )
        return end_effector_location

    def update_joint_locs(self):
        """ Returns x, y locations of the 2 joints (2, 2). """
        #TODO: This should pull directly from robot simulation in ROS.
        t1, t2 = self.arm_angles[0], self.arm_angles[1]
        l1, l2 = self.link_lengths[0], self.link_lengths[1]
        self.joint_locs = np.array(
                [[l1 * np.cos(t1), l1 * np.sin(t1)],
                 end_effector_location])

    def update_reward(self):
        """ Updates reward based on distance to target. """
        self.reward = 1. / (1E-6 + np.linalg.norm(self.joint_locs[1] -
                self.reward_loc))

    def reset(self, seed=None, return_info=False, options=None):
        """ Resets simulation environment. See gym.env.reset(). """

        # We need the following line to seed self.np_random
        ##super().reset(seed=seed)

        # Set a reward location within the arm's reacheable space randomly.
        t1 = np.random.uniform(low=0.0, high=2.*np.pi)
        t2 = np.random.uniform(low=0.0, high=2.*np.pi)

        self.reward_loc = self.calculate_end_effector_location(t1 ,t2)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        """ Applies one step in the simulation.

        Args:
            action (int) discrete action (0,1 = +/- joint 1; 2,3 = +/- joint 2)

        Returns:
            observation (dict): joint_angs, joint_locs, reward_loc
            reward (float): Euclidean distance between Joint 2 and reward
            done (bool): Whether simulation has terminated (i.e. reward reached)
            info (dict): Auxiliary information about environment
        """
        self.update_arm_angles(self._action_to_direction[action])
        self.update_joint_locs()
        self.update_reward()

        # Epsiode is done if we have reached the target.
        done = np.linalg.norm(self.joint_locs[1] - self.reward_loc) <= 0.25
        reward = self.reward
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        """ Renders simulation for viewing. """
        #TODO Rendering will happen in ROS
        #TODO Perhaps make a simulation in pygame to sidestep ROS initially?
        pass

    def close(self):
        """ Closes any open resources used by the environment. """
        pass

if __name__ == '__main__':
    environment = Discrete2DoF()
    environment.reset()
    environment.step(2)

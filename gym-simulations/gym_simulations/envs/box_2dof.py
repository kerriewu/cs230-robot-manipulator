#
# box_2dof.py
#
# Environment for 2-dof robotic arm with box action space.
# CS 230 Final Project
# 1 May 2022
#

import gym
from gym import spaces
import numpy as np

class Box2DoF(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 300}

    def __init__(self, reward_loc=[5., 8.], angle_deltas=[.01745, .01745],
            link_lengths=[6., 4.]):
        """
        Creates a new environment for a 2-dof arm with box action space.

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

        # 8x1 vector including 2 joint angles (rad), 2 joint locations (x, y
        # cartesian coordinate pairs), and the reward location (x. y cartesian
        # coordinate pair).
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(8,))

        # Discrete actions: +joint 1, -joint1, +joint2, -joint2
        self.action_space = spaces.Box(low=-angle_deltas[0], 
                high=angle_deltas[0], shape=(2,))

        """
        If human-rendering is used, `self.screen` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.screen = None
        self.clock = None
        self.screen_width = 512  # Size of the pygame window, pixels
        self.screen_height = 512  # Size of the pygame window, pixels
        self.screen_scale = 20  # pixels per meter 
        

    def _get_obs(self):
        """ Returns an 8x1 array with observations from current state. """
        return np.array([
                        self.arm_angles[0],
                        self.arm_angles[1],
                        self.joint_locs[0][0],
                        self.joint_locs[0][1],
                        self.joint_locs[1][0],
                        self.joint_locs[1][1],
                        self.reward_loc[0],
                        self.reward_loc[1]
                        ]).reshape(8,)

    def _success(self):
        """ Returns true if we have just reached a reward """
        return np.linalg.norm(self.joint_locs[1] - self.reward_loc) <= 0.25

    def _get_info(self):
        """ Returns a dictionary with information about the current state. """
        return {"is_success": self._success(),
                "joint_angs" : self.arm_angles,
                "joint_locs" : self.joint_locs,
                "reward_loc" : self.reward_loc}

    def update_arm_angles(self, delta_angles):
        """ Adds delta_angles (2,) to self.arm_angles and wraps to [-pi, pi] """
        a = self.arm_angles + delta_angles
        self.arm_angles = np.array([(x + np.pi) % (2*np.pi) - np.pi for x in a])

    def calculate_end_effector_location(self, t1, t2):
        l1, l2 = self.link_lengths[0], self.link_lengths[1]
        end_effector_location = np.array([
                    l1 * np.cos(t1) + l2 * np.cos(t1 + t2),
                    l1 * np.sin(t1) + l2 * np.sin(t1 + t2)])
        return end_effector_location

    def update_joint_locs(self):
        """ Returns x, y locations of the 2 joints (2, 2). """
        #TODO: This should pull directly from robot simulation in ROS.
        t1, t2 = self.arm_angles[0], self.arm_angles[1]
        l1, l2 = self.link_lengths[0], self.link_lengths[1]
        end_effector_location = self.calculate_end_effector_location(t1, t2)
        self.joint_locs = np.array(
                [[l1 * np.cos(t1), l1 * np.sin(t1)],
                end_effector_location])

    def update_reward(self):
        """ Updates reward based on distance to target. """
        self.reward = -(np.linalg.norm(self.joint_locs[1] -
                self.reward_loc))
        
    def reset(self, seed=None, return_info=False, options=None):
        """ Resets simulation environment. See gym.env.reset(). """

        # We need the following line to seed self.np_random
        ##super().reset(seed=seed)
        
        # Reset reward location
        t1 = np.random.uniform(low=0.0, high=2*np.pi)
        t2 = np.random.uniform(low=0.0, high=2*np.pi)
        self.reward_loc = self.calculate_end_effector_location(t1, t2)
        
        # Reset arm location
        self.arm_angles = np.array([np.random.uniform(low=0.0, high=2*np.pi),
                                    np.random.uniform(low=0.0, high=2*np.pi)])
        
        # Update environment
        self.update_joint_locs()
        self.update_reward()
        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
        
    def step(self, action):
        """ Applies one step in the simulation.

        Args:
            action (int) continuous action within limits on each joint

        Returns:
            observation (8x1 array): Observations from current state
            reward (float): Euclidean distance between Joint 2 and reward
            done (bool): Whether simulation has terminated (i.e. reward reached)
            info (dict): Auxiliary information about environment
        """
        self.update_arm_angles(action)
        self.update_joint_locs()
        self.update_reward()

        # Epsiode is done if we have reached the target.
        done = self._success()
        reward = self.reward
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        """ Renders simulation for viewing. """
        #TODO Rendering will happen in ROS
        #TODO Perhaps make a simulation in pygame to sidestep ROS initially?
        import pygame # avoid global pygame dependency. 
                      # This method is not called with no-render. 
        black = (0, 0, 0)
        green = (30, 200, 30)
        red = (200, 30, 30)
        cyan = (30, 150, 120)
        white = (255, 255, 255)
        #
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(white)
        origin_px = np.array([self.screen_width // 2, self.screen_height // 2])
        joint_1_px = origin_px + self.joint_locs[0] * [1,-1] * self.screen_scale
        joint_2_px = origin_px + self.joint_locs[1] * [1,-1] * self.screen_scale
        goal_ctr_px = origin_px + self.reward_loc * [1,-1] * self.screen_scale
        goal_wd = 8
        # Goal, cyan:
        pygame.draw.rect(
                canvas,
                cyan,
                (np.int32(goal_ctr_px)[0] - goal_wd//3, 
                 np.int32(goal_ctr_px)[1] - goal_wd//3,
                 goal_wd,
                 goal_wd)
        )
        # Link 2, red:
        pygame.draw.line(
                canvas, 
                red, 
                np.int32(joint_1_px), 
                np.int32(joint_2_px),
                width=3
        )
        pygame.draw.circle(
                canvas,
                red,
                np.int32(joint_2_px),
                5
        )      
        # Link 1, green:
        pygame.draw.line(
                canvas, 
                green, 
                np.int32(origin_px), 
                np.int32(joint_1_px),
                width=5
        )
        pygame.draw.circle(
                canvas,
                green,
                np.int32(joint_1_px),
                5
        )   
        # Origin, black:
        pygame.draw.circle(
                canvas,
                black,
                np.int32(origin_px),
                7
        )               
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, 
                                                   self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if mode == "human":
            assert self.screen is not None
            # The following line copies our drawings from `canvas` to 
            # the visible window
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined 
            # framerate. The following line will automatically add a delay to 
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    
    def close(self):
        """ Closes any open resources used by the environment. """
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = Box2DoF()
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample())
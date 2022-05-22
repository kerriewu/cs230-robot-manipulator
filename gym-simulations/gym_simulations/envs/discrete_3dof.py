#
# discrete_3dof.py
#
# Environment for 3-dof robotic arm with discrete action space.
# CS 230 Final Project
# 17 May 2022
#

import gym
from gym import spaces
import numpy as np

class Discrete3DoF(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 300}

    def __init__(self, reward_loc=[5., 8.], 
            angle_deltas=[.01745, .01745, .01745], link_lengths=[6., 4., 2.5]):
        """
        Creates a new environment for a 3-dof arm with discrete action space.

        Args:
            reward_loc (2,): (x, y) location of reward_loc (m)
            angle_deltas (3,): angle update when actioning each joint (rad)
            link_lengths (3,): Lengths of each link (m)
        """
        self.angle_deltas = np.array(angle_deltas)
        self.reward_loc = np.array(reward_loc)
        self.link_lengths = np.array(link_lengths)
        self.arm_angles = np.array([0., 0., 0.])  # Joint angles, radians
        self.update_joint_locs()
        self.update_reward()

        # 11x1 vector including 3 joint angles (rad), 3 joint locations (x, y
        # cartesian coordinate pairs), and the reward location (x, y cartesian
        # coordinate pair).
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(11,))

        # Discrete actions: +arm1, -arm1, +arm2, -arm2, +arm3, -arm3
        self.action_space = spaces.Discrete(6)

        # Maps action space to angular delta on each joint. 
        self._action_to_direction = {
                0: np.array([  self.angle_deltas[0],  0.,  0.]),
                1: np.array([ -self.angle_deltas[0],  0.,  0.]),
                2: np.array([  0.,  self.angle_deltas[1],  0.]),
                3: np.array([  0., -self.angle_deltas[1],  0.]),
                4: np.array([  0.,  0.,  self.angle_deltas[2]]),
                5: np.array([  0.,  0., -self.angle_deltas[2]]),
        }

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
        """ Returns an 11x1 array with observations from current state. """
        return np.array([
                        self.arm_angles[0],
                        self.arm_angles[1],
                        self.arm_angles[2],
                        self.joint_locs[0][0],
                        self.joint_locs[0][1],
                        self.joint_locs[1][0],
                        self.joint_locs[1][1],
                        self.joint_locs[2][0],
                        self.joint_locs[2][1],
                        self.reward_loc[0],
                        self.reward_loc[1]
                        ]).reshape(11,)

    def _success(self):
        """ Returns true if we have just reached a reward """
        return np.linalg.norm(self.joint_locs[-1] - self.reward_loc) <= 0.25

    def _get_info(self):
        """ Returns a dictionary with information about the current state. """
        return {"is_success": self._success(),
                "joint_angs" : self.arm_angles,
                "joint_locs" : self.joint_locs,
                "reward_loc" : self.reward_loc}

    def update_arm_angles(self, delta_angles):
        """ Adds delta_angles (m3,) to self.arm_angles and wraps to [-pi, pi] """
        a = self.arm_angles + delta_angles
        self.arm_angles = np.array([(x + np.pi) % (2*np.pi) - np.pi for x in a])

    def calculate_joint_locs(self, angles):
        """ Returns a list of the location of each joint. 
        
        Args:
            angles: list of floats; joint angles in radians
            lengths: list of floats; link lengths in meters
        
        Returns:
            joint_locs: List of shape (2,) np arrays; x,y coords for each joint.
        """
        joint_locs = [np.zeros(2)]  # Include origin for now; won't be returned
        for angle, length in zip(np.cumsum(angles), self.link_lengths):
            joint_locs.append(joint_locs[-1] + 
                    length * np.array([np.cos(angle), np.sin(angle)]))
        return joint_locs[1:]  # omit origin
    
    def update_joint_locs(self):
        """ Updates self.joint_locs based on current arm angles. """
        #TODO: This should pull directly from robot simulation in ROS.
        self.joint_locs = self.calculate_joint_locs(self.arm_angles)

    def update_reward(self):
        """ Updates reward based on distance to target. """
        self.reward = -(np.linalg.norm(self.joint_locs[-1] -
                self.reward_loc))

    def reset(self, seed=None, return_info=False, options=None):
        """ Resets simulation environment. See gym.env.reset(). """

        # We need the following line to seed self.np_random
        ##super().reset(seed=seed)
        
        # Reset reward location
        angles = np.random.uniform(low=0.0, high=2*np.pi, size=3)
        self.reward_loc = self.calculate_joint_locs(angles)[-2]

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
            action (int) discrete action (+/- each joint angle)

        Returns:
            observation (11x1 array): Observations from current state
            reward (float): Euclidean distance between Joint 2 and reward
            done (bool): Whether simulation has terminated (i.e. reward reached)
            info (dict): Auxiliary information about environment
        """
        self.update_arm_angles(self._action_to_direction[action])
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
        blue = (30, 30, 200)
        cyan = (30, 150, 120)
        white = (255, 255, 255)
        #
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(white)
        origin_px = np.array([self.screen_width // 2, self.screen_height // 2])
        joint_1_px = origin_px + self.joint_locs[0] * [1,-1] * self.screen_scale
        joint_2_px = origin_px + self.joint_locs[1] * [1,-1] * self.screen_scale
        joint_3_px = origin_px + self.joint_locs[2] * [1,-1] * self.screen_scale
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
        # Link 3, blue:
        pygame.draw.line(
                canvas, 
                blue, 
                np.int32(joint_2_px), 
                np.int32(joint_3_px),
                width=3
        )
        pygame.draw.circle(
                canvas,
                blue,
                np.int32(joint_3_px),
                4
        )      
        # Link 2, red:
        pygame.draw.line(
                canvas, 
                red, 
                np.int32(joint_1_px), 
                np.int32(joint_2_px),
                width=4
        )
        pygame.draw.circle(
                canvas,
                red,
                np.int32(joint_2_px),
                4
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
    env = Discrete3DoF()
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(np.random.choice([0,1,2,3,4,5]))
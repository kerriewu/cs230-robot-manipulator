try:
    from gym_simulations.envs.game_token4 import Token
except (ModuleNotFoundError, ImportError):
    from game_token4 import Token
import numpy as np

class Arm():
    """
    A 4DOF arm object.

    Implements a 4DOF, 3-link arm for use in passing_game.py in a 3D space.
    The 4 degrees of freedom are:
        0: base rotation
        1: first arm link angle
        2: second arm link angle
        3: third arm link angle
    The end effector is located at the end of the third link. The arm can hold
    and drop Tokens (defined in game_token.py).
    """

    def __init__(self, link_lengths, origin):
        self.link_lengths = np.array(link_lengths)
        self.origin = np.array(origin)
        self.held_token = None
        
        self.num_picked_up = 0  # number of tokens picked up
        self.phase = 0  # 0: Other entry; 
                        # 1: Other checkpoint; 
                        # 2: Own checkpoint;
                        # 3: Own bin

        # Angles in radians
        # Indices:
        #   0: base rotation
        #   1-3: arm link angles
        # initial = everything in a straight line going up.
        self.angles = np.array([0.0, 0.0, 0.0, 0.0])
        self.update_angles_and_joint_locs(np.zeros(4,))

    def update_angles_and_joint_locs(self, delta_angles):
        """ Adds delta_angles to self.angles and wraps to [-pi, pi]
            Updates self.joint_locs based on current arm angles.
            Also updates token location if there is a held token.

            Does NOT update locations if the new angles cause intersection with
            the ground plane"""
        angles = self.angles + delta_angles
        angles = np.array([(x + np.pi) % (2*np.pi) - np.pi for x in angles])
        new_joint_locs = self.calculate_joint_locs(angles)
        if (new_joint_locs[0][2] >= 0 and
            new_joint_locs[1][2] >= 0 and
            new_joint_locs[2][2] >= 0):
            self.angles = angles
            self.joint_locs = new_joint_locs
        if self.held_token:
            self.held_token.update_location(self.joint_locs[-1])
            # print(self.held_token.location)

    def calculate_joint_locs(self, angles):
        """ Returns a list of the location of each joint.

        Args:
            angles: list of floats; joint angles in radians
            lengths: list of floats; link lengths in meters

        Returns:
            joint_locs: List of shape (3,) np arrays; x,y,z coords for each joint.
        """
        joint_locs = np.zeros((1,3))  # Include origin for now; won't return
        base_angle = angles[0]
        for angle, length in zip(np.cumsum(angles[1:]), self.link_lengths):
            joint_locs = np.vstack([joint_locs, joint_locs[-1] +
                    length * np.array([np.cos(angle), 0, np.sin(angle)])])
        # Rotate the whole -90 degrees about Y
        R1 = np.array([[ np.cos(-np.pi/2), 0, np.sin(-np.pi/2)],
                       [                0, 1,                0],
                       [-np.sin(-np.pi/2), 0, np.cos(-np.pi/2)]])                
        # Rotate by base angle
        R2 = np.array([[np.cos(base_angle), -np.sin(base_angle), 0],
                      [np.sin(base_angle),  np.cos(base_angle), 0],
                      [                 0,                   0, 1]])
        joint_locs = (R2 @ R1 @ joint_locs.T).T  # matrix multiply each column

        # Add to origin location
        joint_locs += [self.origin]  # Adds origin to each row of joint_locs

        return joint_locs[1:]  # omit origin

    def get_token(self, token):
        """ Get a token if possible """
        # assert self.held_token is None
        # assert token.state == 'dropped'
        # assert np.linalg.norm(self.joint_locs[-1] - token.location) <= 0.25
        self.held_token = token
        token.set_state('held')
        self.held_token.update_location(self.joint_locs[-1])
        self.num_picked_up += 1
        self.phase = (self.phase + 1) % 4
        # print("picked up token")
        # print(token.location)

    def drop_token(self):
        """ Drop token if possible
            Returns: None if no token was dropped
                     The dropped token if a token was dropped """
        if self.held_token is not None:
            token = self.held_token
            self.held_token.set_state('dropped')
            self.held_token = None
            self.phase = (self.phase + 1) % 4
            # print("dropped token")
            # print(token.location)
            return token
        return None
        
if __name__ == '__main__':
    # joint locs calculation test 
    link_lengths = [1, 2, 3]
    origin = np.array([10, 10, 100])
    a = Arm(link_lengths, origin)
    angles = [0, np.pi/2, 0, 0]
    print(a.calculate_joint_locs(angles))
    

from game_token import Token
import numpy as np

class Arm():
    """
    A 4DOF arm object.

    TODO(kerwu) add more docs.
    """

    def __init__(self, link_lengths, origin):
        self.link_lengths = np.array(link_lengths)
        self.origin = np.array(origin)
        self.held_token = None

        # Angles in radians
        # Indices:
        #   0: base rotation
        #   1-3: arm link angles
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

    def calculate_joint_locs(self, angles):
        """ Returns a list of the location of each joint.

        Args:
            angles: list of floats; joint angles in radians
            lengths: list of floats; link lengths in meters

        Returns:
            joint_locs: List of shape (3,) np arrays; x,y,z coords for each joint.
        """
        joint_locs = [np.zeros(3)]  # Include origin for now; won't be returned
        base_angle = angles[0]
        for angle, length in zip(np.cumsum(angles[1:]), self.link_lengths):
            joint_locs.append(joint_locs[-1] +
                    length * np.array([np.cos(angle) * np.cos(base_angle),
                                       np.cos(angle) * np.sin(base_angle),
                                       np.sin(angle)]))

        # Add to origin location
        joint_locs += self.origin

        return joint_locs[1:]  # omit origin

    def get_token(self, token):
        """ Get a token if possible """
        assert self.held_token is None
        assert token.state == 'dropped'
        assert np.linalg.norm(self.joint_locs[-1] - token.location) <= 0.25
        self.held_token = token
        token.set_state('held')

    def drop_token(self):
        """ Drop token if possible
            Returns: None if no token was dropped
                     The dropped token if a token was dropped """
        if self.held_token is not None:
            token = self.held_token
            self.held_token.set_state('dropped')
            self.held_token = None
            return token
        return None

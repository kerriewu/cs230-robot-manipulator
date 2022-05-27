import numpy as np
from roboticstoolbox.robot.ERobot import ERobot, ERobot2
# from roboticstoolbox.tools import URDF
from spatialmath import SE3



# # Update & configure path
# from pathlib import Path
# import sys
# path_root = Path(__file__).parents[2]
# sys.path.append(str(path_root))
# from pathlib import PurePosixPath
# from os.path import splitext
# print(sys.path)
# print("\#")


class TwoPandaWorld(ERobot):
    """
    Class that imports a xacro file with two panda robots. 
    Implementation mirrors example from PythonRoboticsToolbox.
    ``Panda()`` is a class which imports a Franka-Emika Panda robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.
    .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
        >>> print(robot)
    Defined joint configurations are:
    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration
    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "franka_description/robots/two_panda_world_no_limits.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            #gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        #self.grippers[0].tool = SE3(0, 0, 0.1034)

        # self.qdlim = np.array(
            # [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0,
             # 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        # )

        self.qr = np.zeros(20)
        # self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0, 0, 0,
                            # 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0, 0, 0])
        self.qz = np.zeros(20)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)



    # @staticmethod
    # def URDF_read(file_path, tld=None, xacro_tld=None):
        # """
        # Read a URDF file as Links
        # :param file_path: File path relative to the xacro folder
        # :type file_path: str, in Posix file path fprmat
        # :param tld: A custom top-level directory which holds the xacro data,
            # defaults to None
        # :type tld: str, optional
        # :param xacro_tld: A custom top-level within the xacro data,
            # defaults to None
        # :type xacro_tld: str, optional
        # :return: Links and robot name
        # :rtype: tuple(Link list, str)
        # File should be specified relative to ``RTBDATA/URDF/xacro``
        # .. note:: If ``tld`` is not supplied, filepath pointing to xacro data should
            # be directly under ``RTBDATA/URDF/xacro`` OR under ``./xacro`` relative
            # to the model file calling this method. If ``tld`` is supplied, then
            # ```file_path``` needs to be relative to ``tld``
        # """

        # # get the path to the class that defines the robot
        # if tld is None:
            # base_path = rtb_path_to_datafile("xacro")
        # else:
            # base_path = PurePosixPath(tld)
        # # print("*** urdf_to_ets_args: ", classpath)
        # # add on relative path to get to the URDF or xacro file
        # # base_path = PurePath(classpath).parent.parent / 'URDF' / 'xacro'
        # file_path = base_path / PurePosixPath(file_path)
        # name, ext = splitext(file_path)

        # if ext == ".xacro":
            # # it's a xacro file, preprocess it
            # if xacro_tld is not None:
                # xacro_tld = base_path / PurePosixPath(xacro_tld)
            # urdf_string = xacro.main(file_path, xacro_tld)
            # try:
                # urdf = URDF.loadstr(urdf_string, file_path, base_path)
            # except BaseException as e:
                # print("error parsing URDF file", file_path)
                # raise e
        # else:  # pragma nocover
            # urdf_string = open(file_path).read()
            # urdf = URDF.loadstr(urdf_string, file_path, base_path)

        # return urdf.elinks, urdf.name, urdf_string, file_path

    # # --------------------------------------------------------------------- #
        
        

if __name__ == "__main__":  # pragma nocover

    r = TwoPandaWorld()
    print(r)
    r.qz

    for link in r.grippers[0].links:
        print(link)
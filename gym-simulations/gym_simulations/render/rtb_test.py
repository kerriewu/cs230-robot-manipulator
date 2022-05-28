# rtb_test.py

import time
import roboticstoolbox as rtb
robot = rtb.models.DH.Panda()
print(robot)

T = robot.fkine(robot.qz)  # forward kinematics
print(T)

from spatialmath import SE3

T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ikine_LM(T)         # solve IK
print(sol)
	

q_pickup = sol.q
print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved



qt = rtb.jtraj(robot.qz, q_pickup, 50)
# robot.plot(qt.q, movie='panda1.gif')


robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
print(robot)    # display the model



from roboticstoolbox.backends.swift import Swift  # instantiate 3D browser-based visualizer
backend = Swift()
backend.launch()            # activate it
backend.add(robot)          # add robot to the 3D scene
for i in range(100):
    for qk in qt.q:             # for each joint configuration on trajectory
          robot.q = qk          # update the robot state
          backend.step()        # update visualization
          time.sleep(.1)
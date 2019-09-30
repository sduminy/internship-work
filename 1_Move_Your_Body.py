from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
from os.path import join
from time import sleep
import numpy as np

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper.BuildFromURDF(URDF, [PKG])

robot.initDisplay(loadModel=True)

q = zero(robot.nq)
for i in range(6): 
    q[i] = 0.78
robot.display(q)


rgbt = [1.0, 0.2, 0.2, 1.0]  # red, green, blue, transparency
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)  # .1 is the radius

robot.viewer.gui.applyConfiguration("world/sphere", (.5, .1, .2, 1.,0.,0.,0. ))
robot.viewer.gui.refresh()  # Refresh the window.

t_list = np.linspace(0,1,100)

for t in t_list:
    for i in range(6): 
        q[i] = t*3.14
    e_e = robot.placement(q,6)
    robot.viewer.gui.applyConfiguration("world/sphere", (e_e.translation[0,0], e_e.translation[1,0], e_e.translation[2,0], 1.,0.,0.,0. ))
    robot.display(q)
    sleep(0.1)
    
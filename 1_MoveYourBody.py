import numpy as np
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
from os.path import join
from time import sleep

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper.BuildFromURDF(URDF, [PKG])

robot.initDisplay(loadModel=True)

q =  zero(robot.nq)
for i in range(robot.nq):
	q[i] = .78
robot.display(q)

rgbt = [1.0, 0.2, 0.2, 1.0]  #x, y, z, quaternion
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)  #.1 is the radius

robot.viewer.gui.applyConfiguration("world/sphere", (.5, .1, .2, 1., 0., 0., 0.))
robot.viewer.gui.refresh()  #refresh the window

for t in np.linspace(0, 1, 100):
	for i in range(6):
		q[i] = t*3.14
	e_e = robot.placement(q, 6)
	robot.viewer.gui.applyConfiguration("world/sphere", (e_e.translation[0, 0], e_e.translation[1, 0], e_e.translation[2, 0], 1, 0, 0, 0))
	robot.display(q)
	sleep(0.1)


import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from mobilerobot import MobileRobotWrapper
from os.path import join
from time import sleep
from IPython import embed
from numpy.linalg import pinv


PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')

robot = MobileRobotWrapper.BuildFromURDF(URDF, [PKG])

robot.initDisplay(loadModel=True)

robot.display(robot.q0)

NQ = robot.model.nq
NV = robot.model.nv

### Set up display environment
def place(name, M):
	robot.viewer.gui.applyConfiguration(name, se3ToXYZQUAT(M))
	robot.viewer.gui.refresh()

def Rquat(x, y, z, w):
	q = pin.Quaternion(x, y, z, w)
	q.normalize()
	return q.matrix()

Mgoal = pin.SE3(Rquat(.4, .02, -.5, .7), np.matrix([.2, -.4, .7]).T)
robot.viewer.gui.addXYZaxis('world/framegoal', [1., 0., 0., 1.], .015, 4)
place('world/framegoal', Mgoal)
place('world/yaxis', pin.SE3(rotate('x', np.pi/2), np.matrix([0, 0, .1]).T))

IDX_TOOL = robot.model.getFrameId('tool0')
pin.forwardKinematics(robot.model, robot.data, robot.q0)
pin.updateFramePlacements(robot.model, robot.data)
Mtool = robot.data.oMf[IDX_TOOL]

q = robot.q0
dt = 1e-2	#Integration step

"""
### Loop on an inverse kinematics for 200 iterations
for i in range(200):
	pin.forwardKinematics(robot.model, robot.data, q)
	pin.updateFramePlacements(robot.model, robot.data)
	Mtool = robot.data.oMf[IDX_TOOL]
	J = pin.frameJacobian(robot.model, robot.data, q, IDX_TOOL)
	nu = pin.log(Mtool.inverse() * Mgoal).vector
	vq = pinv(J)*nu	
	q = pin.integrate(robot.model, q, vq * dt)
	robot.display(q)
	sleep(dt)"""

### Position the basis on the line
robot.viewer.gui.addCylinder('world/yaxis', .01, 20, [.1, .1, .1, 1.])
place('world/yaxis', pin.SE3(rotate('x', np.pi/2), np.matrix([0, 0, .1]).T))

IDX_BASIS = robot.model.getFrameId('base')

q_basis = q

for j in range(200):
	pin.forwardKinematics(robot.model, robot.data, q_basis)
	pin.updateFramePlacements(robot.model, robot.data)
	Mbasis = robot.data.oMf[IDX_BASIS]
	error = Mbasis.translation[0]-1
	J_basis = pin.frameJacobian(robot.model, robot.data, q, IDX_BASIS)[0,:]
	nu_basis = error
	vq_basis = pinv(J_basis)*nu_basis
	q_basis = pin.integrate(robot.model, q_basis, vq_basis * dt)
	robot.display(q_basis)
	#sleep(dt)
embed()


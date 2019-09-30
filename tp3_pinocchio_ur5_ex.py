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

NQ = robot.model.nq
NV = robot.model.nv


q = rand(NQ)
vq = rand(NV)

robot.display(q)

from time import sleep
embed()
for i in range(100):
    q = pin.integrate(robot.model, q, vq / 100)
    robot.display(q)
    sleep(.01)
    print q.T

IDX_TOOL = 24
IDX_BASIS = 23

pin.updateFramePlacements(robot.model, robot.data)
Mtool = robot.data.oMf[IDX_TOOL]
Mbasis = robot.data.oMf[IDX_BASIS]

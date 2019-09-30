### Robot mobile ###

import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from os.path import join
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.robot_wrapper import RobotWrapper
from scipy.optimize import fmin_bfgs, fmin_slsqp
from time import sleep
from IPython import embed  #embed makes the programm runs till embed()

# Loading the robot
PKG = '/opt/openrobots/share'
#URDF = join(PKG, 'romeo_description/urdf/romeo.urdf')
URDF = join(PKG, 'romeo_description/urdf/romeo_small.urdf')
robot = RomeoWrapper.BuildFromURDF(URDF, [PKG])
print 'nq = ', robot.model.nq
print 'nv = ', robot.model.nv


# Display default position of the robot
robot.initDisplay(loadModel=True)


# Cost function
def cost(x):
    pin.forwardKinematics(robot.model, robot.data, np.matrix(x).T)
    p = robot.data.oMi[-1].translation
    norm_diff = np.linalg.norm(pdes - p.getA()[:, 0])
    return norm_diff


# Class my_CallbackLogger
class my_CallbackLogger:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x[0], x[1], x[2], x[3], x[4], x[5], cost(x)))
       q = np.matrix(x).T
       robot.display(q)
       sleep(.5)
       self.nfeval += 1

embed()
x0 = zero(robot.nq)
x0 = x0.getA()[:, 0]

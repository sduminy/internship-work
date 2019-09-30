import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from os.path import join
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.robot_wrapper import RobotWrapper
from scipy.optimize import fmin_bfgs, fmin_slsqp
from time import sleep

# From 1-dimensional array for Scipy to matrix for Pinocchio and the other way
x = np.array([1.0, 2.0, 3.0])
q = np.matrix(x).T
x = q.getA()[:, 0]

PKG = '/opt/openrobots/share'
"""URDF = join(PKG, 'romeo_description/urdf/romeo.urdf')
robot = RomeoWrapper.BuildFromURDF(URDF, [PKG])"""
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper.BuildFromURDF(URDF, [PKG])


# Target location
pdes = np.array([.5, .1, 0.0])

# Display default position of the robot
robot.initDisplay(loadModel=True)

# Display sphere
rgbt = [0.2, 1.0, 0.2, 1.0]  # red, green, blue, transparency
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)  # .1 is the radius
robot.viewer.gui.applyConfiguration("world/sphere", (pdes[0], pdes[1], pdes[2], 1.,0.,0.,0. ))
robot.viewer.gui.refresh()  # Refresh the window.

def cost(x):
    pin.forwardKinematics(robot.model, robot.data, np.matrix(x).T)
    p = robot.data.oMi[-1].translation  #index -1 : end_effector
    norm_diff = np.linalg.norm(pdes - p.getA()[:, 0])	#p = se3 object
    return norm_diff

class my_CallbackLogger:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x[0], x[1], x[2], x[3], x[4], x[5], cost(x)))
       q = np.matrix(x).T
       robot.display(q)
       sleep(.5)
       self.nfeval += 1

x0 = zero(robot.nq)
x0 = x0.getA()[:, 0]

"""
# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(cost, x0, callback=my_CallbackLogger())
print('*** Xopt in BFGS =\n', xopt_bfgs)

# Display result
q = np.matrix(xopt_bfgs).T
robot.display(q)
print('Result displayed')

"""
### Placing the end effector using log ###

from pinocchio.explog import log
from pinocchio import SE3
"""
pdes = zero(robot.nq)
for i in range(robot.nq): 
    pdes[i] = .78
pin.forwardKinematics(robot.model, robot.data, pdes)
pdes = robot.data.oMi[-1].copy()
pin.forwardKinematics(robot.model, robot.data, zero(robot.nq))"""

pdes = pin.SE3.Random()
pdes.translation = np.array([ [.5], [.1], [0.0] ])
pdes.rotation = np.array([ [1,0,0],[0,1,0],[0,0,1] ])

def cost_log(x):
    pin.forwardKinematics(robot.model, robot.data, np.matrix(x).T)
    p = robot.data.oMi[-1].copy()
    log_diff = log(p.inverse()*pdes)
    norm_log_diff = np.linalg.norm(log_diff.vector)
    return norm_log_diff

class my_CallbackLogger_log:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x[0], x[1], x[2], x[3], x[4], x[5], cost_log(x)))
       q = np.matrix(x).T
       robot.display(q)
       sleep(.5)
       self.nfeval += 1

x0 = zero(robot.nq)
x0 = x0.getA()[:, 0]

# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(cost_log, x0, callback=my_CallbackLogger_log())
print('*** Xopt in BFGS =\n', xopt_bfgs)

# Display result
q = np.matrix(xopt_bfgs).T
robot.display(q)
print('Result displayed')

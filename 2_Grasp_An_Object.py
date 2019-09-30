# Example of use a the optimization toolbox of SciPy.

##
## TUTORIAL
##

import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
 
def cost(x):
    '''Cost f(x, y) = x^2 + 2y^2 - 2xy - 2x '''
    x0, x1 = x
    return -(2 * x0 * x1 + 2 * x0 - x0 ** 2 - 2 * x1 ** 2)

def constraint_eq(x):
    ''' Constraint x^3 = y '''
    return np.array([ x[0] ** 3 - x[1] ])
   
def constraint_ineq(x):
    '''Constraint x >= 2, y >= 2'''
    return np.array([ x[0] - 2, x[1] - 2 ])
   
class CallbackLogger:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f}'.format(self.nfeval, x[0], x[1], cost(x)))
       self.nfeval += 1
 
x0 = np.array([0.0, 0.0])  

"""
# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(cost, x0, callback=CallbackLogger())
print('*** Xopt in BFGS =', xopt_bfgs)
 
# Optimize cost without any constraints in CLSQ
xopt_lsq = fmin_slsqp(cost, [-1.0, 1.0], iprint=2, full_output=1)
print('*** Xopt in LSQ =', xopt_lsq)
 
# Optimize cost with equality and inequality constraints in CLSQ
xopt_clsq = fmin_slsqp(cost, [-1.0, 1.0], f_eqcons=constraint_eq, f_ieqcons=constraint_ineq, iprint=2, full_output=1)
print('*** Xopt in c-lsq =', xopt_clsq)
"""

##
## QUESTION
##

# From 1-dimensional array for Scipy to matrix for Pinocchio and the other way
x = np.array([1.0, 2.0, 3.0])
q = np.matrix(x).T
x = q.getA()[:, 0]

from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
from os.path import join
from time import sleep

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper.BuildFromURDF(URDF, [PKG])

# Target location
pdes = np.array([.5, .1, 0.0])

# Display default position of the robot
robot.initDisplay(loadModel=True)

# Display sphere
rgbt = [0.2,0.1,0.1, 1.0]  # red, green, blue, transparency
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)  # .1 is the radius
robot.viewer.gui.applyConfiguration("world/sphere", (pdes[0], pdes[1], pdes[2], 1.,0.,0.,0. ))
robot.viewer.gui.refresh()  # Refresh the window.


def my_cost(x):
    pin.forwardKinematics(robot.model, robot.data, np.matrix(x).T)
    p = robot.data.oMi[-1].translation
    norm_diff = np.linalg.norm(pdes - p.getA()[:, 0])
    norm_post = np.linalg.norm(np.array([-3.1415/2]) - x[3])
    return (100*norm_diff+norm_post)

class my_CallbackLogger:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x[0], x[1], x[2], x[3], x[4], x[5], my_cost(x)))
       q = np.matrix(x).T
       robot.display(q)
       sleep(0.25)
       self.nfeval += 1

x0 = zero(robot.nq)
x0 = x0.getA()[:, 0]

# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(my_cost, x0, callback=my_CallbackLogger())
print('*** Xopt in BFGS =', xopt_bfgs)

# Display result
q = np.matrix(xopt_bfgs).T
robot.display(q)
print('Result displayed')
"""

##
## QUESTION WITH LOG
##

pdes = zero(robot.nq)
for i in range(6): 
    pdes[i] = 0.78
pin.forwardKinematics(robot.model, robot.data, pdes)
pdes = robot.data.oMi[-1].copy()
pin.forwardKinematics(robot.model, robot.data, zero(robot.nq))

from pinocchio.explog import log
def my_cost_log(x):
    pin.forwardKinematics(robot.model, robot.data, np.matrix(x).T)
    log_current = log(pdes.inverse()*robot.data.oMi[-1].copy())
    return np.linalg.norm(log_current.vector)

class my_CallbackLogger_log:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x[0], x[1], x[2], x[3], x[4], x[5], my_cost_log(x)))
       q = np.matrix(x).T
       robot.display(q)
       sleep(0.25)
       self.nfeval += 1

x0 = zero(robot.nq)
x0 = x0.getA()[:, 0]

# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(my_cost_log, x0, callback=my_CallbackLogger_log())
print('*** Xopt in BFGS =', xopt_bfgs)

# Display result
q = np.matrix(xopt_bfgs).T
robot.display(q)
print('Result displayed')"""


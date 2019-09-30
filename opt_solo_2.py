from pinocchio.utils import *
import robots_loader
import numpy as np
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_slsqp
from time import sleep
from IPython import embed

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

sq = solo.q0

qdes = sq.copy() 	#reference configuration

solo.display(sq)

sleep(2)

sq[3]=np.random.random()/10
sq[4]=-np.random.random()/10

solo.display(sq)

embed()

def cost(x):
    pin.forwardKinematics(solo.model, solo.data, np.concatenate((sq[:7], np.matrix(x).T)))
    solo.framesForwardKinematics(np.concatenate((sq[:7], np.matrix(x).T)))
    # Getting the frame index of each foot
    ID_FL = solo.model.getFrameId("FL_FOOT")
    ID_FR = solo.model.getFrameId("FR_FOOT")
    ID_HL = solo.model.getFrameId("HL_FOOT")
    ID_HR = solo.model.getFrameId("HR_FOOT")
    # Getting the current height (on axis z) of each foot
    hFL = solo.data.oMf[ID_FL].translation[2].getA()[:, 0]
    hFR = solo.data.oMf[ID_FR].translation[2].getA()[:, 0]
    hHL = solo.data.oMf[ID_HL].translation[2].getA()[:, 0]
    hHR = solo.data.oMf[ID_HR].translation[2].getA()[:, 0]
    # Calculating the norm of the position of each foot regarding the first one  
    norm_diff_FL_FR = np.linalg.norm(hFL - hFR)
    norm_diff_FL_HL = np.linalg.norm(hFL - hHL)
    norm_diff_FL_HR = np.linalg.norm(hFL - hHR)
    # Making sure that Solo's configuration stays near the reference one
    norm_diff_q = np.linalg.norm(qdes.getA()[:,0] - sq.getA()[:, 0])
    return (norm_diff_FL_FR + norm_diff_FL_HL + norm_diff_FL_HR + 10e-3*norm_diff_q)

class my_CallbackLogger:
    def __init__(self):
        self.nfeval = 1
 
    def __call__(self,x):
       '''print('===CBK=== {0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f}'.format(self.nfeval, x, cost(x)))'''
       q = np.concatenate((sq[:7], np.matrix(x).T))
       solo.display(q)
       sleep(.5)
       self.nfeval += 1

x0 = sq[7:].getA()[:, 0]

# Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(cost, x0, callback=my_CallbackLogger())
print('*** Xopt in BFGS =\n', xopt_bfgs)

# Display result
q = np.concatenate((sq[:7], np.matrix(xopt_bfgs).T))
solo.display(q)
print('Result displayed')

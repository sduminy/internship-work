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

solo.display(sq)

sleep(2)

sq[3]=np.random.random()/10
sq[4]=-np.random.random()/10

solo.display(sq)

pdesFL = np.matrix(0)
pdesFR = np.matrix(0)
pdesHL = np.matrix(0)
pdesHR = np.matrix(0)

embed()

def cost(x):
    pin.forwardKinematics(solo.model, solo.data, np.concatenate((sq[:7], np.matrix(x).T)))
    solo.framesForwardKinematics(np.concatenate((sq[:7], np.matrix(x).T)))
    ID_FL = solo.model.getFrameId("FL_FOOT")
    ID_FR = solo.model.getFrameId("FR_FOOT")
    ID_HL = solo.model.getFrameId("HL_FOOT")
    ID_HR = solo.model.getFrameId("HR_FOOT")
    pFL = solo.data.oMf[ID_FL].translation[2]
    pFR = solo.data.oMf[ID_FR].translation[2]
    pHL = solo.data.oMf[ID_HL].translation[2]
    pHR = solo.data.oMf[ID_HR].translation[2]  
    norm_diff_FL = np.linalg.norm(pdesFL - pFL.getA()[:, 0])
    norm_diff_FR = np.linalg.norm(pdesFR - pFR.getA()[:, 0])
    norm_diff_HL = np.linalg.norm(pdesHL - pHL.getA()[:, 0])
    norm_diff_HR = np.linalg.norm(pdesHR - pHR.getA()[:, 0])
    return (norm_diff_FL + norm_diff_FR + norm_diff_HL + norm_diff_HR)

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




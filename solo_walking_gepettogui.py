import robots_loader
import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from time import sleep
from IPython import embed
from numpy.linalg import pinv
from math import sin

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

q = solo.q0

qdes = q.copy() 	#reference configuration

solo.display(q)

### Spacemouse configuration
import spacenav as sp
import atexit

try:
	# open the connection
	print("Opening connection to SpaceNav driver ...")
	sp.open()
	print("... connection established.")
	# register the close function if no exception was raised
	atexit.register(sp.close)
except sp.ConnectionError:
	# give some user advice if the connection failed 
	print("No connection to the SpaceNav driver. Is spacenavd running?")
	sys.exit(-1)

# reset exit condition
stop = False

# Jacobian of posture task
J_post = np.eye(8)

# coefficient which minimizes the cost of the posture task
omega = 10e-3

dt = 1e-2	# Integration step

# Convergence gain
Kp = 1

# Getting the frame index of each foot
ID_FL = solo.model.getFrameId("FL_FOOT")
ID_FR = solo.model.getFrameId("FR_FOOT")
ID_HL = solo.model.getFrameId("HL_FOOT")
ID_HR = solo.model.getFrameId("HR_FOOT")

hist_err = [] 	#history of the error

# loop over space navigator events
while not stop:
	### Space mouse configuration : exit the loop when 0 is pressed and released
	# return the next event if there is any
	event = sp.poll()

	# if event signals the release of the first button
	if type(event) is sp.ButtonEvent \
		and event.button == 0 and event.pressed == 0:
		# set exit condition
		stop = True
		
	### Stack of Task : walking
	pin.forwardKinematics(solo.model, solo.data, q)
	pin.updateFramePlacements(solo.model, solo.data)
	
	# Getting the current height (on axis z) and the x-coordinate of each foot
	xzFL = solo.data.oMf[ID_FL].translation[0::2]
	xzFR = solo.data.oMf[ID_FR].translation[0::2]
	xzHL = solo.data.oMf[ID_HL].translation[0::2]
	xzHR = solo.data.oMf[ID_HR].translation[0::2]
	
	# Computing the error in the world frame
	err_FL = np.concatenate((np.zeros([1,1]),hyFL))
	err_FR = np.concatenate((np.zeros([1,1]),hyFR))
	err_HL = np.concatenate((np.zeros([1,1]),hyHL))
	err_HR = np.concatenate((np.zeros([1,1]),hyHR))
	err_FLy = solo.data.oMf[ID_FL].translation[1] - yFL0 - 0.005
	
	# Moving 5cm forward
	#err_FL[1] = 0.05
	err_FR[1] = 0.05
	err_HL[1] = 0.05
	err_HR[1] = 0.05
	
	# Error of posture
	err_post = q - qdes
	
	# Computing the error in the local frame
	oR_FL = solo.data.oMf[ID_FL].rotation
	oR_FR = solo.data.oMf[ID_FR].rotation
	oR_HL = solo.data.oMf[ID_HL].rotation
	oR_HR = solo.data.oMf[ID_HR].rotation
	
	# Getting the different Jacobians
	fJ_FL3 = pin.frameJacobian(solo.model, solo.data, q, ID_FL)[:3,-8:]	#Take only the translation terms
	oJ_FL3 = oR_FL*fJ_FL3	#Transformation from local frame to world frame
	oJ_FLyz = oJ_FL3[1:3,-8:]	#Take the y and z components
	
	fJ_FR3 = pin.frameJacobian(solo.model, solo.data, q, ID_FR)[:3,-8:]
	oJ_FR3 = oR_FR*fJ_FR3
	oJ_FRyz = oJ_FR3[1:3,-8:]
	
	fJ_HL3 = pin.frameJacobian(solo.model, solo.data, q, ID_HL)[:3,-8:]
	oJ_HL3 = oR_HL*fJ_HL3
	oJ_HLyz = oJ_HL3[1:3,-8:]
	
	fJ_HR3 = pin.frameJacobian(solo.model, solo.data, q, ID_HR)[:3,-8:]
	oJ_HR3 = oR_HR*fJ_HR3
	oJ_HRyz = oJ_HR3[1:3,-8:]
	
	# Displacement and posture error
	#nu = np.vstack([err_FL[1:3],err_FR[1:3], err_HL[1:3], err_HR[1:3], omega*err_post[7:]])
	#nu = np.vstack([err_FLy, err_FL[2]])
	nu = err_FLy
	
	# Making a single z-row Jacobian vector plus the posture Jacobian
	#J = np.vstack([oJ_FLyz, oJ_FRyz, oJ_HLyz, oJ_HRyz, omega*J_post])
	#J = np.vstack([oJ_FLyz[0,:], oJ_FLyz[1,:]])
	J = oJ_FL3[0,-8:]
	
	# Computing the velocity
	vq_act = -Kp*pinv(J)*nu
	vq = np.concatenate((np.zeros([6,1]) , vq_act))
	
	# Computing the updated configuration
	q = pin.integrate(solo.model, q, vq * dt)
	
	solo.display(q)
	
	if err_FL[1]<10e-6 : #and err_FL[0]<10e-6:
		stop = True
	
	#hist_err.append(np.linalg.norm(nu))	
	

'''
### Plot the norm of the error during time
import matplotlib.pylab as plt
plt.ion()
plt.plot(hist_err)
plt.grid()
plt.title('Error course vs time')
'''

embed()

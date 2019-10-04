import robots_loader
import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from time import sleep
from IPython import embed
from numpy.linalg import pinv
from math import atan2, asin, cos, sin


solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

q = solo.q0

qdes = q.copy() 	#reference configuration

solo.display(q)

def normalize(quaternion):
	norm_q = np.linalg.norm(quaternion)
	assert(norm_q>1e-6)
	return quaternion/norm_q

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

# initialize the Row-Pitch-Yaw
rpy = np.zeros(3)

dt = 1e-2	# Integration step

# Convergence gain
Kp = 30

# Getting the frame index of each foot
ID_FL = solo.model.getFrameId("FL_FOOT")
ID_FR = solo.model.getFrameId("FR_FOOT")
ID_HL = solo.model.getFrameId("HL_FOOT")
ID_HR = solo.model.getFrameId("HR_FOOT")

hist_err = [] 	#history of the error

# loop over space navigator events
while not stop:
	### Space mouse configuration
	# return the next event if there is any
	event = sp.poll()

	# if event signals the release of the first button
	if type(event) is sp.ButtonEvent \
		and event.button == 0 and event.pressed == 0:
		# set exit condition
		stop = True
		
	# matching the mouse signals with the position of Solo's basis
	if type(event) is sp.MotionEvent :
		Vroll = -event.rx/100.0		#velocity related to the roll axis
		Vyaw = event.ry/100.0		#velocity related to the yaw axis
		Vpitch = -event.rz/100.0	#velocity related to the pitch axis
		rpy[0] += Vroll*dt		#roll
		rpy[1] += Vpitch*dt		#pitch
		rpy[2] += Vyaw*dt		#yaw
		# adding saturation to prevent unlikely configurations
		for i in range(2):
			if rpy[i]>0.175 or rpy[i]<-0.175:
				rpy[i] = np.sign(rpy[i])*0.175
		# convert rpy to quaternion
		quatMat = pin.utils.rpyToMatrix(np.matrix(rpy).T)
		q[3] = Quaternion(quatMat)[0]
		q[4] = Quaternion(quatMat)[1]
		q[5] = Quaternion(quatMat)[2]
		q[6] = Quaternion(quatMat)[3]
		
	### Stack of Task : feet on the ground and posture
	pin.forwardKinematics(solo.model, solo.data, q)
	pin.updateFramePlacements(solo.model, solo.data)
	
	# Getting the current height (on axis z) of each foot
	hFL = solo.data.oMf[ID_FL].translation[2]
	hFR = solo.data.oMf[ID_FR].translation[2]
	hHL = solo.data.oMf[ID_HL].translation[2]
	hHR = solo.data.oMf[ID_HR].translation[2]
	
	# Computing the error in the world frame
	err_FL = np.concatenate((np.zeros([2,1]),hFL))
	err_FR = np.concatenate((np.zeros([2,1]),hFR))
	err_HL = np.concatenate((np.zeros([2,1]),hHL))
	err_HR = np.concatenate((np.zeros([2,1]),hHR))
	
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
	oJ_FLz = oJ_FL3[2,-8:]	#Take the z_component
	
	fJ_FR3 = pin.frameJacobian(solo.model, solo.data, q, ID_FR)[:3,-8:]
	oJ_FR3 = oR_FR*fJ_FR3
	oJ_FRz = oJ_FR3[2,-8:]
	
	fJ_HL3 = pin.frameJacobian(solo.model, solo.data, q, ID_HL)[:3,-8:]
	oJ_HL3 = oR_HL*fJ_HL3
	oJ_HLz = oJ_HL3[2,-8:]
	
	fJ_HR3 = pin.frameJacobian(solo.model, solo.data, q, ID_HR)[:3,-8:]
	oJ_HR3 = oR_HR*fJ_HR3
	oJ_HRz = oJ_HR3[2,-8:]
	
	# Displacement and posture error
	nu = np.vstack([err_FL[2],err_FR[2], err_HL[2], err_HR[2], omega*err_post[7:]])
	
	# Making a single z-row Jacobian vector plus the posture Jacobian
	J = np.vstack([oJ_FLz, oJ_FRz, oJ_HLz, oJ_HRz, omega*J_post])
	
	# Computing the velocity
	vq_act = -Kp*pinv(J)*nu
	vq = np.concatenate( (np.zeros([6,1]) , vq_act))
	
	# Computing the updated configuration
	#q[1:] = q[1:] + vq*dt
	q = pin.integrate(solo.model, q, vq * dt)
	
	solo.display(q)
	
	#hist_err.append(np.linalg.norm(nu))	
	hist_err.append(err_post[7:])
	

'''
### Plot the norm of the error during time
import matplotlib.pylab as plt
plt.ion()
plt.plot(hist_err)
plt.grid()
plt.title('Error course vs time')
'''

embed()

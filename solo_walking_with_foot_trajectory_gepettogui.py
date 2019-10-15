import robots_loader
import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from time import sleep, time
from IPython import embed
from numpy.linalg import pinv
from math import sin, cos

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

q = solo.q0

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

dt = 1e-3	# Integration step
t = 0		# Starting time of the simulation

# Convergence gain
Kp = 100.

# Initial condition
# Front feet
xF0 = 0.19 	
zF0 = 0
# Hind feet
xH0 = -0.19 	
zH0 = 0
# Feet trajectory parameters
T = 0.5	#period of 0.5s
dx = 0.02
dz = 0.05

def ftraj(t, x0, z0):	#arguments : time, initial position x and z
	global T, dx, dz
	x = []
	z = []
	if t >= T:
		t %= T
	if t <= T/2.:
		x.append(x0-dx*cos(2*np.pi*t/T))
		z.append(z0+dz*sin(2*np.pi*t/T))
	else:
		x.append(x0+3*dx-4*dx*dt/T)
		z.append(0)
	return np.matrix([x,z])

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
	
	# Getting the current height (on axis z) and the x-coordinate of the front left foot
	xz_FL = solo.data.oMf[ID_FL].translation[0::2]
	xz_FR = solo.data.oMf[ID_FR].translation[0::2]
	xz_HL = solo.data.oMf[ID_HL].translation[0::2]
	xz_HR = solo.data.oMf[ID_HR].translation[0::2]
	
	# Desired foot trajectory
	t += dt
	xzdes_FL = ftraj(t, xF0, zF0)
	xzdes_HR = ftraj(t, xH0, zH0)
	xzdes_FR = ftraj(t+T/2, xF0, zF0)
	xzdes_HL = ftraj(t+T/2, xH0, zH0)
	
	# Calculating the error
	err_FL = xz_FL - xzdes_FL
	err_FR = xz_FR - xzdes_FR
	err_HL = xz_HL - xzdes_HL
	err_HR = xz_HR - xzdes_HR
	
	# Computing the local Jacobian into the global frame
	oR_FL = solo.data.oMf[ID_FL].rotation
	oR_FR = solo.data.oMf[ID_FR].rotation
	oR_HL = solo.data.oMf[ID_HL].rotation
	oR_HR = solo.data.oMf[ID_HR].rotation
	
	# Getting the different Jacobians
	fJ_FL3 = pin.frameJacobian(solo.model, solo.data, q, ID_FL)[:3,-8:]	#Take only the translation terms
	oJ_FL3 = oR_FL*fJ_FL3	#Transformation from local frame to world frame
	oJ_FLxz = oJ_FL3[0::2,-8:]	#Take the x and z components
	
	fJ_FR3 = pin.frameJacobian(solo.model, solo.data, q, ID_FR)[:3,-8:]
	oJ_FR3 = oR_FR*fJ_FR3
	oJ_FRxz = oJ_FR3[0::2,-8:]
	
	fJ_HL3 = pin.frameJacobian(solo.model, solo.data, q, ID_HL)[:3,-8:]
	oJ_HL3 = oR_HL*fJ_HL3
	oJ_HLxz = oJ_HL3[0::2,-8:]
	
	fJ_HR3 = pin.frameJacobian(solo.model, solo.data, q, ID_HR)[:3,-8:]
	oJ_HR3 = oR_HR*fJ_HR3
	oJ_HRxz = oJ_HR3[0::2,-8:]
	
	# Displacement error
	nu = np.vstack([err_FL, err_FR, err_HL, err_HR])
	
	# Making a single x&z-rows Jacobian vector 
	J = np.vstack([oJ_FLxz, oJ_FRxz, oJ_HLxz, oJ_HRxz])
	
	# Computing the velocity
	vq_act = -Kp*pinv(J)*nu
	vq = np.concatenate((np.zeros([6,1]) , vq_act))
	
	# Computing the updated configuration
	q = pin.integrate(solo.model, q, vq * dt)
	
	solo.display(q)
	
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

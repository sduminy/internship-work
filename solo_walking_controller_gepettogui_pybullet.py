# coding: utf8

import pybullet as p # PyBullet simulator
import time # Time module to sleep()
import pybullet_data
import pinocchio as pin       # Pinocchio library
import numpy as np # Numpy library
import robots_loader # Functions to load the SOLO quadruped
from pinocchio.utils import * # Utilitary functions from Pinocchio
from pinocchio.robot_wrapper import RobotWrapper # Robot Wrapper to load an URDF in Pinocchio
from os.path import dirname, exists, join # OS function for path manipulation
from scipy.optimize import fmin_slsqp # Optimization function to minimize a cost

########################################################################

from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from time import sleep, time
from IPython import embed
from numpy.linalg import pinv
from math import sin, cos

### Spacemouse configuration
import spacenav as sp
import atexit

# function defining the feet's trajectory

def ftraj(t, x0, z0):	#arguments : time, initial position x and z
	global T, dx, dz
	x = []
	z = []
	if t >= T:
		t %= T
	x.append(x0-dx*cos(2*np.pi*t/T))
	if t <= T/2.:
		z.append(z0+dz*sin(2*np.pi*t/T))
	else:
		z.append(0)
	return np.matrix([x,z])
	
########################################################################

h = 1./240. # Time step of the simulation
pi = np.pi # Value of pi
v_prev = np.zeros((14,1)) # velocity during the previous time step, of size (robot.nv,1)

# To get the state of link i (position and velocity), use  p.getLinkState(robotId, linkIndex=i, computeLinkVelocity=1)

# Function to normalize a quarternion
def normalize(quaternion):
    norm_q = np.linalg.norm(quaternion)
    assert(norm_q>1e-6)
    return quaternion/norm_q

## Function called from the main loop of the simulation ##
def callback_torques():
    global v_prev, solo, stop, q, qdes, t
    
    t_start = time()
	
    jointStates = p.getJointStates(robotId, revoluteJointIndices) # State of all joints
    baseState   = p.getBasePositionAndOrientation(robotId)
    baseVel = p.getBaseVelocity(robotId)

    # Info about contact points with the ground
    contactPoints_FL = p.getContactPoints(robotId, planeId, linkIndexA=2)  # Front left  foot 
    contactPoints_FR = p.getContactPoints(robotId, planeId, linkIndexA=5)  # Front right foot 
    contactPoints_HL = p.getContactPoints(robotId, planeId, linkIndexA=8)  # Hind  left  foot 
    contactPoints_HR = p.getContactPoints(robotId, planeId, linkIndexA=11) # Hind  right foot 

    # Sort contacts points to get only one contact per foot
    contactPoints = []
    contactPoints.append(getContactPoint(contactPoints_FL))
    contactPoints.append(getContactPoint(contactPoints_FR))
    contactPoints.append(getContactPoint(contactPoints_HL))
    contactPoints.append(getContactPoint(contactPoints_HR))

    # Joint vector for Pinocchio
    q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(), np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
    v = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(), np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))
    v_dot = (v-v_prev)/h
    v_prev = v.copy()
    
    # Update display in Gepetto-gui
    solo.display(q)

########################################################################
    
    ### Space mouse configuration : exit the loop when 0 is pressed and released
    
    event = sp.poll()	# return the next event if there is any

    # if event signals the release of the first button
    # exit condition : the button 0 is pressed and released
    if type(event) is sp.ButtonEvent and event.button == 0 and event.pressed == 0:
        # set exit condition
        stop = True
 
    ### Stack of Task : walking
    #compute/update all the joints and frames
    pin.forwardKinematics(solo.model, solo.data, qdes)
    pin.updateFramePlacements(solo.model, solo.data)
    
    # Getting the current height (on axis z) and the x-coordinate of the front left foot
    xz_FL = solo.data.oMf[ID_FL].translation[0::2]
    xz_FR = solo.data.oMf[ID_FR].translation[0::2]
    xz_HL = solo.data.oMf[ID_HL].translation[0::2]
    xz_HR = solo.data.oMf[ID_HR].translation[0::2]
    
    # Desired foot trajectory
    t1 = t	#previous time
    t += dt
    t2 = t	#current time
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
    fJ_FL3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_FL)[:3,-8:]	#Take only the translation terms
    oJ_FL3 = oR_FL*fJ_FL3	#Transformation from local frame to world frame
    oJ_FLxz = oJ_FL3[0::2,-8:]	#Take the x and z components
    
    fJ_FR3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_FR)[:3,-8:]
    oJ_FR3 = oR_FR*fJ_FR3
    oJ_FRxz = oJ_FR3[0::2,-8:]
    
    fJ_HL3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_HL)[:3,-8:]
    oJ_HL3 = oR_HL*fJ_HL3
    oJ_HLxz = oJ_HL3[0::2,-8:]
    
    fJ_HR3 = pin.frameJacobian(solo.model, solo.data, qdes, ID_HR)[:3,-8:]
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
    qdes = pin.integrate(solo.model, qdes, vq * dt)
	
    #hist_err.append(np.linalg.norm(nu))
    
    
########################################################################

    # PD Torque controller
    Kp_PD = 0.05
    Kd_PD = 120 * Kp_PD
    #Kd = 2 * np.sqrt(2*Kp) # formula to get a critical damping
    torques = Kd_PD * (qdes[7:] - q[7:]) - Kp_PD * v[6:]
    
    # Saturation to limit the maximal torque
    t_max = 5
    torques = np.maximum(np.minimum(torques, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
    
    # Control loop of 1/dt Hz
    while (time()-t_start)<dt :
		sleep(10e-6)
		
    return torques,stop


## Sort contacts points to get only one contact per foot ##
def getContactPoint(contactPoints):
    for i in range(0,len(contactPoints)):
        # There may be several contact points for each foot but only one of them as a non zero normal force
        if (contactPoints[i][9] != 0): 
            return contactPoints[i]
    return 0 # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen) 

########################################################################
########################### START OF SCRIPT ############################
########################################################################

# Load the robot for Pinocchio
solo = robots_loader.loadSolo(True)
solo.initDisplay(loadModel=True)

########################################################################

q = solo.q0

qdes = (solo.q0).copy()

## Initialization for control with mouse

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

t = 0
dt = h    # Integration step

# Convergence gain
Kp = 100.

# Getting the frame index of each foot
ID_FL = solo.model.getFrameId("FL_FOOT")
ID_FR = solo.model.getFrameId("FR_FOOT")
ID_HL = solo.model.getFrameId("HL_FOOT")
ID_HR = solo.model.getFrameId("HR_FOOT")

# Initial condition for the feet trajectory
# Front feet
xF0 = 0.19 	
zF0 = 0
# Hind feet
xH0 = -0.19 	
zH0 = 0
# Feet trajectory parameters
T = .5	#period of 0.5s
dx = 0.02
dz = 0.05

hist_err = []     #history of the error

########################################################################

# Start the client for PyBullet
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# Set gravity (disabled by default)
p.setGravity(0,0,-9.81)

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load Quadruped robot
robotStartPos = [0,0,0.3]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/solo_description/robots")
robotId = p.loadURDF("solo.urdf",robotStartPos, robotStartOrientation)
p.setTimeStep(dt)	# set the simulation time

'''
# Get the number of joints
numJoints = p.getNumJoints(robotId)
print("Number of joints: ", numJoints)

# Display information about each joint
for i in range(numJoints):
    print(p.getJointInfo(robotId,i))'''

# Disable default motor control for revolute joints
revoluteJointIndices = [0,1,3,4,6,7,9,10]
p.setJointMotorControlArray(robotId, 
                            jointIndices= revoluteJointIndices, 
                             controlMode= p.VELOCITY_CONTROL,
                       targetVelocities = [0.0 for m in revoluteJointIndices],
                                 forces = [0.0 for m in revoluteJointIndices])

# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]
p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

# Launch the simulation
while not stop:
   
    # Compute one step of simulation
    p.stepSimulation()
    #sleep(1./240.) # Default sleep duration, I think it's because the default step of the simulation is 1/240 second so we sleep 1/240 second to get a 1:1 ratio of simulation/real time

    # Callback Pinocchio to get joint torques
    jointTorques = callback_torques()

    # Set control torque for all joints
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques[0])


# Get final position and orientation of the robot
robotFinalPos, robotFinalOrientation = p.getBasePositionAndOrientation(robotId)
print(robotFinalPos, robotFinalOrientation)

embed()

# Shut down the client
p.disconnect()

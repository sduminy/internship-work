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
from time import sleep
from IPython import embed
from numpy.linalg import pinv

### Spacemouse configuration
import spacenav as sp
import atexit

########################################################################

h = 1./240. # Time step of the simulation
pi = 3.1415 # Value of pi
v_prev = np.zeros((14,1)) # velocity during the previous time step, of size (robot.nv,1)

# To get the state of link i (position and velocity), use  p.getLinkState(robotId, linkIndex=i, computeLinkVelocity=1)

# Function to normalize a quarternion
def normalize(quaternion):
    norm_q = np.linalg.norm(quaternion)
    assert(norm_q>1e-6)
    return quaternion/norm_q

## Function called from the main loop of the simulation ##
def callback_torques():
    global v_prev, solo, stop, q

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
    
    ### Space mouse configuration
    # return the next event if there is any
    event = sp.poll()

    # if event signals the release of the first button
    # exit condition : the button 0 is pressed and released
    if type(event) is sp.ButtonEvent \
        and event.button == 0 and event.pressed == 0:
        # set exit condition
        stop = True
        
    # matching the mouse signals with the position of Solo's basis
    if type(event) is sp.MotionEvent :
        Vroll = -event.rx/100.0        #velocity related to the roll axis
        Vyaw = event.ry/100.0          #velocity related to the yaw axis
        Vpitch = -event.rz/100.0       #velocity related to the pitch axis
        rpy[0] += Vroll*dt         #roll
        rpy[1] += Vpitch*dt        #pitch
        rpy[2] += Vyaw*dt          #yaw
        # adding saturation to prevent unlikely configurations
        for i in range(2):
            if rpy[i]>0.175 or rpy[i]<-0.175:
                rpy[i] = np.sign(rpy[i])*0.175
        # convert rpy to quaternion
        quatMat = pin.utils.rpyToMatrix(np.matrix(rpy).T)
        # add the modified component to the quaternion
        q[3] = Quaternion(quatMat)[0]
        q[4] = Quaternion(quatMat)[1]
        q[5] = Quaternion(quatMat)[2]
        q[6] = Quaternion(quatMat)[3]
        
    ### Stack of Task : feet on the ground and posture
    
    # compute/update all joints and frames
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
    fJ_FL3 = pin.frameJacobian(solo.model, solo.data, q, ID_FL)[:3,-8:]    #Take only the translation terms
    oJ_FL3 = oR_FL*fJ_FL3    #Transformation from local frame to world frame
    oJ_FLz = oJ_FL3[2,-8:]    #Take the z_component
    
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
    vq = np.concatenate((np.zeros([6,1]) , vq_act))
    
    # Computing the updated configuration
    q = pin.integrate(solo.model, q, vq * dt)
    
    #hist_err.append(np.linalg.norm(nu))
    #hist_err.append(err_post[7:])
    
########################################################################

    # PD Torque controller
    Kp_PD = 0.05
    Kd_PD = 80 * Kp_PD
    #Kd = 2 * np.sqrt(2*Kp) # formula to get a critical damping
    torques = Kd_PD * (qdes[7:] - q[7:]) - Kp_PD * v[6:]
    
    # Saturation to limit the maximal torque
    t_max = 5
    torques = np.maximum(np.minimum(torques, t_max * np.ones((8,1))), -t_max * np.ones((8,1))) 

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

## Initialization for control with mouse

qdes = (solo.q0).copy()

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

dt = h    # Integration step

# Convergence gain
Kp = 100

# Getting the frame index of each foot
ID_FL = solo.model.getFrameId("FL_FOOT")
ID_FR = solo.model.getFrameId("FR_FOOT")
ID_HL = solo.model.getFrameId("HL_FOOT")
ID_HR = solo.model.getFrameId("HR_FOOT")

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
robotStartPos = [0,0,0.35]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/solo_description/robots")
robotId = p.loadURDF("solo.urdf",robotStartPos, robotStartOrientation)

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
#for i in range (0,10000):
while not stop:
	   
    # Compute one step of simulation
    p.stepSimulation()
    time.sleep(1./240.) # Default sleep duration, I think it's because the default step of the simulation is 1/240 second so we sleep 1/240 second to get a 1:1 ratio of simulation/real time

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

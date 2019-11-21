# coding: utf8

import pybullet as p # PyBullet simulator
import time # Time module to sleep()
import pybullet_data
import pinocchio as pin       # Pinocchio library
import numpy as np # Numpy library
from pinocchio.utils import * # Utilitary functions from Pinocchio

########################################################################

import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm

import tsid

from IPython import embed

########################################################################

 
## Function called from the main loop of the simulation ##
def callback_torques():
    global init, sol, t, v_prev, q, qdes, vdes, trajCom, trajFLfoot, trajFRfoot, trajHLfoot, trajHRfoot, comTask, FLfootTask, FRfootTask, HLfootTask, HRfootTask

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
    v_dot = (v-v_prev)/dt
    v_prev = v.copy()
    
    ####################################################################
    
    pin.forwardKinematics(model, data, q)
    
    HL_foot_ref = robot.framePosition(data, model.getFrameId('HL_FOOT'))
    HR_foot_ref = robot.framePosition(data, model.getFrameId('HR_FOOT'))
    FL_foot_ref = robot.framePosition(data, model.getFrameId('FL_FOOT'))
    FR_foot_ref = robot.framePosition(data, model.getFrameId('FR_FOOT'))
    base = robot.framePosition(data, model.getFrameId('base_link'))
    
    tHL = HL_foot_ref.translation
    tHR = HR_foot_ref.translation
    tFL = FL_foot_ref.translation
    tFR = FR_foot_ref.translation
    tbase = base.translation
    
    ztFR = tFR[2,0]
    zGoal = tbase[2,0] - 0.235	# altitude objective : 0 from basis reference frame
    print(tbase[2,0])
    FR_foot_goal = FR_foot_ref.copy()
    FR_foot_goal.translation = np.matrix([tFR[0,0], tFR[1,0], zGoal]).T
    
    FL_foot_goal = FL_foot_ref.copy()
    FL_foot_goal.translation = np.matrix([tFL[0,0], tFL[1,0], zGoal]).T
    
    HR_foot_goal = HR_foot_ref.copy()
    HR_foot_goal.translation = np.matrix([tHR[0,0], tHR[1,0], zGoal]).T
    
    HL_foot_goal = HL_foot_ref.copy()
    HL_foot_goal.translation = np.matrix([tHL[0,0], tHL[1,0], zGoal]).T
    
    
    trajHLfoot = tsid.TrajectorySE3Constant("traj_HL_foot", HL_foot_goal)
    
    trajHRfoot = tsid.TrajectorySE3Constant("traj_HR_foot", HR_foot_goal)
    
    trajFLfoot = tsid.TrajectorySE3Constant("traj_FL_foot", FL_foot_goal)
    
    trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", FR_foot_goal)
   
    samplePosture = trajPosture.computeNext()
    postureTask.setReference(samplePosture)
    
    sampleHLfoot = trajHLfoot.computeNext()
    HLfootTask.setReference(sampleHLfoot)
    
    sampleHRfoot = trajHRfoot.computeNext()
    HRfootTask.setReference(sampleHRfoot)
    
    sampleFLfoot = trajFLfoot.computeNext()
    FLfootTask.setReference(sampleFLfoot)
    
    sampleFRfoot = trajFRfoot.computeNext()
    FRfootTask.setReference(sampleFRfoot)
    
    HQPData = invdyn.computeProblemData(t, qdes, vdes)
    
    sol = solver.solve(HQPData)
    
    tau = invdyn.getActuatorForces(sol)
    dv = invdyn.getAccelerations(sol)
    
    v_mean = vdes + 0.5*dt*dv
    vdes += dt*dv
    qdes = pin.integrate(model, qdes, dt*v_mean)
    t += dt
    
    robot_display.display(q)
    
    qlist.append(q)
    
    ####################################################################
    
    # PD Torque controller
    Kp_PD = 10.
    Kd_PD = 0.1
    #Kd = 2 * np.sqrt(2*Kp) # formula to get a critical damping
    torques = Kp_PD * (qdes[7:] - q[7:]) + Kd_PD * (vdes[6:] - v[6:])
    
    # Saturation to limit the maximal torque
    t_max = 5
    torques = np.maximum(np.minimum(torques, t_max * np.ones((12,1))), -t_max * np.ones((12,1)))
    
    return torques


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
qlist = []
## Definition of the tasks gains and weights

mu = 0.3  		# friction coefficient
fMin = 1.0		# minimum normal force
fMax = 100.0  	# maximum normal force

foot_frames = ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT']  # tab with all the foot frames names

w_posture = 0.1  	# weight of the posture task
w_foot = 10.0		# weight of the feet tasks
w_forceRef = 1e-3	# weight of force regularization task

kp_posture = 100.0  	# proportionnal gain of the posture task
kp_foot = 10.0	# proportionnal gain of the feet tasks


N_SIMULATION = 10000	# number of time steps simulated
dt = 0.001				# controller time step


## Set the path where the urdf and srdf file of the robot is registered

modelPath = "/opt/openrobots/lib/python2.7/site-packages/../../../share/example-robot-data"
urdf = modelPath + "/solo_description/robots/solo12.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)
# Create the robot wrapper from the urdf model
robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)


## Creation of the robot wrapper for the Gepetto Viewer

robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [modelPath, ], pin.JointModelFreeFlyer())
robot_display.initViewer(loadModel=True)


## Take the model of the robot and load its reference configuration

model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)
# Set the current configuration q to the robot configuration half_sitting
qdes = model.referenceConfigurations['straight_standing']
qdes[2] = 1.0
# Set the current velocity to zero
vdes = np.matrix(np.zeros(robot.nv)).T


## Display the robot in Gepetto Viewer

robot_display.displayCollisions(False)
robot_display.displayVisuals(True)
robot_display.display(qdes)


## Check that the frames of the feet exist

assert [robot.model().existFrame(name) for name in foot_frames]


t = 0.0  # time


## Disable the gravity

robot.set_gravity_to_zero()


## Creation of the Invverse Dynamics HQP problem using the robot
## accelerations (base + joints) and the contact forces

invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
# Compute th eproblem data with a solver based on EiQuadProg
invdyn.computeProblemData(t, qdes, vdes)
# Get the initial data
data = invdyn.data()


## Tasks definition

# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * matlib.ones(robot.nv-6).T) # Proportional gain 
postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(robot.nv-6).T) # Derivative gain 
# Add the task to the HQP with weight = 1e-3, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

# FEET Tasks

# Create a mask to keep only the translation coordinates
mask = matlib.zeros(6).T
mask[2] = 1.	# mask is [0 0 1 0 0 0] so that it will keep only the translational term by z

# HL foot
HLfootTask = tsid.TaskSE3Equality("HL-foot-grounded", robot, 'HL_FOOT')
HLfootTask.setKp(kp_foot * matlib.ones(6).T)
HLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
#HLfootTask.setMask(mask)
HLfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HLfootTask, w_foot, 1, 0.0)

# HR foot
HRfootTask = tsid.TaskSE3Equality("HR-foot-grounded", robot, 'HR_FOOT')
HRfootTask.setKp(kp_foot * matlib.ones(6).T)
HRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
#HRfootTask.setMask(mask)
HRfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HRfootTask, w_foot, 1, 0.0)

# FL foot
FLfootTask = tsid.TaskSE3Equality("FL-foot-grounded", robot, 'FL_FOOT')
FLfootTask.setKp(kp_foot * matlib.ones(6).T)
FLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
#FLfootTask.setMask(mask)
FLfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FLfootTask, w_foot, 1, 0.0)

# FR foot
FRfootTask = tsid.TaskSE3Equality("FR-foot-grounded", robot, 'FR_FOOT')
FRfootTask.setKp(kp_foot * matlib.ones(6).T)
FRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
#FRfootTask.setMask(mask)
FRfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 0 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FRfootTask, w_foot, 1, 0.0)


## TSID Trajectory

# Set the reference trajectory of the tasks

q_ref = qdes[7:] # Initial value of the joints of the robot (in half_sitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

## Initialisation of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


########################################################################
# Initialization of PyBullet variables 

v_prev = np.matrix(np.zeros(robot.nv)).T  # velocity during the previous time step, of size (robot.nv,1)

# Start the client for PyBullet
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# Set gravity (disabled by default)
#p.setGravity(0,0,-9.81)

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load Quadruped robot
robotStartPos = [0,0,1.0] #0.35
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/solo_description/robots")
robotId = p.loadURDF("solo12.urdf",robotStartPos, robotStartOrientation)

# Disable default motor control for revolute joints
revoluteJointIndices = [0,1,2, 4,5,6, 8,9,10, 12,13,14]
p.setJointMotorControlArray(robotId, 
                            jointIndices= revoluteJointIndices, 
                             controlMode= p.VELOCITY_CONTROL,
                       targetVelocities = [0.0 for m in revoluteJointIndices], 
                                 forces = [0.0 for m in revoluteJointIndices])

# Initialize the joint configuration
initial_joint_positions = [0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]
for i in range (len(initial_joint_positions)):
    p.resetJointState(robotId, revoluteJointIndices[i], initial_joint_positions[i])
                                
# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]
#jointTorques = qdes[7:]
p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

realTimeSimulation = True

# Launch the simulation
for i in range (N_SIMULATION):
	
	# Time at the start of the loop
    if realTimeSimulation:
        time_start = time.time()
        
    # Callback Pinocchio to get joint torques
    jointTorques = callback_torques()
    
    if(sol.status != 0):
	    print ("QP problem could not be solved ! Error code:", sol.status)
	    break
		
	# Set control torque for all joints
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

	# Compute one step of simulation
    p.stepSimulation()

	# Sleep to get a real time simulation
    if realTimeSimulation:
        time_spent = time.time() - time_start
        if (time_spent < dt) : 
            time.sleep(dt - time_spent)

	
embed()

# Shut down the client
p.disconnect()

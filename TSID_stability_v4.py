# coding: utf8

# Python libraries
import numpy as np
import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm
import time

# TSID library for the Whole-Body Controller
import tsid
# Pinocchio library for mathematical methods
import pinocchio as pin

#from pinocchio.utils import * #useless ?
from IPython import embed



## Definition of the tasks gains and weights

mu = 0.3  		# friction coefficient
fMin = 1.0		# minimum normal force
fMax = 100.0  	# maximum normal force

foot_frames = ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT']  # tab with all the foot frames names

w_posture = 0.1  	# weight of the posture task
w_foot = 10.0		# weight of the feet tasks
w_forceRef = 1e-3	# weight of force regularization task

kp_posture = 10.0  	# proportionnal gain of the posture task
kp_foot = 1000.0		# proportionnal gain of the feet tasks

N_SIMULATION = 10000	# number of time steps simulated
dt = 0.001			# controller time step



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
q = model.referenceConfigurations['straight_standing']
q[2] = 0.30	# move the robot from the ground
q[3] = -0.1
q[4] = 0.05 # bow the robot
# Set the current velocity to zero
v = np.matrix(np.zeros(robot.nv)).T


## Display the robot in Gepetto Viewer

robot_display.displayCollisions(False)
robot_display.displayVisuals(True)
robot_display.display(q)

## Check that the frames of the feet exist

assert [robot.model().existFrame(name) for name in foot_frames]


t = 0.0  # time


## Disable the gravity

robot.set_gravity_to_zero()

## Creation of the Invverse Dynamics HQP problem using the robot
## accelerations (base + joints) and the contact forces

invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
# Compute th eproblem data with a solver based on EiQuadProg
invdyn.computeProblemData(t, q, v)
# Get the initial data
data = invdyn.data()

base = robot.framePosition(data, model.getFrameId('root_joint')).copy()


## Tasks definition

# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * matlib.ones(robot.nv-6).T) # Proportional gain 
postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(robot.nv-6).T) # Derivative gain
# Add the task to the HQP with weight = 1e-3, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

# FOOT Task

# HL foot
HLfootTask = tsid.TaskSE3Equality("HL-foot-grounded", robot, 'HL_FOOT')
HLfootTask.setKp(kp_foot * matlib.ones(6).T)
HLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
HLfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HLfootTask, w_foot, 1, 0.0)

# HR foot
HRfootTask = tsid.TaskSE3Equality("HR-foot-grounded", robot, 'HR_FOOT')
HRfootTask.setKp(kp_foot * matlib.ones(6).T)
HRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
HRfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HRfootTask, w_foot, 1, 0.0)

# FL foot
FLfootTask = tsid.TaskSE3Equality("FL-foot-grounded", robot, 'FL_FOOT')
FLfootTask.setKp(kp_foot * matlib.ones(6).T)
FLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
FLfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FLfootTask, w_foot, 1, 0.0)

# FR foot
FRfootTask = tsid.TaskSE3Equality("FR-foot-grounded", robot, 'FR_FOOT')
FRfootTask.setKp(kp_foot * matlib.ones(6).T)
FRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
FRfootTask.useLocalFrame(False)
# Add the task to the HQP with weight = 1.0, priority level = 0 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FRfootTask, w_foot, 1, 0.0)



## TSID Trajectory

# Set the reference trajectory of the tasks

q_ref = q[7:] # Initial value of the joints of the robot (in half_sitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)


## Initialization of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


## Display the position of the foot in time and space
 
xFRpos = []
yFRpos = []
zFRpos = []
zFLpos = []
zHRpos = []
zHLpos = []

## Simulation loop

# At each time step compute the next desired trajectory of the tasks
# Set them as new references for each tasks 
# Compute the new problem data (HQP problem update)
# Solve the problem with the solver

# Integrate the control (which is in acceleration and is given to the robot in position):
# One simple euler integration from acceleration to velocity
# One integration (velocity to position) with pinocchio to have the freeFlyer updated
# Display the result on the gepetto viewer

for i in range(N_SIMULATION):
	
	time_start = time.time()
	
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
	zGoal = tbase[2,0] - 0.235	# match with the ref config of the feet
		
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
	
	
	HQPData = invdyn.computeProblemData(t, q, v)
    
    
	sol = solver.solve(HQPData)
	if (sol.status != 0):
		print ("QP problem could not be solved ! Error code:", sol.status)
		break
	
	
	tau = invdyn.getActuatorForces(sol)
	dv = invdyn.getAccelerations(sol)
	
	v_mean = v + 0.5*dt*dv
	v += dt*dv
	q = pin.integrate(model, q, dt*v_mean)
	t += dt	
	
	robot_display.display(q)
	
	
	xFRpos.append(robot.framePosition(data, model.getFrameId('FR_FOOT')).translation[0,0])
	yFRpos.append(robot.framePosition(data, model.getFrameId('FR_FOOT')).translation[1,0])
	zFRpos.append(robot.framePosition(data, model.getFrameId('FR_FOOT')).translation[2,0])
	zFLpos.append(robot.framePosition(data, model.getFrameId('FL_FOOT')).translation[2,0])
	zHRpos.append(robot.framePosition(data, model.getFrameId('HR_FOOT')).translation[2,0])
	zHLpos.append(robot.framePosition(data, model.getFrameId('HL_FOOT')).translation[2,0])
	
			
	time_spent = time.time() - time_start
	if (time_spent < dt) : time.sleep(dt - time_spent)

	
## Ploting the different trajectories of the foot during time and in the space

import matplotlib.pylab as plt

ts = np.linspace(0, N_SIMULATION*dt, N_SIMULATION)

err_zFRpos = abs(FR_foot_goal.translation[2,0] - zFRpos)
err_zFLpos = abs(FL_foot_goal.translation[2,0] - zFLpos)
err_zHRpos = abs(HR_foot_goal.translation[2,0] - zHRpos)
err_zHLpos = abs(HL_foot_goal.translation[2,0] - zHLpos)

plt.figure(1)
plt.plot(ts, zFRpos, label='FR foot')
plt.plot(ts, zFLpos, label='FL foot')
plt.plot(ts, zHRpos, label='HR foot')
plt.plot(ts, zHLpos, label='HL foot')
plt.grid()
plt.title('Position of each foot along z-axis function of time')
plt.legend()

plt.figure(2)
plt.plot(ts, err_zFRpos, label='FR foot')
plt.plot(ts, err_zFLpos, label='FL foot')
plt.plot(ts, err_zHRpos, label='HR foot')
plt.plot(ts, err_zHLpos, label='HL foot')
plt.grid()
plt.title('Position errors of each foot along z_axis function of time')
plt.legend()

plt.figure(3)
plt.plot(xFRpos, zFRpos)
plt.xlabel('x')
plt.ylabel('z' )
plt.grid()
plt.title('Trajectory of the FR foot in the space')

plt.show()

embed()

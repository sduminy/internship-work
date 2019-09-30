from pinocchio.utils import *
import robots_loader
import numpy as np
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_slsqp
from time import sleep
from IPython import embed
from numpy.linalg import pinv

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

sq = solo.q0

solo.display(sq)

def place(name, M):
	solo.viewer.gui.applyConfiguration(name, se3ToXYZQUAT(M))
	solo.viewer.gui.refresh()

def Rquat(x, y, z, w):
	q = pin.Quaternion(x, y, z, w)
	q.normalize()
	return q.matrix()

### Moving the basis along x_axis
embed()
ID_BASIS = solo.model.getFrameId('base_link')
q_basis = sq
dt = 1e-2

for j in range(200):
	pin.forwardKinematics(solo.model, solo.data, q_basis)
	pin.updateFramePlacements(solo.model, solo.data)
	Mbasis = solo.data.oMf[ID_BASIS]
	error = Mbasis.translation[0]-1
	J_basis = pin.frameJacobian(solo.model, solo.data, q_basis, ID_BASIS)[0,:]
	nu_basis = error
	vq_basis = pinv(J_basis)*nu_basis
	q_basis = pin.integrate(solo.model, q_basis, vq_basis * dt)
	solo.display(q_basis)
	sleep(dt)

embed()

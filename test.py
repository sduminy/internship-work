from pinocchio.utils import *
import robots_loader
import numpy as np
from IPython import embed 
import pinocchio as pin

import spacenav as sp
import atexit

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

q = solo.q0

solo.display(q)

vq = np.zeros([14,1])

dt = 1e-2

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

for i in range(200):
	event = sp.wait()
	vq[3] = event.rx
	vq[4] = event.ry
	vq[5] = event.rz
	q = pin.integrate(solo.model, q, vq*dt)
	solo.display(q)

embed()

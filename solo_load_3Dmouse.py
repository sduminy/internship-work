#import sys
#sys.path.append("~/Documents/pyspacenav")

from pinocchio.utils import *
import robots_loader
import numpy as np
from IPython import embed 
import pinocchio as pin

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

# load Solo robot
solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)

q = solo.q0

solo.display(q)

vq = np.zeros([14,1])

dt = 1e-2

# loop over space navigator events
while not stop:
	# wait for next event
	event = sp.wait()

	# if event signals the release of the first button
	if type(event) is sp.ButtonEvent \
		and event.button == 0 and event.pressed == 0:
		# set exit condition
		stop = True
	# matching the mouse signals with the position of Solo's basis
	if type(event) is sp.MotionEvent :
		vq[3] = -event.rx/100.0
		vq[5] = event.ry/100.0
		#vq[4] = event.rz/100.0
		q = pin.integrate(solo.model, q, vq*dt)
		solo.display(q)
	


embed()

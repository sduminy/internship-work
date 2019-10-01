from pinocchio.utils import *
import robots_loader
import numpy as np
from IPython import embed 

solo = robots_loader.loadSolo()
solo.initDisplay(loadModel=True)
solo.display(solo.q0)

embed()

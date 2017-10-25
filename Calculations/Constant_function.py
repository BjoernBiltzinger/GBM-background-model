#!/usr/bin python2.7

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from work_module import calculate
from work_module import detector
from work_module import readfile
calc = calculate()
det = detector()
rf = readfile()

#docstrings of the different self-made classes within the self-made module
#cdoc = calc.__doc__
#ddoc = det.__doc__
#rdoc = rf.__doc__

day = 150926

sat_data = rf.poshist(150926)
sat_time = sat_data[0]
sat_pos = sat_data[1]

#print np.where(np.fabs(sat_pos[0] - sat_pos[0,0]) < 3000)
#print np.where(np.fabs(sat_pos[1] - sat_pos[1,0]) < 2000)
#print np.where(np.fabs(sat_pos[2] - sat_pos[2,0]) < 2000)

#constant function corresponding to the diffuse y-ray background
ycon = np.ones(len(sat_time))

daytime = (sat_time - sat_time[0] + 5)/3600.

plt.plot(daytime, ycon, 'g-')

plt.show()

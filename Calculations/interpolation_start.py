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

import scipy.optimize as optimization
from scipy import interpolate

#docstrings of the different self-made classes within the self-made module
#cdoc = calc.__doc__
#ddoc = det.__doc__
#rdoc = rf.__doc__

day = 150926
detector = det.n0

ctime_data = rf.ctime(detector, day)
echan = ctime_data[0]
total_counts = ctime_data[1]
echan_counts = ctime_data[2]
total_rate = ctime_data[3]
echan_rate = ctime_data[4]
bin_time = ctime_data[5]
good_time = ctime_data[6]
exptime = ctime_data[7]

#bin_time = np.zeros((len(cstart),2), float) #combine the time limits of the counting intervals in one matrix
#bin_time[:,0] = cstart
#bin_time[:,1] = cstop
#bin_time_mid = np.array((bin_time[:,0]+bin_time[:,1])/2)


sat_data = rf.poshist(day)
sat_time = sat_data[0]
sat_pos = sat_data[1]
sat_lat = sat_data[2]
sat_lon = sat_data[3]
sat_q = sat_data[4]

#plot positions
#plt.plot(sat_time, sat_pos[0], 'b-', sat_time, sat_pos[1], 'r-', sat_time, sat_pos[2], 'y-')

#plot quaternions
#plt.plot(sat_time, sat_q[0], 'b-', sat_time, sat_q[1], 'r-', sat_time, sat_q[2], 'y-', sat_time, sat_q[3], 'g-')

#plot lat & lon
#plt.plot(sat_time, sat_lat, 'b-', sat_time, sat_lon, 'r-')

#plot spline to data bin_time
#bin_time = bin_time[bin_time < sat_time[len(sat_time) - 1]]
#tck = interpolate.splrep(sat_time, sat_q[1], s=0)
#sat_q2 = interpolate.splev(bin_time, tck, der=0)
#plt.plot(sat_time, sat_q[1], 'r-', bin_time, sat_q2, '.')
#print sat_q2, len(sat_q2), len(bin_time)


#plot spline to data sat_time
#tck = interpolate.splrep(sat_time, sat_q[1], s=0)
#sat_time2 = np.linspace(sat_time[0], sat_time[len(sat_time) - 1], 230000)
#sat_q2 = interpolate.splev(sat_time2, tck, der=0)
#plt.plot(sat_time, sat_q[1], 'r-', sat_time2, sat_q2, 'y--')
#print sat_q2, len(sat_q2)

plt.show()

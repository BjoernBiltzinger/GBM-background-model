#!/usr/bin python2.7

import subprocess
from subprocess import Popen, PIPE
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from scipy import interpolate
import scipy.optimize as optimization
from astropy.time import Time
from astropy.table import Table
import fileinput
from datetime import datetime
import matplotlib.path as mplPath
from work_module import calculate
from work_module import detector
from work_module import readfile
from work_module import writefile
calc = calculate()
det = detector()
rf = readfile()
wf = writefile()

#docstrings of the different self-made classes within the self-made module
#cdoc = calc.__doc__
#ddoc = det.__doc__
#rdoc = rf.__doc__



day = 150926
detector = det.n5

ctime = rf.ctime(detector, day)
total_counts = ctime[1]
bin_time = ctime[5]
bin_time_mid = np.array((bin_time[:,0]+bin_time[:,1])/2)

sat_data = rf.poshist(day)
sat_time = sat_data[0]
sat_pos = sat_data[1]
sat_lat = sat_data[2]
sat_lon = sat_data[3]
sat_q = sat_data[4]
sat_lat = calc.intpol(sat_lat, day, 0, sat_time, bin_time_mid, detector)[0]
sat_lon = calc.intpol(sat_lon, day, 0, sat_time, bin_time_mid, detector)[0]
sat_lon[np.where(sat_lon > 180.)] = sat_lon[np.where(sat_lon > 180.)] - 360.
sat_geo = np.column_stack((sat_lat, sat_lon))
sat_geo[len(sat_geo) - 1] = 2*sat_geo[len(sat_geo) - 2] - sat_geo[len(sat_geo) - 3]
sat_geo[np.where(total_counts == 0)] = 0

saa = rf.saa()

saa_path = mplPath.Path(np.array([saa[:,0], saa[:,1], saa[:,2], saa[:,3], saa[:,4], saa[:,5], saa[:,6], saa[:,7], saa[:,8], saa[:,9], saa[:,10], saa[:,11], saa[:,12]]))


#sat_geo[saa_path.contains_points(sat_geo)] = 0


#plt.plot(sat_time, sat_geo[:,0], 'b-', sat_time, sat_geo[:,1], 'r-')
plt.plot(sat_geo[:,1], sat_geo[:,0], 'b-', saa[1], saa[0], 'r-')

plt.xlabel('longitude')
plt.ylabel('latitude')

#plt.title('Sun-angle of the ' + detector.__name__ + '-detector on the 26th Sept 2015')

plt.show()

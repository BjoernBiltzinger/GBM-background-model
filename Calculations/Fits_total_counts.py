#!/usr/bin python2.7

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from scipy import interpolate
import scipy.optimize as optimization
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
detector = det.n5

ctime_data = rf.ctime(detector, day)
echan = ctime_data[0]
total_counts = ctime_data[1]
echan_counts = ctime_data[2]
total_rate = ctime_data[3]
echan_rate = ctime_data[4]
bin_time = ctime_data[5]
good_time = ctime_data[6]
exptime = ctime_data[7]
bin_time_mid = np.array((bin_time[:,0]+bin_time[:,1])/2)


sat_data = rf.poshist_bin(day, bin_time_mid)
sat_time_bin = sat_data[0]
sat_pos_bin = sat_data[1]
sat_lat_bin = sat_data[2]
sat_lon_bin = sat_data[3]
sat_q_bin = sat_data[4]

sun_data = calc.sun_ang_bin(detector, day, bin_time_mid)
sun_ang_bin = sun_data[0]

earth_data = calc.earth_ang_bin(detector, day, bin_time_mid)
earth_ang_bin = earth_data[0]


#periodical function corresponding to the orbital behaviour -> reference day is 150926, periodical shift per day is approximately 0.199*math.pi
sat_time = rf.poshist(day)[0]
#J2000-position oriented orbit
#ysin0 = np.sin((2*math.pi*np.arange(len(sat_time)))/5715 + (0.7 + (day - 150926)*0.199)*math.pi)
#ysin = calc.intpol(ysin0, day, 0, sat_time, bin_time_mid, detector)[0]
#LON-oriented orbit (earth rotation considered -> orbit within the magnetic field of the earth)
ysin0 = np.sin((2*math.pi*np.arange(len(sat_time)))/6120.85)
ysin = calc.intpol(ysin0, day, 0, sat_time, bin_time_mid, detector)[0]


#constant function corresponding to the diffuse y-ray background
ycon = np.ones(len(sat_time_bin))

bin_time_mid_norm = bin_time_mid - bin_time_mid[0]

ycon[np.where(total_counts == 0)] = 0
ysin[np.where(total_counts == 0)] = 0
earth_ang_bin[np.where(total_counts == 0)] = 0
sun_ang_bin[np.where(total_counts == 0)] = 0

'''
def fit_function(x, a, b, c, d, e):#, f):
    return a*ycon + b*(calc.intpol(np.sin((2*math.pi*np.arange(len(sat_time)))/6120.85 + c), day, 0, sat_time, bin_time_mid, detector)[0]) - d*earth_ang_bin - e*sun_ang_bin# + f*x

x0 = np.array([115., 1., 0.1, -1.8, 0.2])#, 0.0004])
sigma = np.array((total_counts + 10)**(0.5))

fit_results = optimization.curve_fit(fit_function, bin_time_mid_norm, total_counts, x0, sigma)

print fit_results[0]
#print np.where(total_counts == 0)
#print bin_time_mid[np.where(total_counts == 0)]
#print sat_time_bin[np.where(total_counts == 0)]
#print bin_time_mid[0]
#print sat_time_bin[0]

a = fit_results[0][0]
b = fit_results[0][1]
c = fit_results[0][2]
d = fit_results[0][3]
e = fit_results[0][4]
#f = fit_results[0][5]

fit_curve = a*ycon + b*(calc.intpol(np.sin((2*math.pi*np.arange(len(sat_time)))/6120.85 + c), day, 0, sat_time, bin_time_mid, detector)[0]) - d*earth_ang_bin - e*sun_ang_bin# + f*bin_time_mid_norm


#plot-algorhythm
fig, ax1 = plt.subplots()

#add two independent y-axes
ax2 = ax1.twinx()
ax3 = ax1.twinx()
axes = [ax1, ax2, ax3]

#Make some space on the right side for the extra y-axis
fig.subplots_adjust(right=0.75)

# Move the last y-axis spine over to the right by 20% of the width of the axes
axes[-1].spines['right'].set_position(('axes', 1.2))

# To make the border of the right-most axis visible, we need to turn the frame on. This hides the other plots, however, so we need to turn its fill off.
axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

plot1 = ax1.plot(bin_time_mid, total_counts, 'b-', label = 'Counts')
plot2 = ax1.plot(bin_time_mid, fit_curve, 'r-', label = 'Fit')
plot3 = ax2.plot(sat_time_bin, sun_ang_bin, 'y-', label = 'Sun angle')
plot4 = ax2.plot(sat_time_bin, earth_ang_bin, 'c-', label = 'Earth angle')
plot5 = ax3.plot(sat_time_bin, ysin, 'g--', label = 'Orbital period')
plot6 = ax3.plot(sat_time_bin, ycon, 'b--', label = 'Constant background')

plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6
labels = [l.get_label() for l in plots]
ax3.legend(plots, labels, loc=1)

ax1.grid()

ax1.set_xlabel('Time of day in 24h')
ax1.set_ylabel('Number of counts')
ax2.set_ylabel('Angle in degrees')
ax3.set_ylabel('Number')

#ax1.set_xlim([0, 24.1])
ax1.set_ylim([0, 550])
#ax2.set_xlim([0, 24.1])
ax2.set_ylim([0, 360])
#ax3.set_xlim([0, 24.1])
ax3.set_ylim([-1.5, 1.5])

plt.title('Counts and angles of the ' + detector.__name__ + '-detector on the 26th Sept 2015')

#plt.axis([0, 24.1, 0, 500])

#plt.legend()

plt.show()'''

'''
plt.plot(bin_time_mid, total_counts - fit_curve, 'b-')

plt.xlabel('time')
plt.ylabel('background-subtracted noise')

plt.title('Count-diagramme after background subtraction')

plt.ylim([-200, 200])

plt.show()
'''

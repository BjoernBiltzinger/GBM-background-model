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
detector = det.n5

#angle between the sun and the detector
calc_sun_ang = calc.sun_ang(detector, day)
sun_ang = calc_sun_ang[0]
sat_time = calc_sun_ang[1]

#angle between the earth and the detector
earth_ang = calc.earth_ang(detector, day)[0]

#counts in the detector during that day
ctime = rf.ctime(detector, day)
counts = ctime[1]
rate = ctime[3]
bin_time = ctime[5]
count_time = np.array((bin_time[:,0] + bin_time[:,1])/2)

#periodical function corresponding to the orbital behaviour -> reference day is 150926, periodical shift per day is approximately 0.199*math.pi
ysin = np.sin((2*math.pi*np.arange(len(sat_time)))/5715 + (0.7 + (day - 150926)*0.199)*math.pi)*20 + 300

#convertion to daytime in 24h
daytime_sun = (sat_time - sat_time[0] + 5)/3600.
daytime_counts = (count_time - sat_time[0] + 5)/3600.


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
plot1 = ax1.plot(daytime_counts, counts, 'b-', label = 'Counts')#daytime_sun, earth_ang, 'r-')
plot2 = ax2.plot(daytime_sun, sun_ang, 'y-', label = 'Sun angle')
plot3 = ax2.plot(daytime_sun, earth_ang, 'r-', label = 'Earth angle')
plot4 = ax2.plot(daytime_sun, ysin, 'g-', label = 'Orbital period')

plots = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in plots]
ax2.legend(plots, labels, loc=1)

ax1.grid()

ax1.set_xlabel('Time of day in 24h')
ax1.set_ylabel('Number of counts')
ax2.set_ylabel('Angle in degrees')

ax1.set_xlim([0, 24.1])
ax1.set_ylim([0, 550])
ax2.set_xlim([0, 24.1])
ax2.set_ylim([0, 360])

plt.title('Counts and angles of the ' + detector.__name__ + '-detector on the 26th Sept 2015')

#plt.axis([0, 24.1, 0, 500])

#plt.legend()

plt.show() 

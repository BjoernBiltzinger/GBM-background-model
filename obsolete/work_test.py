#!/usr/bin python2.7

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from gbmbkgpy.utils.external_prop import ExternalProps, writefile
from obsolete.work_module_refactor import calculate
from obsolete.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()

#docstrings of the different self-made classes within the self-made module
#cdoc = calc.__doc__
#ddoc = det.__doc__
#rdoc = rf.__doc__



day = 150926
detector = det.n1


#peak effective area interpolation
fitsname = 'peak_eff_area_angle_calib_GBM_all.fits'
__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'calibration'
path = os.path.join(os.path.dirname(__dir__), directory)
fitsfilepath = os.path.join(path, fitsname)
fits = fits.open(fitsfilepath, mode='update')
data = fits[1].data
fits.close()
x = data.field(0)
y1 = data.field(1)#for NaI echan[0:2]
y2 = data.field(2)#for NaI echan[4]
y3 = data.field(3)#for NaI echan[6]
y4 = data.field(4)#for BGO (898 keV)
y5 = data.field(5)#for BGO (1836 keV)
#normalize the factor to 0 degrees
'''y1 = y1/y1[90]
y2 = y2/y2[90]
y3 = y3/y3[90]
y4 = y4/y4[90]
y5 = y5/y5[90]'''

y = np.array([y1[90], y2[90], y3[90]])
x = np.array([33, 279, 662])

'''def fit_function(x, a, b, c, d, e, f, g):
    return a*cgb + geo_orb(b, c, counts) + d*earth_ang_bin + e*sun_ang_bin + j2000_orb(f, g, counts)

x0 = np.array([26., 0.2, -1.3, -0.2, -0.004, -1., 1.5])

fit_results = optimization.curve_fit(fit_function, bin_time_mid, counts, x0, sigma)
coeff = fit_results[0]'''

#y = np.log(y)

plt.plot(x, y, 'bo')
plt.xlim([0, 700])

plt.show()



#earth orbit calculation depending on longitude
'''ctime_data = rf.ctime(detector, day)
echan = ctime_data[0]
total_counts = ctime_data[1]
echan_counts = ctime_data[2]
total_rate = ctime_data[3]
echan_rate = ctime_data[4]
bin_time = ctime_data[5]
good_time = ctime_data[6]
exptime = ctime_data[7]
bin_time_mid = np.array((bin_time[:,0]+bin_time[:,1])/2)


sat_data = rf.poshist(day)
sat_time = sat_data[0]
sat_pos = sat_data[1]
sat_lat = sat_data[2]
sat_lon = sat_data[3]
sat_q = sat_data[4]

print np.where(np.fabs(sat_lon - 360.) < 0.1)'''


#distance satellite-earth plot
#day = 150926
'''sat = rf.poshist(day)
sat_time = sat[0]
sat_pos = sat[1]
sat_lon = sat[3]

distance = (LA.norm(sat_pos, axis=0) - 6371000.785)/1000

plt.plot(sat_time, sat_pos[0], 'b-')#, sat_time, sat_pos[0], 'r-', sat_time, distance, 'b-', sat_time, sat_pos[1], 'y-', sat_time, sat_pos[2], 'g-')

#labeling the axes
plt.ylabel('distance')
plt.xlabel('time')

#the range of the axes: [xmin, xmax, ymin, ymax]
#plt.ylim([520, 565])

#anzeigen lassen
plt.show()

print np.where(np.fabs(sat_pos[0]) < 5000)
print sat_time[np.where(np.fabs(sat_pos[0]) < 5000)]
print (86329-598)*2/31
#difference = sat_time[81910] - sat_time[2339]
#orbit_time_sec = difference/14
#orbit_time = orbit_time_sec/60
#orbits = orbit_time*15/60
#print orbit_time, orbits
#print np.where(np.fabs(distance - 541) < 0.005)'''


#effective angle calculation test with module
'''sun_or = calc.sun_ang(detector, day)
sun_ang = sun_or[0]
sat_time = sun_or[1]
sun_ang_eff = calc.ang_eff(sun_ang, detector)

daytime = (sat_time - sat_time[0] + 5)/3600.

plt.plot(daytime, sun_ang_eff, 'b-')#, daytime, sun_ang, 'r-')

plt.xlabel('time of day')
plt.ylabel('occultation angle')

plt.title('Sun-angle of the ' + detector.__name__ + '-detector on the 26th Sept 2015')

plt.show()'''

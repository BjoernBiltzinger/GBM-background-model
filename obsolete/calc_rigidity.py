#!/usr/bin python2.7

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

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

def calc_rigidity(day, bin_time_mid = 0):
    """This function interpolates the rigidity of the lookup table from Humble et al., 1979, evaluates the values for the satellite position on a given day and stores the data in an array of the form: rigidity\n
    Input:\n
    calc_rigidity ( day = JJMMDD )\n
    Output:\n
    0 = rigidity\n
    1 = sat_lon_bin\n
    2 = sat_lat_bin\n
    3 = sat_time_bin"""
    
    #data from the paper mentioned in the docstring. Rigidity at 400 km altitude
    lon = np.arange(0., 360.001, 30.)
    lat = np.arange(40., -40.001, -10.)
    rigidity_matrix = [[6.1, 6.94, 7.66, 8.47, 9.08, 9.17, 7.44, 5., 3.26, 2.08, 2.44, 4.43, 6.1], [10.54, 11.07, 11.87, 12.73, 12.73, 11.99, 10.75, 9.08, 6.28, 4.39, 4.80, 8.81, 10.54], [12.48, 13.25, 14.2, 14.93, 14.49, 13.65, 12.48, 11.52, 10.13, 7.01, 8.9, 11.3, 12.48], [12.99, 13.92, 14.93, 15.53, 15.23, 14.34, 13.51, 12.86, 11.87, 10.75, 10.96, 12.23, 12.99], [12.36, 13.25, 14.2, 14.93, 14.78, 14.2, 13.92, 13.38, 12.73, 11.87, 11.64, 12.11, 12.36], [10.96, 11.41, 12.36, 12.99, 13.12, 13.12, 13.12, 13.12, 12.73, 11.99, 11.52, 11.07, 10.96], [8.72, 8.81, 9.35, 9.63, 9.83, 10.33, 11.3, 11.99, 11.99, 11.52, 10.86, 9.93, 8.72], [6.34, 5.8, 5.2, 4.75, 4.85, 5.41, 7.01, 9.35, 10.64, 10.54, 9.83, 8.3, 6.34], [3.97, 3.42, 2.72, 2.06, 2.02, 2.42, 3.71, 5.2, 7.97, 9.08, 8.55, 6.1, 3.97]]
    
    #interpolation
    tck = interpolate.interp2d(lon, lat, rigidity_matrix, kind='cubic')
    
    #get the satellite data
    sat_data = rf.poshist_bin(day, bin_time_mid)
    sat_lon = sat_data[3]
    sat_lat = sat_data[2]
    sat_time = sat_data[0]
    
    #evaluate the rigidity -> for-loop is slow, but otherwise the tck-function doesn't seem to accept the sat_lon and sat_lat arrays
    rigidity = []
    for i in range(0, len(sat_lon)):
        rig = tck(sat_lon[i], sat_lat[i])
        rigidity.append(rig)
    rigidity = np.array(rigidity)
    
    return rigidity, sat_lon, sat_lat, sat_time

day = 150926

data = calc_rigidity(day)
rigidity = data[0]
sat_lon =data[1]
sat_lat = data[2]
sat_time = data[3]

lat_data = rf.mcilwain(day)
mc_b = lat_data[1]
mc_l = lat_data[2]

mc_b = calc.intpol(mc_b, day, 0, sat_time)[0]
mc_l = calc.intpol(mc_l, day, 0, sat_time)[0]

#plt.plot(sat_time, rigidity, 'b-')
plt.plot(sat_time, 8/rigidity, 'b--')
#plt.plot(sat_time, mc_b, 'r-')
plt.plot(sat_time, mc_l-0.5, 'g-')
#plt.plot(sat_time, 14.823/(mc_l**2.0311), 'g--')

plt.show()


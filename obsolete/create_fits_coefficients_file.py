#!/usr/bin python2.7

import math
import os
from datetime import datetime

import numpy as np
import scipy.optimize as optimization

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
detector = det.n0
data_type = 'ctime'
year = int('20' + str(day)[0:2])

#get the iso-date-format from the day
date = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))

#get the ordinal indicator for the date
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

#read the measurement data
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

for i in range(0,8):
    echan = i
    counts = echan_counts[echan]#define which energy-channels one wants to look at
    
    #read the satellite data
    sat_data = rf.poshist_bin(day, bin_time_mid, detector, data_type)
    sat_time_bin = sat_data[0]
    sat_pos_bin = sat_data[1]
    sat_lat_bin = sat_data[2]
    sat_lon_bin = sat_data[3]
    sat_q_bin = sat_data[4]
    
    #calculate the sun data
    sun_data = calc.sun_ang_bin(detector, day, bin_time_mid, data_type)
    sun_ang_bin = sun_data[0]
    sun_ang_bin = calc.ang_eff(sun_ang_bin, echan)[0]
    
    #calculate the earth data
    earth_data = calc.earth_ang_bin(detector, day, bin_time_mid, data_type)
    earth_ang_bin = earth_data[0]
    earth_ang_bin = calc.ang_eff(earth_ang_bin, echan)[0]
    
    #read the SFL data
    flares = rf.flares(year)
    flares_day = flares[0]
    flares_time = flares[1]
    if np.any(flares_day == day) == True:
        flares_today = flares_time[:,np.where(flares_day == day)]
        flares_today = np.squeeze(flares_today, axis=(1,))/3600.
    else:
        flares_today = np.array(-5)
    
    #periodical function corresponding to the orbital behaviour -> reference day is 150926, periodical shift per day is approximately 0.199*math.pi
    sat_time = rf.poshist(day)[0]
    def j2000_orb(f, g, counts):#J2000-position oriented orbit
        j2000_orb = f*(calc.intpol(np.sin((2*math.pi*np.arange(len(sat_time)))/5531 + g), day, 0, sat_time, bin_time_mid, detector)[0])
        j2000_orb[np.where(counts == 0)] = 0
        return j2000_orb
    
    def geo_orb(b, c, counts):#LON-oriented orbit (earth rotation considered -> orbit within the magnetic field of the earth)
        geo_orb = b*(calc.intpol(np.sin((2*math.pi*np.arange(len(sat_time)))/6120.85 + c), day, 0, sat_time, bin_time_mid, detector)[0])
        geo_orb[np.where(counts == 0)] = 0
        return geo_orb
    
    
    #constant function corresponding to the diffuse y-ray background
    cgb = np.ones(len(counts))
    
    cgb[np.where(counts == 0)] = 0
    earth_ang_bin[np.where(counts == 0)] = 0
    sun_ang_bin[np.where(counts == 0)] = 0
    
    
    def fit_function(x, a, b, c, d, e, f, g):
        return a*cgb + geo_orb(b, c, counts) + d*earth_ang_bin + e*sun_ang_bin + j2000_orb(f, g, counts)
    
    x0 = np.array([26., 0.2, -1.3, -0.2, -0.004, -1., 1.5])
    sigma = np.array((counts + 1)**(0.5))
    
    fit_results = optimization.curve_fit(fit_function, bin_time_mid, counts, x0, sigma)
    coeff = fit_results[0]
    
    a = fit_results[0][0]
    b = fit_results[0][1]
    c = fit_results[0][2]
    d = fit_results[0][3]
    e = fit_results[0][4]
    f = fit_results[0][5]
    g = fit_results[0][6]
    
    geo_orb = geo_orb(b, c, counts)
    j2000_orb = j2000_orb(f, g, counts)
    
    fit_curve = a*cgb + geo_orb + d*earth_ang_bin + e*sun_ang_bin + j2000_orb
    
    #write results to a data-file
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'Fits/' + str(day)
    fits_path = os.path.join(os.path.dirname(__dir__), directory)
    filename = 'fit_coeff_' + str(day) + '_' + str(detector.__name__) + '.txt'
    filepath = os.path.join(fits_path, str(filename))
    if not os.path.exists(fits_path):
        try:
            os.makedirs(fits_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    coeff_file = open(filepath, 'a')
    coeff_file.write(str(echan) + ' ' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' ' + str(e) + ' ' + str(f) + ' ' + str(g) + '\n')
    coeff_file.close()


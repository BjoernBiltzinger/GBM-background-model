#!/usr/bin python2.7

import subprocess
from subprocess import Popen, PIPE
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
import pyfits
from numpy import linalg as LA
import ephem
from scipy import interpolate
import scipy.optimize as optimization
from astropy.time import Time
from astropy.table import Table
from astropy.io import fits
import fileinput
from datetime import datetime
from work_module import calculate
from work_module import detector
from work_module import readfile
from work_module import writefile
calc = calculate()
det = detector()
rf = readfile()
wf = writefile()

def read_fit_coeff(days, detector):
    """This function reads the fitting coefficient files of a certain detector for different days and stores the data in arrays of the form: data (day, echans, cgb, geo_orb, earth, sun, j2000_orb).\n
    Input:\n
    read_fit_coeff ( days ([YYMMDD, YYMMDD, ...]), detector (det.n5) )\n
    Output:\n
    0 = data (day, echans, cgb, geo_orb, earth, sun, j2000_orb)\n"""
    
    data = []
    i = 0
    #read the file
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    for day in days:
        directory = 'Fits/' + str(day)
        fits_path = os.path.join(os.path.dirname(__dir__), directory)
        filename = 'fit_coeff_' + str(day) + '_' + str(detector.__name__) + '.txt'
        filepath = os.path.join(fits_path, str(filename))
        fits = open(filepath)
        lines = fits.readlines()
        fits.close()
        
        echans = []
        cgb = []
        geo_orb = [[],[]]
        earth = []
        sun = []
        j2000_orb = [[],[]]
        for line in lines:
            p = line.split()
            echans.append(float(p[0]))
            cgb.append(float(p[1]))
            geo_orb[0].append(float(p[2]))
            geo_orb[1].append(float(p[3]))
            earth.append(float(p[4]))
            sun.append(float(p[5]))
            j2000_orb[0].append(float(p[6]))
            j2000_orb[1].append(float(p[7]))
        
        temp = [day, echans, cgb, geo_orb, earth, sun, j2000_orb]
        data.append(temp)
        i = i+1
    
    return data




days = np.array([150321, 150926, 150927, 151126])
detector = det.n5

data = read_fit_coeff(days, detector)

plot1 = plt.plot(data[0][1], data[0][4], 'b-', label = data[0][0])
plot2 = plt.plot(data[1][1], data[1][4], 'r-', label = data[1][0])
plot3 = plt.plot(data[2][1], data[2][4], 'c-', label = data[2][0])
plot4 = plt.plot(data[3][1], data[3][4], 'g-', label = data[3][0])

plots = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in plots]
plt.legend(plots, labels, loc=1)

plt.show()

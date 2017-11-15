#!/usr/bin python2.7

from astropy.io       import fits
from astropy.table    import Table
from astropy.time     import Time
from datetime         import date
from datetime         import datetime
import ephem
import fileinput
import getpass
import math
import matplotlib.pyplot                   as plt
import matplotlib                          as mpl
import numpy                               as np
from numpy            import linalg        as LA
import os
import pyfits
import scipy.optimize                      as optimization
from scipy            import integrate
from scipy            import interpolate
from scipy.optimize   import curve_fit
import subprocess
from subprocess       import Popen, PIPE
import urllib2
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector
from gbmbkgpy.external_prop import ExternalProps, writefile

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()

day = np.array([140102])
detector = np.array(['nb'])
#detector = np.array(['n5'])
#echan = np.array([2, 3, 4, 5, 6])
echan = np.array([5, 6, 7])

#if len(day) == 1:
#    for j in range(0, len(detector)):
#        for k in range(0,len(echan)):
#            calc.curve_fit_plots(day, detector[j], echan[k])
#else:
for i in range(0, len(day)):
    print day[i]
    for j in range(0, len(detector)):
        print detector[j]
        for k in range(0, len(echan)):
            print echan[k]
            calc.curve_fit_plots(day[i], detector[j], echan[k], plot = 'yes')

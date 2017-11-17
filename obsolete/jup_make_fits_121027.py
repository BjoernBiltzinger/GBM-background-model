#!/usr/bin python2.7

import numpy                               as np

from gbmbkgpy.utils.external_prop import ExternalProps, writefile
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()

day = np.array([121027])
detector = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb'])
#detector = np.array(['n5'])
#echan = np.array([2, 3, 4, 5, 6])
echan = np.array([0, 1, 2, 3, 4, 5, 6, 7])

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

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
from scipy import integrate
from astropy.time import Time
from astropy.table import Table
from astropy.io import fits
import fileinput
from datetime import datetime
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector
from gbmbkgpy.external_prop import ExternalProps, writefile

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()


def read_earth_occ():
    """This function reads the earth occultation fits file and stores the data in arrays of the form: earth_ang, angle_d, area_frac, free_area, occ_area.\n
    Input:\n
    read_earth_occ ( )\n
    Output:\n
    0 = angle between detector direction and the earth in 0.5 degree intervals\n
    1 = opening angles of the detector (matrix)\n
    2 = fraction of the occulted area to the FOV area of the detector (matrix)\n
    3 = FOV area of the detector (matrix)\n
    4 = occulted area (matrix)"""
    
    #read the file
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'earth_occultation'
    fits_path = os.path.join(os.path.dirname(__dir__), directory)
    fitsname = 'earth_occ_kt.fits'
    fitsfilepath = os.path.join(fits_path, fitsname)
    fits = pyfits.open(fitsfilepath)
    angle_d = []
    area_frac = []
    free_area = []
    occ_area = []
    for i in range(1, len(fits)):
        data = fits[i].data
        angle_d.append(data.angle_d)
        area_frac.append(data.area_frac)
        free_area.append(data.free_area)
        occ_area.append(data.occ_area)
    fits.close()
    
    angle_d = np.array(angle_d, dtype = 'f')
    area_frac = np.array(area_frac, dtype = 'f')
    free_area = np.array(free_area, dtype = 'f')
    occ_area = np.array(occ_area, dtype = 'f')
    
    earth_ang = np.arange(0, 180.5, .5)
    
    return earth_ang, angle_d, area_frac, free_area, occ_area


data = read_earth_occ()
earth_ang = data[0]
angle_d = data[1]
area_frac = data[2]
free_area = data[3]
occ_area = data[4]



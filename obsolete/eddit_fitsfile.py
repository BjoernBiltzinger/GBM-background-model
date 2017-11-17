#!/usr/bin python2.7

import os

import numpy as np
from astropy.io import fits

from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()

#read the file
fitsname = 'peak_eff_area_angle_calib_GBM_all.fits'
__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'calibration'
path = os.path.join(os.path.dirname(__dir__), directory)
fitsfilepath = os.path.join(path, fitsname)
fits = fits.open(fitsfilepath, mode='update')
data = fits[1].data
fits.close()

print np.where(np.array(data.field(0))==0)

'''header = fits[1].header
cols = data.columns

#print data.field(0)
header['TTYPE1'] = 'Angle'
header['TTYPE2'] = 'NaI_33_keV'
header['TTYPE3'] = 'NaI_279_keV'
header['TTYPE4'] = 'NaI_662_keV'
header['TTYPE5'] = 'BGO_898_keV'
header['TTYPE6'] = 'BGO_1836_keV'''

'''print data.columns[0]

cols.units[0] = 'degrees'
cols.units[1] = 'cm^2'
cols.units[2] = 'cm^2'
cols.units[3] = 'cm^2'
cols.units[4] = 'cm^2'
cols.units[5] = 'cm^2'

print cols.names

#for i in range(0, len(cols.names)):
#    fits[1].header[2*i+8] = cols.names[i]'''

#fits.flush()





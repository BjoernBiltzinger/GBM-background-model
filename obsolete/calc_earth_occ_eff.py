#!/usr/bin python2.7

import os

import matplotlib.pyplot as plt
import numpy as np
import pyfits
from astropy.io import fits
from scipy import interpolate

from gbmbkgpy.utils.external_prop import ExternalProps
from obsolete.work_module_refactor import calculate
from obsolete.work_module_refactor import detector
from obsolete.work_module_refactor import writefile

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
    fitsname = 'earth_occ_calc_kt.fits'
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


def calc_earth_occ_eff(earth_ang, echan, datatype = 'ctime', detectortype = 'NaI'):
    """This function converts the earth angle into an effective earth occultation considering the angular dependence of the effective area and stores the data in an array of the form: earth_occ_eff\n
    Input:\n
    calc_earth_occ_eff ( earth_ang (in degrees), echan (integer in the range of 0-7 or 0-127), datatype='ctime' (or 'cspec'), detectortype='NaI' (or 'BGO') )\n
    Output:\n
    0 = effective earth occultation"""
    
    fitsname = 'peak_eff_area_angle_calib_GBM_all.fits'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'calibration'
    path = os.path.join(os.path.dirname(__dir__), directory)
    fitsfilepath = os.path.join(path, fitsname)
    fitsfile = fits.open(fitsfilepath, mode='update')
    data = fitsfile[1].data
    fitsfile.close()
    x = data.field(0)
    y1 = data.field(1)#for NaI (33 keV)
    y2 = data.field(2)#for NaI (279 keV)
    y3 = data.field(3)#for NaI (662 keV)
    y4 = data.field(4)#for BGO (898 keV)
    y5 = data.field(5)#for BGO (1836 keV)
    
    data = read_earth_occ()
    earth_ang_0 = data[0]
    angle_d = data[1][0]
    area_frac = data[2]
    free_area = data[3][0]
    occ_area = data[4]
    
    if detectortype == 'NaI':
        if datatype == 'ctime':
            #ctime linear-interpolation factors
            y1_fac = np.array([1., 1., 1., 1., 1., 0., 0., 0.])
            y2_fac = np.array([0., 0., 5./279., 42./279., 165./279., 1., 0., 0.])
            y3_fac = np.array([0., 0., 0., 0., 0., 138./662., 1., 1.])
            
            #resulting effective area curve
            y = y1_fac[echan]*y1 + y2_fac[echan]*y2 + y3_fac[echan]*y3
            
            #normalize
            y = y/y[90]
            
            #calculate the angle factors
            tck = interpolate.splrep(x, y)
            ang_fac = interpolate.splev(angle_d, tck, der=0)
            
        else:
            print 'datatype cspec not yet implemented'
        
    else:
        print 'detectortype BGO not yet implemented'
    
    free_circ_eff = [free_area[0]*ang_fac[0]]
    
    for i in range(1, len(free_area)):
        circ_area = free_area[i] - free_area[i-1]
        circ_area_eff = circ_area*ang_fac[i]
        free_circ_eff.append(circ_area_eff)
    
    free_circ_eff = np.array(free_circ_eff)
    
    occ_circ_eff = []
    
    '''occ_circ_eff = [occ_area[200][0]*ang_fac[0]]
    for i in range(1, len(occ_area[200])):
            circ_area = occ_area[200][i] - occ_area[200][i-1]
            circ_area_eff = circ_area*ang_fac[i]
            occ_circ_eff.append(circ_area_eff)'''
    
    for j in range(0, len(earth_ang_0)):
        occ_circ_eff_0 = [occ_area[j][0]*ang_fac[0]]
        for i in range(1, len(occ_area[j])):
            circ_area = occ_area[j][i] - occ_area[j][i-1]
            circ_area_eff = circ_area*ang_fac[i]
            occ_circ_eff_0.append(circ_area_eff)
        
        occ_circ_eff.append(occ_circ_eff_0)
    
    occ_circ_eff = np.array(occ_circ_eff)
    #eff_area_frac = np.sum(occ_circ_eff)/np.sum(free_circ_eff)
    eff_area_frac_0 = np.sum(occ_circ_eff, axis = 1)/np.sum(free_circ_eff)
    
    tck = interpolate.splrep(earth_ang_0, eff_area_frac_0, s=0)
    eff_area_frac = interpolate.splev(earth_ang, tck, der=0)
    
    return free_circ_eff, angle_d, occ_circ_eff, earth_ang_0, eff_area_frac

calc = calc_earth_occ_eff(0, 0)
free_circ_eff = calc[0]
angle_d = calc[1]
occ_circ_eff = calc[2]
earth_ang = calc[3]
eff_area_frac = calc[4]

data = read_earth_occ()
earth_ang = data[0]
angle_d = data[1]
area_frac = data[2][:,-1]
free_area = data[3]
occ_area = data[4][:,-1]

fitsname = 'peak_eff_area_angle_calib_GBM_all.fits'
__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'calibration'
path = os.path.join(os.path.dirname(__dir__), directory)
fitsfilepath = os.path.join(path, fitsname)
fitsfile = fits.open(fitsfilepath, mode='update')
data = fitsfile[1].data
fitsfile.close()
x = data.field(0)
y1 = data.field(1)#for NaI (33 keV)
y2 = data.field(2)#for NaI (279 keV)
y3 = data.field(3)#for NaI (662 keV)
y4 = data.field(4)#for BGO (898 keV)
y5 = data.field(5)#for BGO (1836 keV)

y1_fac = np.array([1., 1., 1., 1., 1., 0., 0., 0.])
y2_fac = np.array([0., 0., 5./279., 42./279., 165./279., 1., 0., 0.])
y3_fac = np.array([0., 0., 0., 0., 0., 138./662., 1., 1.])

#resulting effective area curve
y = y1_fac[0]*y1 + y2_fac[0]*y2 + y3_fac[0]*y3

#normalize
y = y/y[90]

#calculate the angle factors
tck = interpolate.splrep(x, y)
ang_fac = interpolate.splev(earth_ang, tck, der=0)


plt.plot(earth_ang, eff_area_frac, 'b-')
plt.plot(earth_ang, ang_fac, 'r-')
plt.plot(earth_ang, area_frac, 'b--')
plt.plot(earth_ang, occ_area, 'y-')

plt.show()



#!/usr/bin python2.7

import math
import os

import numpy as np
import pyfits
from astropy.table import Table

from gbmbkgpy.utils.external_prop import ExternalProps, writefile
from obsolete.work_module_refactor import calculate
from obsolete.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()

class n0:
    azimuth = 45.8899994*2*math.pi/360.
    zenith = 20.5799999*2*math.pi/360.
    azimuthg = 45.8899994
    zenithg = 20.5799999

class n1:
    azimuth = 45.1100006*2*math.pi/360.
    zenith = 45.3100014*2*math.pi/360.
    azimuthg = 45.1100006
    zenithg = 45.3100014

class n2:
    azimuth = 58.4399986*2*math.pi/360.
    zenith = 90.2099991*2*math.pi/360.
    azimuthg = 58.4399986
    zenithg = 90.2099991

class n3:
    azimuth = 314.869995*2*math.pi/360.
    zenith = 45.2400017*2*math.pi/360.
    azimuthg = 314.869995
    zenithg = 45.2400017

class n4:
    azimuth = 303.149994*2*math.pi/360.
    zenith = 90.2699966*2*math.pi/360.
    azimuthg = 303.149994
    zenithg = 90.2699966

class n5:
    azimuth = 3.34999990*2*math.pi/360.
    zenith = 89.7900009*2*math.pi/360.
    azimuthg = 3.34999990
    zenithg = 89.7900009

class n6:
    azimuth = 224.929993*2*math.pi/360.
    zenith = 20.4300003*2*math.pi/360.
    azimuthg = 224.929993
    zenithg = 20.4300003

class n7:
    azimuth = 224.619995*2*math.pi/360.
    zenith = 46.1800003*2*math.pi/360.
    azimuthg = 224.619995
    zenithg = 46.1800003

class n8:
    azimuth = 236.610001*2*math.pi/360.
    zenith = 89.9700012*2*math.pi/360.
    azimuthg = 236.610001
    zenithg = 89.9700012

class n9:
    azimuth = 135.190002*2*math.pi/360.
    zenith = 45.5499992*2*math.pi/360.
    azimuthg = 135.190002
    zenithg = 45.5499992

class na:
    azimuth = 123.730003*2*math.pi/360.
    zenith = 90.4199982*2*math.pi/360.
    azimuthg = 123.730003
    zenithg = 90.4199982

class nb:
    azimuth = 183.740005*2*math.pi/360.
    zenith = 90.3199997*2*math.pi/360.
    azimuthg = 183.740005
    zenithg = 90.3199997

class b0:
    azimuth = math.acos(1)
    zenith = math.asin(1)
    azimuthg = 0.0
    zenithg = 90.0

class b1:
    azimuth = math.pi
    zenith = math.asin(1)
    azimuthg = 180.0
    zenithg = 90.0

def read_lat_spacecraft(week):
    """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
    Input:\n
    read_lat_spacecraft ( week = WWW )\n
    Output:\n
    0 = time\n
    1 = mcilwain parameter B\n
    2 = mcilwain parameter L"""
    
    #read the file
    filename = 'lat_spacecraft_weekly_w' + str(week) + '_p202_v001.fits'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(os.path.dirname(__dir__), 'lat')
    filepath = os.path.join(fits_path, str(filename))
    fits = pyfits.open(filepath)
    data = fits[1].data
    fits.close()
    
    #extract the data
    lat_time = data.START #Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
    mc_b = data.B_MCILWAIN #Position in J2000 equatorial coordinates
    mc_l = data.L_MCILWAIN
    return lat_time, mc_b, mc_l

week = 380
day = 150914
datum = '20' + str(day)[:2] + '-' + str(day)[2:4] + '-' + str(day)[4:]
lat_data = read_lat_spacecraft(week)#pay attention to the first and last day of the weekly file, as they are split in two!
lat_time = lat_data[0]
mc_b = lat_data[1]
mc_l = lat_data[2]

date = calc.met_to_date(lat_time)[3]
date = np.array(date)

x = []
for i in range(0, len(date)):
    date[i] = str(date[i])
    if date[i][:10] == datum:
        x.append(i)

x = np.array(x)

x1 = x[0]-1
x2 = x[-1]+2

lat_time = lat_time[x1:x2]
mc_b = mc_b[x1:x2]
mc_l = mc_l[x1:x2]


interpol1 = calc.intpol(mc_b, day, 1, 0, lat_time)
mc_b = interpol1[0]
sat_time = interpol1[1]

interpol2 = calc.intpol(mc_l, day, 1, 0, lat_time)
mc_l = interpol2[0]


__dir__ = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(os.path.dirname(__dir__), 'mcilwain')

filename = 'glg_mcilwain_all_' + str(day) + '_kt.txt'

filepath = os.path.join(path, str(filename))

mc_file = open(filepath, 'w')
for i in range(0, len(sat_time)):
    mc_file.write(str(sat_time[i]) + ' ' + str(mc_b[i]) + ' ' + str(mc_l[i]) + '\n')
mc_file.close()


content = Table.read(filepath, format='ascii')
fitsname = 'glg_mcilwain_all_' + str(day) + '_kt.fits'
fitsfilepath = os.path.join(path, fitsname)
content.write(fitsfilepath, overwrite=True)

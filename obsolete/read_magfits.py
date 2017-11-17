#!/usr/bin python2.7

import math
import os

import matplotlib.pyplot as plt
import pyfits

from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()

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

def read_magfits(day):
    """This function reads a magnetic field fits file and stores the data in arrays of the form: t_magn, h_magn, x_magn, y_magn, z_magn.\n
    Input:\n
    read_magfits ( day = YYMMDD )\n
    Output:\n
    0 = t_magn\n
    1 = h_magn\n
    2 = x_magn\n
    3 = y_magn\n
    4 = z_magn"""
    
    #read the file
    fitsname = 'magn_' + str(day) + '_kt.fits'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'magnetic_field/' + str(day)
    path = os.path.join(os.path.dirname(__dir__), directory)
    filepath = os.path.join(path, str(fitsname))
    fits = pyfits.open(filepath)
    data = fits[1].data
    fits.close()
    
    #extract the data
    altitude = data.Altitude #altitude of the satellite above the WGS 84 ellipsoid
    t_magn = data.F_nT #total intensity of the geomagnetic field
    h_magn = data.H_nT # horizontal intensity of the geomagnetic field
    x_magn = data.X_nT #north component of the geomagnetic field
    y_magn = data.Y_nT #east component of the geomagnetic field
    z_magn = data.Z_nT #vertical component of the geomagnetic field
    return t_magn, h_magn, x_magn, y_magn, z_magn

day = 150926

magn_data = read_magfits(day)
t_magn = magn_data[0]
h_magn = magn_data[1]
x_magn = magn_data[2]
y_magn = magn_data[3]
z_magn = magn_data[4]

sat_data = rf.poshist(day)
sat_time = sat_data[0]
sat_pos = sat_data[1]
sat_lat = sat_data[2]
sat_lon = sat_data[3]
sat_q = sat_data[4]


plt.plot(sat_time, t_magn, 'b-', sat_time, h_magn, 'r--', sat_time, z_magn, 'g--')

plt.show()

#!/usr/bin python2.7

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem

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

def read_flares(year):
    """This function reads the YYYY.txt file containing the GOES solar flares of the corresponding year and returns the data in arrays of the form: day, time\n
    Input:\n
    year = YYYY\n
    Output\n
    0 = day ('YYMMDD')
    1 = time[start][stop] (in seconds on that day -> accuracy ~ 1 minute)\n"""
    filename = str(year) + '.dat'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    saa_path = os.path.join(os.path.dirname(__dir__), 'flares')
    filepath = os.path.join(saa_path, str(filename))
    flares = open(filepath)
    lines = flares.readlines()
    flares.close()
    day = []
    start = []
    stop = []#define day, start & stop arrays
    for line in lines:#write file data into the arrays
        p = line.split()
        day.append(int(p[0][5:11]))
        start.append(int(p[1][0:2])*3600. + int(p[1][2:4])*60.)
        stop.append(int(p[2][0:2])*3600. + int(p[2][2:4])*60.)
    
    #create numpy arrays
    day = np.array(day)
    start = np.array(start)
    stop = np.array(stop)
    time = np.array([start, stop])
    
    return day, time

year = 2015
flares = read_flares(2015)
day = flares[0]
time = flares[1]

print day, time

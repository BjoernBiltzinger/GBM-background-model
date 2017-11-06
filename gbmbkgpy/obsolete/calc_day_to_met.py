#!/usr/bin python2.7

from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from astropy.time import Time

def calc_day_to_met(day):
    """This function converts a day into the MET format and stores it in the form: met, mjdutc.\n
    Input:\n
    calc_day_to_met ( day = YYMMDD )\n
    Output:\n
    0 = met\n
    1 = mjdutc"""
    
    times = '20' + str(day)[0:2] + '-' + str(day)[2:4] + '-' + str(day)[4:6] + 'T00:00:00.0'
    t = Time(times, format='isot', scale='utc')
    mjdutc = t.mjd
    met = (mjdutc - 51910 - 0.0007428703703)*86400.0 + 68.184
    return met, mjdutc

time = calc_day_to_met(150926)
met = time[0]
mjdutc = time[1]

print 'met: ', met
print 'mjdutc: ', mjdutc


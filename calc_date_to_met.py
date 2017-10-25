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

def calc_date_to_met(year, month = 01, day = 01, hour = 00, minute = 00, seconds = 00.):
    """This function converts a date into the MET format and stores it in the form: met, mjdutc.\n
    Input:\n
    calc_date_to_met ( year, month = 01, day = 01, hour = 00, minute = 00, seconds = 00. )\n
    Output:\n
    0 = met\n
    1 = mjdutc"""
    
    times = str(year) + '-' + str(month) + '-' + str(day) + 'T' + str(hour) + ':' + str(minute) + ':' + str(seconds)
    t = Time(times, format='isot', scale='utc')
    mjdutc = t.mjd
    met = (mjdutc - 51910 - 0.0007428703703)*86400.0 + 68.184
    return met, mjdutc

time = calc_date_to_met(2015, 9, 26)
met = time[0]
mjdutc = time[1]

print 'met: ', met
print 'mjdutc: ', mjdutc


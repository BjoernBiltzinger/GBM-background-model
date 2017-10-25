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

def calc_met_to_date(met):
    """This function converts a MET to other times and stores it in the form: mjdutc, mjdtt, isot, date, decimal.\n
    Input:\n
    calculate.met_to_date ( met )\n
    Output:\n
    0 = mjdutc\n
    1 = mjdtt\n
    2 = isot\n
    3 = date\n
    4 = decimal"""
    
    mjdutc = ((met - 68.184) /86400.0)+51910+0.0007428703703 #-68.184 added to account for diff between TT and UTC and the 4 leapseconds since 2001
    mjdtt = ((met) /86400.0)+51910+0.00074287037037
    mjdtt = Time(mjdtt, scale='tt', format='mjd')
    isot = Time(mjdtt, scale='utc', format='isot')
    date = Time(mjdtt, scale='utc', format='iso')
    decimal = Time(mjdtt, scale='utc', format='decimalyear')
    return mjdutc, mjdtt, isot, date, decimal

time = calc_met_to_date(24*3600)
mjdutc = time[0]
mjdtt = time[1]
isot = time[2]
date = time[3]
decimal = time[4]

print 'mjdutc: ', mjdutc
print 'mjdtt: ', mjdtt
print 'isot: ', isot
print 'date: ', date
print 'decimal: ', decimal

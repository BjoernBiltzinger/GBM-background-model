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

#define a function
def read_poshist(day):
    """This function reads a posthist file and stores the data in arrays of the form: sat_time, sat_x, sat_y, sat_z, sat_lat, sat_lon, sat_q1, sat_q2, sat_q3, sat_q4.\n Input read_poshist ( day = YYMMDD )\n 0 = time\n 1 = x\n 2 = y\n 3 = z\n 4 = lat\n 5 = lon\n 6 = q1\n 7 = q2\n 8 = q3\n 9 = q4\n"""
    filename = 'glg_poshist_all_' + str(day) + '_v00.fit'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(os.path.dirname(__dir__), 'poshist')
    filepath = os.path.join(fits_path, str(filename))
    fits = pyfits.open(filepath)
    data = fits[1].data
    fits.close()
    sat_time = data.SCLK_UTC #Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
    sat_x = data.POS_X
    sat_y = data.POS_Y
    sat_z = data.POS_Z #Position in J2000 equatorial coordinates
    sat_lat = data.SC_LAT
    sat_lon = data.SC_LON #Earth-angles -> considers earth rotation (needed for SAA)
    sat_q1 = data.QSJ_1
    sat_q2 = data.QSJ_2
    sat_q3 = data.QSJ_3
    sat_q4 = data.QSJ_4 #Quaternionen -> 4D-Raum mit dem man Drehungen beschreiben kann (rocking motion)
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system
    nt=np.size(sat_time)
    scx=np.zeros((nt,3),float)
    scx[:,0]=(sat_q1**2 - sat_q2**2 - sat_q3**2 + sat_q4**2)
    scx[:,1]=2.*(sat_q1*sat_q2 + sat_q4*sat_q3)
    scx[:,2]=2.*(sat_q1*sat_q3 - sat_q4*sat_q2)
    scy=np.zeros((nt,3),float)
    scy[:,0]=2.*(sat_q1*sat_q2 - sat_q4*sat_q3)
    scy[:,1]=(-sat_q1**2 + sat_q2**2 - sat_q3**2 + sat_q4**2)
    scy[:,2]=2.*(sat_q2*sat_q3 + sat_q4*sat_q1)
    scz=np.zeros((nt,3),float)
    scz[:,0]=2.*(sat_q1*sat_q3 + sat_q4*sat_q2)
    scz[:,1]=2.*(sat_q2*sat_q3 - sat_q4*sat_q1)
    scz[:,2]=(-sat_q1**2 - sat_q2**2 + sat_q3**2 + sat_q4**2)
    x = np.zeros((3), float)
    x[0] = 1.
    y = np.zeros((3), float)
    y[1] = 1.
    z = np.zeros((3), float)
    z[2] = 1.
    #calculate the orientations of the satellite axes in geographical coordinates (J2000)
    sat_coorx=np.zeros((nt,3),float)
    sat_coorx[:,0]=scx[:,0]*x[0]+scx[:,1]*x[1]+scx[:,2]*x[2]
    sat_coorx[:,1]=scy[:,0]*x[0]+scy[:,1]*x[1]+scy[:,2]*x[2]
    sat_coorx[:,2]=scz[:,0]*x[0]+scz[:,1]*x[1]+scz[:,2]*x[2]
    sat_coory=np.zeros((nt,3),float)
    sat_coory[:,0]=scx[:,0]*y[0]+scx[:,1]*y[1]+scx[:,2]*y[2]
    sat_coory[:,1]=scy[:,0]*y[0]+scy[:,1]*y[1]+scy[:,2]*y[2]
    sat_coory[:,2]=scz[:,0]*y[0]+scz[:,1]*y[1]+scz[:,2]*y[2]
    sat_coorz=np.zeros((nt,3),float)
    sat_coorz[:,0]=scx[:,0]*z[0]+scx[:,1]*z[1]+scx[:,2]*z[2]
    sat_coorz[:,1]=scy[:,0]*z[0]+scy[:,1]*z[1]+scy[:,2]*z[2]
    sat_coorz[:,2]=scz[:,0]*z[0]+scz[:,1]*z[1]+scz[:,2]*z[2]
    #calculate the ra and dec for the satellite axes
    sat_rax = np.arctan2(sat_coorx[:,1], sat_coorx[:,0])*360./(2.*math.pi)
    sat_decx = np.arcsin(sat_coorx[:,2])*360./(2.*math.pi)
    sat_ray = np.arctan2(sat_coory[:,1], sat_coory[:,0])*360./(2.*math.pi)
    sat_decy = np.arcsin(sat_coory[:,2])*360./(2.*math.pi)
    sat_raz = np.arctan2(sat_coorz[:,1], sat_coorz[:,0])*360./(2.*math.pi)
    sat_decz = np.arcsin(sat_coorz[:,2])*360./(2.*math.pi)
    #put ra and dec together in one array as [:,0] and [:,1]
    sat_radx = np.zeros((nt,2), float)
    sat_radx[:,0] = sat_rax
    sat_radx[:,1] = sat_decx
    sat_rady = np.zeros((nt,2), float)
    sat_rady[:,0] = sat_ray
    sat_rady[:,1] = sat_decy
    sat_radz = np.zeros((nt,2), float)
    sat_radz[:,0] = sat_raz
    sat_radz[:,1] = sat_decz
    return sat_time, sat_x, sat_y, sat_z, sat_lat, sat_lon, sat_q1, sat_q2, sat_q3, sat_q4, sat_radx, sat_rady, sat_radz

def read_saa():
    """This function reads the saa.dat file and returns the polygon in the form: saa[lat][lon]\n 0 = saa\n"""
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    saa_path = os.path.join(os.path.dirname(__dir__), 'saa')
    filepath = os.path.join(saa_path, 'saa.dat')
    poly = open(filepath)
    lines = poly.readlines()
    poly.close()
    saa_lat = []
    saa_lon = []
    for line in lines:
        p = line.split()
        saa_lat.append(float(p[0]))
        saa_lon.append(float(p[1]))
    saa = np.array([saa_lat, saa_lon])
    return saa


day = 150926
sat_data = read_poshist(day)
sat_time = sat_data[0]
sat_x = sat_data[1]
sat_y = sat_data[2]
sat_z = sat_data[3]
sat_lat = sat_data[4]
sat_lon = sat_data[5]
sat_q1 = sat_data[6]
sat_q2 = sat_data[7]
sat_q3 = sat_data[8]
sat_q4 = sat_data[9]
sat_radx = sat_data[10]
sat_rady = sat_data[11]
sat_radz = sat_data[12]

saa = read_saa()

plt.plot(sat_lon - 180, sat_lat, 'b-', saa[1], saa[0], 'r--')
plt.axis([-185, 185, -40, 40])
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()

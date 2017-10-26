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


def read_poshist(day):
    """This function reads a posthist file and stores the data in arrays of the form: sat_time, sat_pos, sat_lat, sat_lon, sat_q.\n
    Input read_poshist( day = YYMMDD )\n 0 = time\n 1 = position (x, y, z)\n 2 = latitude\n 3 = longitude\n 4 = quaternion matrix (q1, q2, q3, q4)"""
    filename = 'glg_poshist_all_' + str(day) + '_v00.fit'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(os.path.dirname(__dir__), 'poshist')
    filepath = os.path.join(fits_path, str(filename))
    fits = pyfits.open(filepath)
    data = fits[1].data
    fits.close()
    sat_time = data.SCLK_UTC #Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
    sat_pos = np.array([data.POS_X, data.POS_Y, data.POS_Z]) #Position in J2000 equatorial coordinates
    sat_lat = data.SC_LAT
    sat_lon = data.SC_LON #Earth-angles -> considers earth rotation (needed for SAA)
    sat_q = np.array([data.QSJ_1, data.QSJ_2, data.QSJ_3, data.QSJ_4]) #Quaternionen -> 4D-Raum mit dem man Drehungen beschreiben kann (rocking motion)
    return sat_time, sat_pos, sat_lat, sat_lon, sat_q

def calc_sat_to_geo(sat_q, sat_coor):
    """This function converts the satellite coordinates into geographical coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: geo_coor, geo_rad\n
    Input calc_sat_to_geo( sat_q = quaternion-matrix, sat_coor = 3D-array(x, y, z) )\n
    0 = geo_coor\n
    1 = geo_rad"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    nt=np.size(sat_q[0])
    scx=np.zeros((nt,3),float)
    scx[:,0]=(sat_q[0]**2 - sat_q[1]**2 - sat_q[2]**2 + sat_q[3]**2)
    scx[:,1]=2.*(sat_q[0]*sat_q[1] + sat_q[3]*sat_q[2])
    scx[:,2]=2.*(sat_q[0]*sat_q[2] - sat_q[3]*sat_q[1])
    scy=np.zeros((nt,3),float)
    scy[:,0]=2.*(sat_q[0]*sat_q[1] - sat_q[3]*sat_q[2])
    scy[:,1]=(-sat_q[0]**2 + sat_q[1]**2 - sat_q[2]**2 + sat_q[3]**2)
    scy[:,2]=2.*(sat_q[1]*sat_q[2] + sat_q[3]*sat_q[0])
    scz=np.zeros((nt,3),float)
    scz[:,0]=2.*(sat_q[0]*sat_q[2] + sat_q[3]*sat_q[1])
    scz[:,1]=2.*(sat_q[1]*sat_q[2] - sat_q[3]*sat_q[0])
    scz[:,2]=(-sat_q[0]**2 - sat_q[1]**2 + sat_q[2]**2 + sat_q[3]**2)

    #create geo_to_sat rotation matrix
    sat_mat = np.array([scx, scy, scz])

    #create sat_to_geo rotation matrix
    geo_mat = np.transpose(sat_mat)

    #convert sat_coordinates to geo_coordinates
    sat_coor = np.array(sat_coor)
    geo_coor=np.zeros((3,nt),float)
    geo_coor[0]=geo_mat[0,:,0]*sat_coor[0]+geo_mat[0,:,1]*sat_coor[1]+geo_mat[0,:,2]*sat_coor[2]
    geo_coor[1]=geo_mat[1,:,0]*sat_coor[0]+geo_mat[1,:,1]*sat_coor[1]+geo_mat[1,:,2]*sat_coor[2]
    geo_coor[2]=geo_mat[2,:,0]*sat_coor[0]+geo_mat[2,:,1]*sat_coor[1]+geo_mat[2,:,2]*sat_coor[2]

    #calculate ra and dec
    geo_ra = np.arctan2(-geo_coor[1], -geo_coor[0])*360./(2.*math.pi)+180.
    geo_dec = np.arctan(geo_coor[2]/(geo_coor[0]**2 + geo_coor[1]**2)**0.5)*360./(2.*math.pi)

    #put ra and dec together in one array as [:,0] and [:,1]
    geo_rad = np.zeros((nt,2), float)
    geo_rad[:,0] = geo_ra
    geo_rad[:,1] = geo_dec
    return geo_coor, geo_rad

def calc_det_or( detector, day ):
    """This function reads a posthist file and the detector assembly table to calculate the detector orientation and stores it in arrays of the form: det_coor, det_rad, sat_pos, sat_time\n
    Input:\n
    calc_det_or( detector = n0/n1/b0.., day = YYMMDD )\n
    Output:\n
    0 = detector coordinates[x[], y[], z[]]\n
    1 = detector geocentric angles [(right ascension, declination)]\n
    2 = satellite position [x[], y[], z[]]\n
    3 = time (MET) in seconds"""

    #get satellite data for the convertion
    sat_data = read_poshist(day)
    sat_time = sat_data[0]
    sat_pos = sat_data[1]
    sat_q = sat_data[4]

    #get detector orientation data (in sat-coordinates) from the defined detector-class
    az = detector.azimuth
    zen = detector.zenith
    det_pos = np.array([math.cos(az)*math.sin(zen), math.sin(az)*math.sin(zen), math.cos(zen)]) #convert into unit-vector in the satellite coordinate system
    
    #convert the orientation in geo-coordinates
    det_geo = calc_sat_to_geo(sat_q, det_pos)
    det_coor = det_geo[0] #unit-vector
    det_rad = det_geo[1]
    
    return det_coor, det_rad, sat_pos, sat_time


day = 150926
n0_or = calc_det_or(n0, day)
n0_coor = n0_or[0]
n0_rad = n0_or[1]
sat_pos = n0_or[2]
sat_time = n0_or[3]

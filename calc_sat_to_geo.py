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

def calc_sat_to_geo(sat_q, sat_coor):
    """This function converts the satellite coordinates into geographical coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: geo_coor, geo_rad\n
    Input:\n
    calc_sat_to_geo ( sat_q = quaternion-matrix, sat_coor = 3D-array(x, y, z) )\n
    Output:\n
    0 = geocentric coordinates[x[], y[], z[]]\n
    1 = geocentric angles[(right ascension, declination)]"""
    
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

    #transpose into sat_to_geo rotation matrix
    geo_mat = np.transpose(sat_mat)

    #convert satellite coordinates into geocentric coordinates
    sat_coor = np.array(sat_coor)
    geo_coor=np.zeros((3,nt),float)
    geo_coor[0]=geo_mat[0,:,0]*sat_coor[0]+geo_mat[0,:,1]*sat_coor[1]+geo_mat[0,:,2]*sat_coor[2]
    geo_coor[1]=geo_mat[1,:,0]*sat_coor[0]+geo_mat[1,:,1]*sat_coor[1]+geo_mat[1,:,2]*sat_coor[2]
    geo_coor[2]=geo_mat[2,:,0]*sat_coor[0]+geo_mat[2,:,1]*sat_coor[1]+geo_mat[2,:,2]*sat_coor[2]

    #calculate the right ascension and declination
    geo_ra = np.arctan2(-geo_coor[1], -geo_coor[0])*360./(2.*math.pi)+180.
    geo_dec = np.arctan(geo_coor[2]/(geo_coor[0]**2 + geo_coor[1]**2)**0.5)*360./(2.*math.pi)

    #put the right ascension and declination together in one array as [:,0] and [:,1]
    geo_rad = np.zeros((nt,2), float)
    geo_rad[:,0] = geo_ra
    geo_rad[:,1] = geo_dec
    return geo_coor, geo_rad

day = 150926
sat_data = read_poshist(day)
sat_time = sat_data[0]
sat_pos = sat_data[1]
sat_lat = sat_data[2]
sat_lon = sat_data[3]
sat_q = sat_data[4]
sat_coor = [1, 1, 1]
geo_loc = calc_sat_to_geo(sat_q, sat_coor)
geo_coor = geo_loc[0]
geo_rad = geo_loc[1]

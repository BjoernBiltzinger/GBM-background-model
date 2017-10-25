#!/usr/bin python2.7

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
import ephem
from numpy import linalg as LA




#define the detector orientations on the FERMI-satellite
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










#get position data of the satellite
def read_poshist(day):
    """This function reads a posthist file and stores the data in arrays of the form: sat_time, sat_pos, sat_lat, sat_lon, sat_q.\n
    Input read_poshist ( day = YYMMDD )\n 0 = time\n 1 = position (x, y, z)\n 2 = latitude\n 3 = longitude\n 4 = quaternion matrix (q1, q2, q3, q4)"""
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









#get the data from a cspec-file
def read_cspec(detector, day, seconds = 0):
    """This function reads a ctime file and stores the data in arrays of the form: emin, emax, total_counts, echan_counts, exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n Input read_ctime ( detector, day = YYMMDD, seconds = SSS )\n 0 = emin\n 1 = emax\n 2 = total_counts\n 3 = echan_counts\n 4 = exptime\n 5 = total_rate\n 6 = echan_rate\n 7 = cstart\n 8 = cstop\n 9 = gtstart\n 10 = gtstop\n"""
    if seconds == 0:
        filename = 'glg_cspec_' + str(detector) + '_' + str(day) + '_v00.pha'
    else:
        filename = 'glg_cspec_' + str(detector) + '_' + 'bn' + str(day) + str(seconds) + '_v00.pha'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(os.path.dirname(__dir__), 'cspec')
    filepath = os.path.join(fits_path, str(filename))
    fits = pyfits.open(filepath)
    energy = fits[1].data
    spectrum = fits[2].data
    goodtime = fits[3].data
    fits.close()
    emin = energy['E_MIN'] #lower limit of the energy channels
    emax = energy['E_MAX'] #upper limit of the energy channels
    counts = spectrum['COUNTS']
    total_counts = np.sum(counts, axis=1) #total number of counts for each time intervall
    echan_counts = np.vstack(([counts[:,0].T], [counts[:,1].T], [counts[:,2].T], [counts[:,3].T], [counts[:,4].T], [counts[:,5].T], [counts[:,6].T], [counts[:,7].T], [counts[:,8].T], [counts[:,9].T], [counts[:,10].T], [counts[:,11].T], [counts[:,12].T], [counts[:,13].T], [counts[:,14].T], [counts[:,15].T], [counts[:,16].T], [counts[:,17].T], [counts[:,18].T], [counts[:,19].T], [counts[:,20].T], [counts[:,21].T], [counts[:,22].T], [counts[:,23].T], [counts[:,24].T], [counts[:,25].T], [counts[:,26].T], [counts[:,27].T], [counts[:,28].T], [counts[:,29].T], [counts[:,30].T], [counts[:,31].T], [counts[:,32].T], [counts[:,33].T], [counts[:,34].T], [counts[:,35].T], [counts[:,36].T], [counts[:,37].T], [counts[:,38].T], [counts[:,39].T], [counts[:,40].T], [counts[:,41].T], [counts[:,42].T], [counts[:,43].T], [counts[:,44].T], [counts[:,45].T], [counts[:,46].T], [counts[:,47].T], [counts[:,48].T], [counts[:,49].T], [counts[:,50].T], [counts[:,51].T], [counts[:,52].T], [counts[:,53].T], [counts[:,54].T], [counts[:,55].T], [counts[:,56].T], [counts[:,57].T], [counts[:,58].T], [counts[:,59].T], [counts[:,60].T], [counts[:,61].T], [counts[:,62].T], [counts[:,63].T], [counts[:,64].T], [counts[:,65].T], [counts[:,66].T], [counts[:,67].T], [counts[:,68].T], [counts[:,69].T], [counts[:,70].T], [counts[:,71].T], [counts[:,72].T], [counts[:,73].T], [counts[:,74].T], [counts[:,75].T], [counts[:,76].T], [counts[:,77].T], [counts[:,78].T], [counts[:,79].T], [counts[:,80].T], [counts[:,81].T], [counts[:,82].T], [counts[:,83].T], [counts[:,84].T], [counts[:,85].T], [counts[:,86].T], [counts[:,87].T], [counts[:,88].T], [counts[:,89].T], [counts[:,90].T], [counts[:,91].T], [counts[:,92].T], [counts[:,93].T], [counts[:,94].T], [counts[:,95].T], [counts[:,96].T], [counts[:,97].T], [counts[:,98].T], [counts[:,99].T], [counts[:,100].T], [counts[:,101].T], [counts[:,102].T], [counts[:,103].T], [counts[:,104].T], [counts[:,105].T], [counts[:,106].T], [counts[:,107].T], [counts[:,108].T], [counts[:,109].T], [counts[:,110].T], [counts[:,111].T], [counts[:,112].T], [counts[:,113].T], [counts[:,114].T], [counts[:,115].T], [counts[:,116].T], [counts[:,117].T], [counts[:,118].T], [counts[:,119].T], [counts[:,120].T], [counts[:,121].T], [counts[:,122].T], [counts[:,123].T], [counts[:,124].T], [counts[:,125].T], [counts[:,126].T], [counts[:,127].T], ))
    exptime = spectrum['EXPOSURE'] #length of the time intervall
    total_rate = np.divide(total_counts, exptime) #total count rate for each time intervall
    echan_rate = np.divide(echan_counts, exptime) #count rate per time intervall for each energy channel
    cstart = spectrum['TIME'] #start time of the time intervall
    cstop = spectrum['ENDTIME'] #end time of the time intervall
    gtstart = goodtime['START'] #start time of data collecting times (exiting SAA)
    gtstop = goodtime['STOP'] #end time of data collecting times (entering SAA)
    #times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
    return emin, emax, total_counts, echan_counts, exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop










#get the data from a ctime-file
def read_ctime(detector, day, seconds = 0):
    """This function reads a ctime file and stores the data in arrays of the form: emin, emax, total_counts, echan_counts, exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n Input read_ctime ( detector, day = YYMMDD, seconds = SSS )\n 0 = emin\n 1 = emax\n 2 = total_counts\n 3 = echan_counts\n 4 = exptime\n 5 = total_rate\n 6 = echan_rate\n 7 = cstart\n 8 = cstop\n 9 = gtstart\n 10 = gtstop\n"""
    if seconds == 0:
        filename = 'glg_ctime_' + str(detector) + '_' + str(day) + '_v00.pha'
    else:
        filename = 'glg_ctime_' + str(detector) + '_' + 'bn' + str(day) + str(seconds) + '_v00.pha'
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(os.path.dirname(__dir__), 'ctime')
    filepath = os.path.join(fits_path, str(filename))
    fits = pyfits.open(filepath)
    energy = fits[1].data
    spectrum = fits[2].data
    goodtime = fits[3].data
    fits.close()
    emin = energy['E_MIN'] #lower limit of the energy channels
    emax = energy['E_MAX'] #upper limit of the energy channels
    counts = spectrum['COUNTS']
    total_counts = np.sum(counts, axis=1) #total number of counts for each time intervall
    echan_counts = np.vstack(([counts[:,0].T], [counts[:,1].T], [counts[:,2].T], [counts[:,3].T], [counts[:,4].T], [counts[:,5].T], [counts[:,6].T], [counts[:,7].T]))
    exptime = spectrum['EXPOSURE'] #length of the time intervall
    total_rate = np.divide(total_counts, exptime) #total count rate for each time intervall
    echan_rate = np.divide(echan_counts, exptime) #count rate per time intervall for each energy channel
    cstart = spectrum['TIME'] #start time of the time intervall
    cstop = spectrum['ENDTIME'] #end time of the time intervall
    gtstart = goodtime['START'] #start time of data collecting times (exiting SAA)
    gtstop = goodtime['STOP'] #end time of data collecting times (entering SAA)
    #times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
    return emin, emax, total_counts, echan_counts, exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop










#convert satellite coordinates into geographical coordinates
def calc_sat_to_geo(sat_q, sat_coor):
    """This function converts the satellite coordinates into geographical coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: geo_coor, geo_rad\n
    Input calc_sat_to_geo ( sat_q = quaternion-matrix, sat_coor = 3D-array(x, y, z) )\n
    0 = geo_coor\n
    1 = geo_rad"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    nt=np.size(sat_q[0])
    scx=np.zeros((nt,3),float)
    scx[:,0]=(sat_q[3]**2 + sat_q[0]**2 - sat_q[1]**2 - sat_q[2]**2)
    scx[:,1]=2.*(sat_q[0]*sat_q[1] + sat_q[3]*sat_q[2])
    scx[:,2]=2.*(sat_q[0]*sat_q[2] - sat_q[3]*sat_q[1])
    scy=np.zeros((nt,3),float)
    scy[:,0]=2.*(sat_q[0]*sat_q[1] - sat_q[3]*sat_q[2])
    scy[:,1]=(sat_q[3]**2 - sat_q[0]**2 + sat_q[1]**2 - sat_q[2]**2)
    scy[:,2]=2.*(sat_q[1]*sat_q[2] + sat_q[3]*sat_q[0])
    scz=np.zeros((nt,3),float)
    scz[:,0]=2.*(sat_q[0]*sat_q[2] + sat_q[3]*sat_q[1])
    scz[:,1]=2.*(sat_q[1]*sat_q[2] - sat_q[3]*sat_q[0])
    scz[:,2]=(sat_q[3]**2 - sat_q[0]**2 - sat_q[1]**2 + sat_q[2]**2)

    #create geo_to_sat rotation matrix
    sat_mat = np.array([scx, scy, scz])

    #create sat_to_geo rotation matrix
    geo_mat = np.transpose(sat_mat)

    #convert sat_coordinates to geo_coordinates
    sat_coor = np.array(sat_coor)
    #print geo_mat
    #print sat_coor
    geo_coor=np.zeros((3,nt),float)
    geo_coor[0]=geo_mat[0,:,0]*sat_coor[0]+geo_mat[0,:,1]*sat_coor[1]+geo_mat[0,:,2]*sat_coor[2]
    geo_coor[1]=geo_mat[1,:,0]*sat_coor[0]+geo_mat[1,:,1]*sat_coor[1]+geo_mat[1,:,2]*sat_coor[2]
    geo_coor[2]=geo_mat[2,:,0]*sat_coor[0]+geo_mat[2,:,1]*sat_coor[1]+geo_mat[2,:,2]*sat_coor[2]

    #calculate the ra and dec for the satellite axes
    #print geo_coor[0], geo_coor[1], geo_coor[2]
    geo_ra = np.arctan2(-geo_coor[1], -geo_coor[0])*360./(2.*math.pi)+180.
    geo_dec = np.arctan(geo_coor[2]/(geo_coor[0]**2 + geo_coor[1]**2)**0.5)*360./(2.*math.pi)
    #put ra and dec together in one array as [:,0] and [:,1]
    geo_rad = np.zeros((nt,2), float)
    geo_rad[:,0] = geo_ra
    geo_rad[:,1] = geo_dec
    return geo_coor, geo_rad










#convert geographical coordinates into satellite coordinates
def calc_geo_to_sat(sat_q, geo_coor):
    """This function converts the geographical coordinates into satellite coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: sat_coor, sat_rad\n
    Input calc_geo_to_sat ( sat_q = quaternion-matrix, geo_coor = 3D-array(x, y, z) )\n
    0 = sat_coor\n
    1 = sat_rad"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    nt=np.size(sat_q[0])
    scx=np.zeros((nt,3),float)
    scx[:,0]=(sat_q[3]**2 + sat_q[0]**2 - sat_q[1]**2 - sat_q[2]**2)
    scx[:,1]=2.*(sat_q[0]*sat_q[1] + sat_q[3]*sat_q[2])
    scx[:,2]=2.*(sat_q[0]*sat_q[2] - sat_q[3]*sat_q[1])
    scy=np.zeros((nt,3),float)
    scy[:,0]=2.*(sat_q[0]*sat_q[1] - sat_q[3]*sat_q[2])
    scy[:,1]=(sat_q[3]**2 - sat_q[0]**2 + sat_q[1]**2 - sat_q[2]**2)
    scy[:,2]=2.*(sat_q[1]*sat_q[2] + sat_q[3]*sat_q[0])
    scz=np.zeros((nt,3),float)
    scz[:,0]=2.*(sat_q[0]*sat_q[2] + sat_q[3]*sat_q[1])
    scz[:,1]=2.*(sat_q[1]*sat_q[2] - sat_q[3]*sat_q[0])
    scz[:,2]=(sat_q[3]**2 - sat_q[0]**2 - sat_q[1]**2 + sat_q[2]**2)
    
    #create geo_to_sat rotation matrix
    sat_mat = np.array([scx, scy, scz])

    #convert geo_coordinates to sat_coordinates
    geo_coor = np.array(geo_coor)
    sat_coor=np.zeros((3,nt),float)
    sat_coor[0]=sat_mat[0,:,0]*geo_coor[0]+sat_mat[0,:,1]*geo_coor[1]+sat_mat[0,:,2]*geo_coor[2]
    sat_coor[1]=sat_mat[1,:,0]*geo_coor[0]+sat_mat[1,:,1]*geo_coor[1]+sat_mat[1,:,2]*geo_coor[2]
    sat_coor[2]=sat_mat[2,:,0]*geo_coor[0]+sat_mat[2,:,1]*geo_coor[1]+sat_mat[2,:,2]*geo_coor[2]

    #calculate ra and dec
    sat_az = np.arctan2(-sat_coor[1], -sat_coor[0])*360./(2.*math.pi)+180.
    sat_zen = 90. - np.arctan((sat_coor[2]/(sat_coor[0]**2 + sat_coor[1]**2)**0.5))*360./(2.*math.pi)

    #put az and zen together in one array as [:,0] and [:,1]
    sat_rad = np.zeros((nt,2), float)
    sat_rad[:,0] = np.array(sat_az)
    sat_rad[:,1] = np.array(sat_zen)
    return sat_coor, sat_rad










#calculate the detector orientation in geographical coordinates for one detector
def calc_det_or( detector, day ):
    """This function reads a posthist file and the detector assembly table to calculate the detector orientation and stores it in arrays of the form: det_coor, det_rad, sat_pos, sat_time\n
    Input calc_det_or( detector = n0/n1/b0.., day = YYMMDD )\n
    0 = det_coor (x, y, z)\n
    1 = det_rad (ra, dec)\n
    2 = sat_pos (x, y, z)\n
    3 = sat_time"""

    #get satellite data for the convertion
    sat_data = read_poshist(day)
    sat_time = sat_data[0]
    sat_pos = sat_data[1]
    sat_q = sat_data[4]

    #get detector orientation data (in sat-coordinates) from the defined detector-class
    az = detector.azimuth
    zen = detector.zenith
    det_pos = np.array([math.cos(az)*math.sin(zen), math.sin(az)*math.sin(zen), math.cos(zen)])
    
    #convert the orientation in geo-coordinates
    det_geo = calc_sat_to_geo(sat_q, det_pos)
    det_coor = det_geo[0] #unit-vector
    det_rad = det_geo[1]
    
    return det_coor, det_rad, sat_pos, sat_time










#calculate the position of the sun for one day
def calc_sun_pos(day):
    """This function calculates the course of the sun during a certain day and stores the data in arrays of the form: sun_pos, sun_rad\n
    Input calc_sun_pos ( day = YYMMDD )\n
    0 = sun_pos\n
    1 = sun_rad"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    data = read_poshist(day)
    sat_time = np.array(data[0])/(3600*24)+36890.50074287037037037
    sat_pos = np.array(data[1])
    sun = ephem.Sun()
    sun_ra = []
    sun_dec = []
    for i in range(0, len(sat_time)):
        sun.compute(sat_time[i])
        sun_ra.append(sun.ra)
        sun_dec.append(sun.dec)
    sun_rad = np.array([sun_ra, sun_dec])
    sun_pos = np.array([np.cos(sun_ra)*np.cos(sun_dec), np.sin(sun_ra)*np.cos(sun_dec), np.sin(sun_dec)])
    return sun_pos, sun_rad










#calculate the orientation of the sun on one day with respect to one specific detector
def calc_sun_ang(detector, day):
    """This function calculates the sun orientation for one detector and stores the data in arrays of the form: sun_ang, sat_time\n
    Input calc_sun_ang ( detector, day = JJMMDD )\n
    0 = sun_ang\n
    1 = sat_time"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    data_det = calc_det_or(detector, day)
    det_coor = data_det[0]
    det_rad = data_det[1]
    sat_pos = data_det[2]
    sat_time = np.array(data_det[3])
    data_sun = calc_sun_pos(day)
    sun_pos = data_sun[0]
    sun_rad = data_sun[1]
    scalar_product = det_coor[0]*sun_pos[0] + det_coor[1]*sun_pos[1] + det_coor[2]*sun_pos[2]
    ang_det_sun = np.arccos(scalar_product)
    sun_ang = (ang_det_sun)*360./(2.*math.pi)
    sun_ang = np.array(sun_ang)
    return sun_ang, sat_time










#calculate the earth occultation on one day for a specific detector
def calc_earth_ang(detector, day):
    """This function calculates the earth occultation for one detector and stores the data in arrays of the form: earth_ang, sat_time\n
    Input calc_earth_ang ( detector, day = JJMMDD )\n
    0 = earth_ang\n
    1 = sat_time"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    data = calc_det_or(detector, day)
    det_coor = data[0]
    det_rad = data[1]
    sat_pos = data[2]
    sat_time = np.array(data[3])
    sat_dist = LA.norm(sat_pos, axis=0)
    sat_pos_unit = sat_pos/sat_dist
    geo_pos_unit = -sat_pos_unit
    scalar_product = det_coor[0]*geo_pos_unit[0] + det_coor[1]*geo_pos_unit[1] + det_coor[2]*geo_pos_unit[2]
    ang_det_geo = np.arccos(scalar_product)
    earth_ang = (ang_det_geo)*360./(2.*math.pi)
    earth_ang = np.array(earth_ang)
    return earth_ang, sat_time










#calculate the burst orientation on one day for a specific detector
def calc_burst_ang(detector, day, burst_ra, burst_dec):
    """This function calculates the burst orientation for one detector and stores the data in arrays of the form: burst_ang, sat_time\n
    Input calc_burst_ang ( detector, day = JJMMDD, burst_ra, burst_dec )\n
    0 = burst_ang\n
    1 = sat_time"""
    #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
    data_det = calc_det_or(detector, day)
    det_coor = data_det[0]
    det_rad = data_det[1]
    sat_pos = data_det[2]
    sat_time = np.array(data_det[3])
    burst_rad = np.array([burst_ra, burst_dec], float)
    burst_pos = np.array([np.cos(burst_ra)*np.cos(burst_dec), np.sin(burst_ra)*np.cos(burst_dec), np.sin(burst_dec)], float)
    scalar_product = det_coor[0]*burst_pos[0] + det_coor[1]*burst_pos[1] + det_coor[2]*burst_pos[2]
    ang_det_burst = np.arccos(scalar_product)
    burst_ang = (ang_det_burst)*360./(2.*math.pi)
    burst_ang = np.array(burst_ang)
    return burst_ang, sat_time










#get the saa-polynom from the data-file
def read_saa():
    """This function reads the saa.dat file and returns the polygon in the form: saa[lat][lon]\n
    0 = saa"""
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










#do the calculations
#day = 150926
#detector = n5

day = 151126
burst_ra = 338.61*2.*math.pi/360.
burst_dec = 30.88*2.*math.pi/360.

sat_data = read_poshist(day)
sat_q = sat_data[4]

#for i in range(0, 10):
#    detector = eval('n' + str(i))
#    burst_loc = calc_burst_ang(detector, day, burst_ra, burst_dec)
#    burst_ang = burst_loc[0]
#    sat_time = burst_loc[1]
#    sun_ang = calc_sun_ang(detector, day)[0]
#    earth_ang = calc_earth_ang(detector, day)[0]
#    sun = ephem.Sun()
#    sun.compute(sat_time[25271])
#    sun_ra = sun.ra
#    sun_dec = sun.dec
#    sun_rad = np.array([sun_ra, sun_dec])
#    print burst_ang[25271], sun_ang[25271], earth_ang[25271], sun_rad


#sat_data = read_poshist(day)
#sat_time = sat_data[0]
#sat_pos = sat_data[1]
#sat_lat = sat_data[2]
#sat_lon = sat_data[3]
#sat_q = sat_data[4]
#sat_loc = calc_geo_to_sat(sat_q, sat_pos)
#sat_coor = sat_loc[0]
#sat_rad = sat_loc[1]
#geo_loc = calc_sat_to_geo(sat_q, sat_coor)
#geo_coor = geo_loc[0]
#geo_rad = geo_loc[1]

#ctime_n5 = read_ctime(detector.__name__, day)#emin, emax, total_counts, echan_counts, exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop
#total_counts = ctime_n5[2]
#total_rate = ctime_n5[5]
#cstart = ctime_n5[7]
#cstop = ctime_n5[8]

#count_time = np.array((cstart+cstop)/2)

#sun_or = calc_sun_ang(detector, day)
#sun_ang = sun_or[0]
#sat_time = sun_or[1]

#earth_or = calc_earth_ang(detector, day)
#earth_ang = earth_or[0]

#daytime_sun = (sat_time - sat_time[0] + 5)/3600.
#daytime_counts = (count_time - sat_time[0] + 5)/3600.

#plt.plot(daytime_counts, total_counts, 'b-', daytime_sun, sun_ang, 'y-', daytime_sun, earth_ang, 'r-')

#plt.xlabel('time of day')
#plt.ylabel('counts')

#plt.title('Sun- and Earth-angle of the ' + detector.__name__ + '-detector on the 26th Sept 2015')

#plt.axis([0, 24.1, 0, 500])

#plt.show()

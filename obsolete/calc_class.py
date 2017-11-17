#!/usr/bin python2.7

import math

import ephem
import numpy as np
from numpy import linalg as LA

from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()

class calculate:
    """This class contains all calculation functions needed for the GBM background model"""
    
    def burst_ang(self, detector, day, burst_ra, burst_dec):
        """This function calculates the burst orientation for one detector and stores the data in arrays of the form: burst_ang, sat_time\n
        Input:\n
        calculate.burst_ang ( detector, day = JJMMDD, burst_ra, burst_dec )\n
        Output:\n
        0 = angle between the burst and the detector\n
        1 = time (MET) in seconds"""
        
        #get the detector and satellite data
        data_det = self.det_or(detector, day)
        det_coor = data_det[0] #unit-vector of the detector orientation
        det_rad = data_det[1] #detector orientation in right ascension and declination
        sat_pos = data_det[2] #position of the satellite
        sat_time = np.array(data_det[3]) #time (MET) in seconds

        #convert the burst angles into a unit-vector
        burst_rad = np.array([burst_ra, burst_dec], float)
        burst_pos = np.array([np.cos(burst_ra)*np.cos(burst_dec), np.sin(burst_ra)*np.cos(burst_dec), np.sin(burst_dec)], float) #unit-vector pointing to the burst location
    
        #calculate the angle between the burst location and the detector orientation
        scalar_product = det_coor[0]*burst_pos[0] + det_coor[1]*burst_pos[1] + det_coor[2]*burst_pos[2]
        ang_det_burst = np.arccos(scalar_product)
        burst_ang = (ang_det_burst)*360./(2.*math.pi) #convert to degrees
        burst_ang = np.array(burst_ang)
        return burst_ang, sat_time










    def det_or(self, detector, day):
        """This function reads a posthist file and the detector assembly table to calculate the detector orientation and stores it in arrays of the form: det_coor, det_rad, sat_pos, sat_time\n
        Input:\n
        calculate.det_or( detector = n0/n1/b0.., day = YYMMDD )\n
        Output:\n
        0 = detector coordinates[x[], y[], z[]]\n
        1 = detector geocentric angles [(right ascension, declination)]\n
        2 = satellite position [x[], y[], z[]]\n
        3 = time (MET) in seconds"""

        #get satellite data for the convertion
        sat_data = rf.poshist(day)
        sat_time = sat_data[0]
        sat_pos = sat_data[1]
        sat_q = sat_data[4]

        #get detector orientation data (in sat-coordinates) from the defined detector-class
        az = det.detector.azimuth
        zen = det.detector.zenith
        det_pos = np.array([math.cos(az)*math.sin(zen), math.sin(az)*math.sin(zen), math.cos(zen)]) #convert into unit-vector in the satellite coordinate system
    
        #convert the orientation in geo-coordinates
        det_geo = self.sat_to_geo(sat_q, det_pos)
        det_coor = det_geo[0] #unit-vector
        det_rad = det_geo[1]
        return det_coor, det_rad, sat_pos, sat_time










    def earth_ang(self, detector, day):
        """This function calculates the earth occultation for one detector and stores the data in arrays of the form: earth_ang, sat_time\n
        Input:\n
        calculate.earth_ang ( detector, day = JJMMDD )\n
        Output:\n
        0 = angle between the detector orientation and the earth position\n
        1 = time (MET) in seconds"""
        
        #get the detector and satellite data
        data = self.det_or(detector, day)
        det_coor = data[0] #unit-vector of the detector orientation
        det_rad = data[1] #detector orientation in right ascension and declination
        sat_pos = data[2] #position of the satellite
        sat_time = np.array(data[3]) #time (MET) in seconds
    
        #calculate the earth location unit-vector
        sat_dist = LA.norm(sat_pos, axis=0) #get the altitude of the satellite (length of the position vector)
        sat_pos_unit = sat_pos/sat_dist #convert the position vector into a unit-vector
        geo_pos_unit = -sat_pos_unit
        
        #calculate the angle between the earth location and the detector orientation
        scalar_product = det_coor[0]*geo_pos_unit[0] + det_coor[1]*geo_pos_unit[1] + det_coor[2]*geo_pos_unit[2]
        ang_det_geo = np.arccos(scalar_product)
        earth_ang = ang_det_geo*360./(2.*math.pi)
        earth_ang = np.array(earth_ang)
        return earth_ang, sat_time










    def geo_to_sat(self, sat_q, geo_coor):
        """This function converts the geographical coordinates into satellite coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: sat_coor, sat_rad\n
        Input:\n
        calculate.geo_to_sat ( sat_q = quaternion-matrix, geo_coor = 3D-array(x, y, z) )\n
        Output:\n
        0 = satellite coordinates[x[], y[], z[]]\n
        1 = satellite angle[(azimuth, zenith)]"""
        
        #calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
        nt = np.size(sat_q[0])
        scx = np.zeros((nt,3),float)
        scx[:,0] = (sat_q[0]**2 - sat_q[1]**2 - sat_q[2]**2 + sat_q[3]**2)
        scx[:,1] = 2.*(sat_q[0]*sat_q[1] + sat_q[3]*sat_q[2])
        scx[:,2] = 2.*(sat_q[0]*sat_q[2] - sat_q[3]*sat_q[1])
        scy = np.zeros((nt,3),float)
        scy[:,0] = 2.*(sat_q[0]*sat_q[1] - sat_q[3]*sat_q[2])
        scy[:,1] = (-sat_q[0]**2 + sat_q[1]**2 - sat_q[2]**2 + sat_q[3]**2)
        scy[:,2] = 2.*(sat_q[1]*sat_q[2] + sat_q[3]*sat_q[0])
        scz = np.zeros((nt,3),float)
        scz[:,0] = 2.*(sat_q[0]*sat_q[2] + sat_q[3]*sat_q[1])
        scz[:,1] = 2.*(sat_q[1]*sat_q[2] - sat_q[3]*sat_q[0])
        scz[:,2] = (-sat_q[0]**2 - sat_q[1]**2 + sat_q[2]**2 + sat_q[3]**2)
        
        #create geo_to_sat rotation matrix
        sat_mat = np.array([scx, scy, scz])
        
        #convert geo_coordinates to sat_coordinates
        geo_coor = np.array(geo_coor)
        sat_coor = np.zeros((3,nt),float)
        sat_coor[0] = sat_mat[0,:,0]*geo_coor[0] + sat_mat[0,:,1]*geo_coor[1] + sat_mat[0,:,2]*geo_coor[2]
        sat_coor[1] = sat_mat[1,:,0]*geo_coor[0] + sat_mat[1,:,1]*geo_coor[1] + sat_mat[1,:,2]*geo_coor[2]
        sat_coor[2] = sat_mat[2,:,0]*geo_coor[0] + sat_mat[2,:,1]*geo_coor[1] + sat_mat[2,:,2]*geo_coor[2]
        
        #calculate the azimuth and zenith
        sat_az = np.arctan2(-sat_coor[1], -sat_coor[0])*360./(2.*math.pi)+180.
        sat_zen = 90. - np.arctan((sat_coor[2]/(sat_coor[0]**2 + sat_coor[1]**2)**0.5))*360./(2.*math.pi)
        
        #put azimuth and zenith together in one array as [:,0] and [:,1]
        sat_rad = np.zeros((nt,2), float)
        sat_rad[:,0] = np.array(sat_az)
        sat_rad[:,1] = np.array(sat_zen)
        return sat_coor, sat_rad 










    def sat_to_geo(self, sat_q, sat_coor):
        """This function converts the satellite coordinates into geographical coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: geo_coor, geo_rad\n
        Input:\n
        calculate.sat_to_geo ( sat_q = quaternion-matrix, sat_coor = 3D-array(x, y, z) )\n
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










    def sun_ang(self, detector, day):
        """This function calculates the sun orientation for one detector and stores the data in arrays of the form: sun_ang, sat_time\n
        Input:\n
        calculate.sun_ang ( detector, day = JJMMDD )\n
        Output:\n
        0 = angle between the sun location and the detector orientation\n
        1 = time (MET) in seconds"""
        
        #get the detector and satellite data
        data_det = self.det_or(detector, day)
        det_coor = data_det[0] #unit-vector of the detector orientation
        det_rad = data_det[1] #detector orientation in right ascension and declination
        sat_pos = data_det[2] #position of the satellite
        sat_time = np.array(data_det[3]) #time (MET) in seconds
        
        #get the sun data
        data_sun = self.sun_pos(day)
        sun_pos = data_sun[0]
        sun_rad = data_sun[1]
        
        #calculate the angle between the sun location and the detector orientation
        scalar_product = det_coor[0]*sun_pos[0] + det_coor[1]*sun_pos[1] + det_coor[2]*sun_pos[2]
        ang_det_sun = np.arccos(scalar_product)
        sun_ang = (ang_det_sun)*360./(2.*math.pi)
        sun_ang = np.array(sun_ang)
        return sun_ang, sat_time










    def sun_pos(self, day):
        """This function calculates the course of the sun during a certain day and stores the data in arrays of the form: sun_pos, sun_rad\n
        Input:\n
        calculate.sun_pos ( day = YYMMDD )\n
        Output:\n
        0 = unit-vector of the sun position[x[], y[], z[]]\n
        1 = geocentric angles of the sun position[(right ascension, declination)]"""
        
        #get the satellite data
        data = rf.poshist(day)
        sat_time = np.array(data[0])/(3600*24)+36890.50074287037037037
        sat_pos = np.array(data[1])
        
        #calculate the geocentric angles of the sun for each time-bin
        sun = ephem.Sun()
        sun_ra = []
        sun_dec = []
        for i in range(0, len(sat_time)):
            sun.compute(sat_time[i]) #generate the sun information from the ephem module for the sat_time[i]
            sun_ra.append(sun.ra) #add to the right ascension vector
            sun_dec.append(sun.dec) #add to the declination vector
        
        #put the right ascension and declination together in one array as [:,0] and [:,1]
        sun_rad = np.zeros((len(sun_ra),2), float)
        sun_rad[:,0] = sun_ra
        sun_rad[:,1] = sun_dec
        sun_rad = np.array(sun_rad)
        
        #derive the unit-vector of the sun location in geocentric coordinates
        sun_pos = np.array([np.cos(sun_ra)*np.cos(sun_dec), np.sin(sun_ra)*np.cos(sun_dec), np.sin(sun_dec)])
        return sun_pos, sun_rad


calc = calculate()
detector = calc.n5
sun_ang = calc.sun_ang(detector, 150926)[0]
print sun_ang

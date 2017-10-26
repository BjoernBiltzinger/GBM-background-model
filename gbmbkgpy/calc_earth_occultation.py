#!/usr/bin python2.7

import subprocess
from subprocess import Popen, PIPE
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from scipy import interpolate
import scipy.optimize as optimization
from scipy import integrate
from astropy.time import Time
from astropy.table import Table
from astropy.io import fits
import fileinput
from datetime import datetime
from work_module import calculate
from work_module import detector
from work_module import readfile
from work_module import writefile
calc = calculate()
det = detector()
rf = readfile()
wf = writefile()

def calc_earth_occultation(day, detector, bin_time_mid = 0):
    '''This function calculates the earth occultation area fraction and stores it in an array of the form: earth_occ\n
    Input:\n
    calc_ang_resp ( ang (in degrees), detector )\n
    Output:\n
    0 = effective angle'''
    
    #get the position data of the satellite for the given day
    poshist = rf.poshist_bin(day, bin_time_mid, detector)
    sat_time_bin = poshist[0]
    sat_pos_bin = poshist[1]
    sat_lat_bin = poshist[2]
    sat_lon_bin = poshist[3]
    bin_time_mid = poshist[5]
    
    #get the distance from the satellite to the center of the earth
    '''sat_dist = LA.norm(sat_pos_bin, axis=0)
    sat_dist_mean = np.mean(sat_dist)'''
    sat_dist = 6912000. #it was decided to use a fix distance to minimize the computing needed. In reality the distance is decreasing with time as the satellite is falling towards the earth. This is about the distance of the satellite during the year 2015
    
    #get the earth_radius at the satellite's position
    '''ell_a = 6378137
    ell_b = 6356752.3142
    sat_lat_rad = sat_lat_bin*2*math.pi/360.
    earth_radius_sat = ell_a * ell_b / ((ell_a**2 * np.sin(sat_lat_rad)**2 + ell_b**2 * np.cos(sat_lat_rad)**2)**0.5)'''
    earth_radius = 6371000.8 #geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
    atmosphere = 12000. #the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
    r = earth_radius + atmosphere #the full radius of the occulting earth-sphere
    
    #define the opening angles of the overlapping cones (earth and detector). The defined angles are just half of the opening angle, from the central line to the surface of the cone.
    theta = math.asin(r/sat_dist) #earth-cone
    opening_ang = 80.0*2*math.pi/360. #detector-cone
    
    #get the angle between the detector direction and the earth direction
    earth_ang = calc.earth_ang_bin(detector, day, bin_time_mid)[0]*2*math.pi/360.
    
    #not used geometric considerations
    '''
    contur_ang = earth_ang - theta #angle between the detector direction and the horizon of the earth
    hor_dist = (sat_dist)**2 - r**2 #distance of the horizon to the satellite
    x = hor_dist*np.cos(contur_ang*2*math.pi/360) #x-distance of the earth in the plane of the detector direction and the earth direction (not sure whether this is needed)
    y = hor_dist*np.sin(contur_ang*2*math.pi/360) #y-distance of the earth in the plane of the detector direction and the earth direction (not sure whether this is needed)
    d = x*np.tan(opening_ang*2*math.pi/360) #radius of the detector cone at the x-distance of the earth (not sure whether this is needed)
    l = r+d-y #self-defined parameter (not sure whether this is needed)'''
    
    #geometric considerations for the two overlapping spherical cap problem
    phi = math.pi/2 - earth_ang #angle between the earth-direction and the axis perpendicular to the detector-orientation on the earth-detector-plane
    f = (np.cos(theta) - np.cos(opening_ang)*np.sin(phi))/(np.cos(phi)) #distance perpendicular to the detector-orientation to the intersection-plane of the spherical caps
    beta = np.arctan2(f,(np.cos(opening_ang))) #angle of the intersection-plane to the detector orientation
    beta2 = earth_ang - beta #angle of the intersection-plane to the earth-orientation
    a = np.sqrt(1 - f**2 - (np.cos(opening_ang))**2) #diagonal of the overlapping crosssections of the spherical caps
    gamma_max = np.arctan(a/(2*f)) #angle of the intersectionpoints on the plane perpendicular to the detector-orientation (top-view)
    
    #same considerations for the earth-component
    phi_e = math.pi/2 - earth_ang #angle between the earth-direction and the axis perpendicular to the detector-orientation on the earth-detector-plane
    f_e = (np.cos(opening_ang) - np.cos(theta)*np.sin(phi))/(np.cos(phi)) #distance perpendicular to the detector-orientation to the intersection-plane of the spherical caps
    beta_e = np.arctan2(f_e,(np.cos(theta))) #angle of the intersection-plane to the detector orientation
    beta2_e = earth_ang - beta_e #angle of the intersection-plane to the earth-orientation
    a_e = np.sqrt(1 - f_e**2 - (np.cos(theta))**2) #diagonal of the overlapping crosssections of the spherical caps
    gamma_max_e = np.arctan(a_e/(2*f_e)) #angle of the intersectionpoints on the plane perpendicular to the detector-orientation (top-view)
    
    A_d = []
    
    for i in range(0, int(len(bin_time_mid))):#/10)):
        def func1(Phi, gamma):
            return math.sin(Phi)
        def bounds_y():
            return [-gamma_max[i], gamma_max[i]]
        def bounds_x(gamma):
            return [math.atan(math.tan(beta[i])/math.cos(gamma)), opening_ang]
        area = integrate.nquad(func1, [bounds_x, bounds_y])
        print i, area[0]
        A_d.append(area[0])
    
    A_d = np.array(A_d)
    
    A_e = []
    
    for i in range(0, int(len(bin_time_mid))):
        #/10)):
        def func2(Phi, gamma):
            return math.sin(Phi)
        def bounds_y():
            return [-gamma_max_e[i], gamma_max_e[i]]
        def bounds_x(gamma):
            return [math.atan(math.tan(beta_e[i])/math.cos(gamma)), theta]
        area = integrate.nquad(func2, [bounds_x, bounds_y])
        print i, area[0]
        A_e.append(area[0])
    
    A_e = np.array(A_e)
    
    A_d_an = -2*gamma_max*np.sin(opening_ang) + ((np.sign(np.cos(gamma_max)))*np.arcsin(np.fabs(np.cos(beta))*np.sin(gamma_max))) - ((np.sign(np.cos(-gamma_max)))*np.arcsin(np.fabs(np.cos(beta))*np.sin(-gamma_max)))
    
    A_e_an = -2*gamma_max_e*np.sin(theta) + ((np.sign(np.cos(gamma_max_e)))*np.arcsin(np.fabs(np.cos(beta_e))*np.sin(gamma_max_e))) - ((np.sign(np.cos(-gamma_max_e)))*np.arcsin(np.fabs(np.cos(beta_e))*np.sin(-gamma_max_e)))
    
    A_d_an2 = 2*(np.arctan2((np.sqrt(-(np.tan(beta))**2/((np.sin(opening_ang))**2) + (np.tan(beta))**2 + 1)*np.sin(opening_ang)),np.tan(beta))-np.cos(opening_ang)*np.arccos(np.tan(beta)/np.tan(opening_ang)) - (np.arctan2((np.sqrt(-(np.tan(beta))**2/((np.sin(beta))**2) + (np.tan(beta))**2 + 1)*np.sin(beta)),np.tan(beta))-np.cos(beta)*np.arccos(np.tan(beta)/np.tan(beta))))
    
    A_e_an2 = 2*(np.arctan2((np.sqrt(-(np.tan(beta_e))**2/((np.sin(theta))**2) + (np.tan(beta_e))**2 + 1)*np.sin(theta)),np.tan(beta_e))-np.cos(theta)*np.arccos(np.tan(beta_e)/np.tan(theta)) - (np.arctan2((np.sqrt(-(np.tan(beta_e))**2/((np.sin(beta_e))**2) + (np.tan(beta_e))**2 + 1)*np.sin(beta_e)),np.tan(beta_e))-np.cos(beta_e)*np.arccos(np.tan(beta_e)/np.tan(beta_e))))
    
    y = int(len(bin_time_mid)/10 -1)
    A_e_an2[np.where(earth_ang < beta)] = A_e_an2[np.where(earth_ang < beta)] - 2*math.pi
    #A_e_an[np.where(earth_ang < beta)] = -A_e_an[np.where(earth_ang < beta)]
    A_e[np.where(earth_ang < beta)] = 2*math.pi*(1 - math.cos(theta)) + A_e[np.where(earth_ang < beta)]
    
    A_an2 = A_d_an2 + A_e_an2
    A = A_d + A_e
    
    free_area = 2*math.pi*(1 - math.cos(opening_ang))
    earth_occ = A/free_area
    earth_occ_an2 = A_an2/free_area
    #unocc_area_fraction = 1 - A/free_area
    
    l = math.sqrt(sat_dist**2 + r**2)
    
    A_hend = []
    
    for i in range(0, int(len(bin_time_mid))):
        #/10)):
        def a(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i):
            return -4. * (d/l)**2. * (math.cos(Phi)**2. * math.sin(earth_ang[i])**2 + math.cos(earth_ang[i])**2)
        def b(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):
            return math.cos(earth_ang[i]) * 4. * d / l * (1. + d**2. / l**2. - r**2. / l**2.)
        def c(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):
            return 4. * (d/l)**2. * math.cos(Phi)**2 * math.sin(earth_ang[i])**2 - 1 - 2 * (d/l)**2 + 2. * (r/l)**2. - (d / l)**4. + 2 * (r/l)**2 * (d/l)**2. - (r/l)**4.
        def theta0(Phi, d = sat_dist, l = l, earth_ang = earth_ang[i], r = r):
            D = (b()**2 - 4 * a(Phi) * c(Phi))
            if (D <= 1e-13):
                return 0.
            else:
                #print 'D: %e' % D
                #print 'term: ', (-b()-math.sqrt(D))/(2.*a(Phi))
                return min(math.acos((-b()-math.sqrt(D))/(2.*a(Phi))), opening_ang)
        def theta1(Phi, d = sat_dist, l = l, earth_ang = earth_ang[i], r = r):
            D = (b()**2 - 4 * a(Phi) * c(Phi))
            if (D <= 1e-13):
                return 0.
            else:
                return min(math.acos((-b()+math.sqrt(D))/(2.*a(Phi))), opening_ang)
        '''def phi1(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r, opening_ang = opening_ang):
            c_phi1 = (l**2 + d**2 - r**2 - 2*d*l*math.cos(opening_ang)*math.cos(earth_ang[i]))/(2*d*l*math.sin(opening_ang)*math.sin(earth_ang[i]))
            p10 = math.acos(c_phi1)
            p11 = math.sqrt(-2*d**2 *((2*math.cos(earth_ang[i])**2 -1)*l**2 + r**2) + d**4 + (l**2 - r**2)**2)/(2*d*l*math.sin(earth_ang[i]))
            #p12 = -p12
            p13 = math.pi /2.
            #p14 = -p13
            return min(p10, p11, p13)'''
        
        def func3(Phi):
            return -math.cos(theta1(Phi)) + math.cos(theta0(Phi))
        def f_below_horizon(Phi):
            if (Phi < 0.5 * math.pi):
                return -math.cos(theta1(Phi)) + 1.
            else:
                return math.cos(theta0(Phi)) - 1.
        
        #D = (b()**2 - 4 * a(Phi) * c(Phi))
        
        if earth_ang[i] <= theta:
            area = 2*integrate.quad(f_below_horizon, 0., math.pi)
        elif earth_ang[i] + theta <= opening_ang:
            area = 2*math.pi*(1-math.cos(theta))
        else:
            area = 2*integrate.quad(func3, 0., 0.5 * math.pi)
        print i, area[0]
        A_hend.append(area[0])
    
    A_hend = np.array(A_hend)
    
    earth_occ_hend = A_hend/free_area
    
    #A_d_an2 = A_d_an2[:y]
    #A_e_an2 = A_e_an2[:y]
    #earth_occ_an2 = earth_occ_an2[:y]

    return earth_occ, A_d_an, A_d_an2, A_e_an2, sat_time_bin, A_e_an, A_e, A_d, earth_occ_an2, A_hend, earth_occ_hend, y, earth_ang




day = 150926
detector = det.n5


calculation = calc_earth_occultation(day, detector)
earth_occ = calculation[0]
A_d = calculation[7]
A_d_an = calculation[1]
A_d_an2 = calculation[2]
A_e_an2 = calculation[3]
y = calculation[11]
sat_time_bin = calculation[4]
A_e_an = calculation[5]
A_e = calculation[6]
earth_occ_an2 = calculation[8]
A_hend = calculation[9]
earth_occ_hend = calculation[10]
earth_ang = calculation[12]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#sat_time_bin = np.arange(len(A_d))

plot1 = ax1.plot(sat_time_bin, A_d, 'b-')
plot6 = ax1.plot(sat_time_bin, A_e, 'b--')
plot4 = ax1.plot(sat_time_bin, A_e_an2, 'r--')
#plot2 = ax1.plot(sat_time_bin, A_d_an, 'y-')
#plot3 = ax1.plot(sat_time_bin, A_e_an, 'y--')
plot5 = ax1.plot(sat_time_bin, A_d_an2, 'r-')
plot7 = ax1.plot(sat_time_bin, earth_occ, 'g-')
plot8 = ax1.plot(sat_time_bin, earth_occ_an2, 'g--')
plot9 = ax1.plot(sat_time_bin, A_hend, 'y-')
plot10 = ax1.plot(sat_time_bin, earth_occ_hend, 'y--')
#plot6 = ax1.plot(sat_time_bin, A, 'g-')
#plot8 = ax1.plot(sat_time_bin, unocc_area_fraction, 'b-')
#plot5 = ax1.plot(sat_time_bin, f, 'b--')
plot11 = ax2.plot(sat_time_bin, earth_ang, 'k-' )

#ax1.set_xlim([0, 24.1])
#ax1.set_ylim([0.4, 1.1])
#ax2.set_xlim([0, 24.1])
#ax2.set_ylim([-40, 240])

plt.show()

#!/usr/bin python2.7

import math
import os

import numpy as np
from astropy.io import fits
from scipy import interpolate

from gbmbkgpy.utils.external_prop import ExternalProps, writefile
from obsolete.work_module_refactor import calculate
from obsolete.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()


def calc_earth_occ(angle):
    """This function calculates the overlapping area fraction for a certain earth-angle and stores the data in arrays of the form: opening_ang, earth_occ\n
    Input:\n
    calc_earth_occ ( angle )\n
    Output:\n
    0 = angle of the detector-cone\n
    1 = area fraction of the earth-occulted area to the entire area of the detector-cone"""
    
    #get the distance from the satellite to the center of the earth
    sat_dist = 6912000. #it was decided to use a fix distance to minimize the computing needed. In reality the distance is decreasing with time as the satellite is falling towards the earth. This is about the distance of the satellite during the year 2015
    
    #get the earth_radius at the satellite's position
    earth_radius = 6371000.8 #geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
    atmosphere = 12000. #the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
    r = earth_radius + atmosphere #the full radius of the occulting earth-sphere
    
    #define the opening angles of the overlapping cones (earth and detector). The defined angles are just half of the opening angle, from the central line to the surface of the cone.
    theta = math.asin(r/sat_dist) #earth-cone
    opening_ang = np.arange(math.pi/36000., math.pi/2.+math.pi/36000., math.pi/36000.) #detector-cone
    
    #get the angle between the detector direction and the earth direction
    earth_ang = angle*2.*math.pi/360. #input parameter
    
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
    
    #calculate one part of the overlapping area of the spherical caps. This area belongs to the detector-cone
    A_d_an2 = 2*(np.arctan2((np.sqrt(-(np.tan(beta))**2/((np.sin(opening_ang))**2) + (np.tan(beta))**2 + 1)*np.sin(opening_ang)),np.tan(beta))-np.cos(opening_ang)*np.arccos(np.tan(beta)/np.tan(opening_ang)) - (np.arctan2((np.sqrt(-(np.tan(beta))**2/((np.sin(beta))**2) + (np.tan(beta))**2 + 1)*np.sin(beta)),np.tan(beta))-np.cos(beta)*np.arccos(np.tan(beta)/np.tan(beta))))
    
    #calculate the other part of the overlapping area. This area belongs to the earth-cone
    A_e_an2 = 2*(np.arctan2((np.sqrt(-(np.tan(beta_e))**2/((np.sin(theta))**2) + (np.tan(beta_e))**2 + 1)*np.sin(theta)),np.tan(beta_e))-np.cos(theta)*np.arccos(np.tan(beta_e)/np.tan(theta)) - (np.arctan2((np.sqrt(-(np.tan(beta_e))**2/((np.sin(beta_e))**2) + (np.tan(beta_e))**2 + 1)*np.sin(beta_e)),np.tan(beta_e))-np.cos(beta_e)*np.arccos(np.tan(beta_e)/np.tan(beta_e))))
    
    #take the limitations of trignometric functions into account. -> Get rid of 2*pi jumps
    A_e_an2[np.where(earth_ang < beta)] = A_e_an2[np.where(earth_ang < beta)] - 2*math.pi
    A_d_an2[np.where(f < 0)] = A_d_an2[np.where(f < 0)] - 2*math.pi
    
    #combine the two area segments to get the total area
    A_an2 = A_d_an2 + A_e_an2
    
    #calculate the unocculted area of the detector cone
    free_area = 2*math.pi*(1 - np.cos(opening_ang))
    
    #add values to the overlapping area, where either the detector-cone is completely embedded within the earth-cone or the other way around. Within this function both could be the case, because we are changing the angle of the detector-cone!
    A_an2[np.where(opening_ang <= theta - earth_ang)] = free_area
    A_an2[np.where(opening_ang >= theta + earth_ang)] = 2*math.pi*(1 - np.cos(theta))
    A_an2[np.where(opening_ang <= earth_ang - theta)] = 0.
    
    #if the earth will never be within the detector-cone, the overlapping area will always be 0
    #if earth_ang > opening_ang[-1] + theta:
    #    A_an2 = np.zeros(len(opening_ang))
    
    #Apparently the numeric calculation of the analytic solution doesn't always return a value (probably because of runtime error). As a result there are several 'nan' entries in the A_an2 array. To get rid of those we interpolate over all the calculated solutions. We have chosen enough steps for the opening_ang to eliminate any errors due to this interpolation, because we get enough good results from the calculation.
    tck = interpolate.splrep(opening_ang[np.logical_not(np.isnan(A_an2))], A_an2[np.logical_not(np.isnan(A_an2))], s=0)
    A_an2 = interpolate.splev(opening_ang, tck, der=0)
    
    #calculate the fraction of the occulted area
    earth_occ = A_an2/free_area
    
    #Hendrik's approach to this problem. But somehow it's not working right. Haven't figured out yet why.
    '''l = math.sqrt(sat_dist**2 + r**2)
    
    A_hend = []
    
    for i in range(0, int(len(opening_ang))):
        #/10)):
        def a(Phi, d = sat_dist, l = l, earth_ang = earth_ang):
            return -4. * (d/l)**2. * (math.cos(Phi)**2. * math.sin(earth_ang)**2 + math.cos(earth_ang)**2)
        def b(d = sat_dist, l = l, earth_ang = earth_ang, r = r):
            return math.cos(earth_ang) * 4. * d / l * (1. + d**2. / l**2. - r**2. / l**2.)
        def c(Phi, d = sat_dist, l = l, earth_ang = earth_ang, r = r):
            return 4. * (d/l)**2. * math.cos(Phi)**2 * math.sin(earth_ang)**2 - 1 - 2 * (d/l)**2 + 2. * (r/l)**2. - (d / l)**4. + 2 * (r/l)**2 * (d/l)**2. - (r/l)**4.
        def theta0(Phi, d = sat_dist, l = l, earth_ang = earth_ang, r = r, i = i):
            D = (b()**2 - 4 * a(Phi) * c(Phi))
            if (D <= 1e-13):
                return 0.
            else:
                #print 'D: %e' % D
                #print 'term: ', (-b()-math.sqrt(D))/(2.*a(Phi))
                return min(math.acos((-b()-math.sqrt(D))/(2.*a(Phi))), opening_ang[i])
        def theta1(Phi, d = sat_dist, l = l, earth_ang = earth_ang, r = r, i = i):
            D = (b()**2 - 4 * a(Phi) * c(Phi))
            if (D <= 1e-13):
                return 0.
            else:
                return min(math.acos((-b()+math.sqrt(D))/(2.*a(Phi))), opening_ang[i])
        """def phi1(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r, opening_ang = opening_ang):
            c_phi1 = (l**2 + d**2 - r**2 - 2*d*l*math.cos(opening_ang)*math.cos(earth_ang[i]))/(2*d*l*math.sin(opening_ang)*math.sin(earth_ang[i]))
            p10 = math.acos(c_phi1)
            p11 = math.sqrt(-2*d**2 *((2*math.cos(earth_ang[i])**2 -1)*l**2 + r**2) + d**4 + (l**2 - r**2)**2)/(2*d*l*math.sin(earth_ang[i]))
            #p12 = -p12
            p13 = math.pi /2.
            #p14 = -p13
            return min(p10, p11, p13)"""
        
        def func3(Phi):
            return -math.cos(theta1(Phi)) + math.cos(theta0(Phi))
        def f_below_horizon(Phi):
            if (Phi < 0.5 * math.pi):
                return -math.cos(theta1(Phi)) + 1.
            else:
                return -math.cos(theta0(Phi)) + 1.
        
        #D = (b()**2 - 4 * a(Phi) * c(Phi))
        
        if earth_ang <= theta:
            area = 2*integrate.quad(f_below_horizon, 0., math.pi)
        elif earth_ang + theta <= opening_ang[i]:
            area = 2*math.pi*(1-math.cos(theta))
        else:
            area = 2*integrate.quad(func3, 0., 0.5 * math.pi)
        print i, area[0], earth_ang*360/2/math.pi, theta*360/2/math.pi
        A_hend.append(area[0])
        print opening_ang[i]*360/2/math.pi
    
    A_hend = np.array(A_hend)
    
    earth_occ_hend = A_hend/free_area'''
    
    ### extend to total view of detector ###
    
    #opening_ang_big = math.pi - np.flipud(opening_ang)
    
    #free_area_big = 4*math.pi - np.flipud(free_area)
    
    
    #sat_dist = 6912000. #it was decided to use a fix distance to minimize the computing needed. In reality the distance is decreasing with time as the satellite is falling towards the earth. This is about the distance of the satellite during the year 2015
    
    #get the earth_radius at the satellite's position
    #earth_radius = 6371000.8 #geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
    #atmosphere = 12000. #the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
    #r = earth_radius + atmosphere #the full radius of the occulting earth-sphere
    
    #define the opening angles of the overlapping cones (earth and detector). The defined angles are just half of the opening angle, from the central line to the surface of the cone.
    #theta = math.asin(r/sat_dist)
    
    #total_earth_area = 2*math.pi*(1 - np.cos(theta))
    
    #A_an2_big = total_earth_area - np.flipud(A_an2)
    
    
    
    return opening_ang, earth_occ, free_area, A_an2


def write_earth_occ_fits_file(angle):
    """This function writes earth occultation files for all earth-angle given and returns the following information about the files: filepaths, directory\n
    Input:\n
    write_earth_occ_file ( angle )\n
    Output:\n
    0 = filepaths\n
    1 = directory"""
    
    number = len(angle)
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'earth_occultation'
    fits_path = os.path.join(os.path.dirname(__dir__), directory)
    fitsname = 'earth_occ_calc_kt.fits'
    fitsfilepath = os.path.join(fits_path, fitsname)

    prihdu = fits.PrimaryHDU()
    hdulist = [prihdu]
    
    for i in range(0, number):
        
        calculation = calc_earth_occ(angle[i])
        angle_d = calculation[0]*180./math.pi
        area_frac = calculation[1]
        free_area = calculation[2]
        occ_area = calculation[3]
        
        col1 = fits.Column(name = 'angle_d', format = 'E', array = angle_d, unit = 'deg')
        col2 = fits.Column(name = 'area_frac', format = 'E', array = area_frac)
        col3 = fits.Column(name = 'free_area', format = 'E', array = free_area, unit = 'rad')
        col4 = fits.Column(name = 'occ_area', format = 'E', array = occ_area, unit = 'rad')
        cols = fits.ColDefs([col1, col2, col3, col4])
        hdulist.append(fits.TableHDU.from_columns(cols, name = str(angle[i])))
        
    thdulist = fits.HDUList(hdulist)
    thdulist.writeto(fitsfilepath)
    
    return fitsfilepath

angle = np.arange(0, 180.5, .5)
#angle = np.array([0., 1.])
#angle = np.array([170.])

occ_files = write_earth_occ_fits_file(angle)

'''__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'earth_occultation'
fits_path = os.path.join(os.path.dirname(__dir__), directory)

fitsname = 'earth_occ_kt.fits'
fitsfilepath = os.path.join(fits_path, fitsname)

#hdu = fits.PrimaryHDU()
#hdu.writeto(fitsfilepath)

hdulist = fits.open(fitsfilepath)
tbhdu = fits.TableHDU(name = '0.0')
hdulist.append(tbhdu)
print hdulist[1].header
hdulist[1].data = [[1,2],[3,4]]'''



'''filepaths = occ_files[0]
fits_path = occ_files[1]
filenames = occ_files[2]

__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'earth_occultation'
fits_path = os.path.join(os.path.dirname(__dir__), directory)

fitsname = 'earth_occ_kt.fits'

t = Table.read(filepaths[0], format='ascii')

fitsfilepath = os.path.join(fits_path, fitsname)
t.write(fitsfilepath, overwrite=True)'''

#print contents

'''calculation = calc_earth_occ(angle)
angle_d = calculation[0]*180./math.pi
area_frac = calculation[1]
free_area = calculation[2]
A_an2 = calculation[3]
#A_d_an2 = calculation[4]
#A_e_an2 = calculation[5]
#A_hend = calculation[6]
#earth_occ_hend = calculation[7]
#a = calculation[8]

#print area_frac
#print len(angle_d)
#print len(area_frac)
#print len(A_an2)

plt.plot(angle_d, area_frac, 'b-')
plt.plot(angle_d, free_area, 'r--')
#plt.plot(angle_d, A_d_an2, 'y-')
#plt.plot(angle_d, A_e_an2, 'r-')
plt.plot(angle_d, A_an2, 'g-')
#plt.plot(angle_d, A_hend, 'r-')
#plt.plot(angle_d, earth_occ_hend, 'y-')

#plt.ylim(-15, 15)

plt.show()'''

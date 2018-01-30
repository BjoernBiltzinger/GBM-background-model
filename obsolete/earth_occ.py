#!/usr/bin python2.7

import math
import numpy as np
from scipy import interpolate


def _calc_earth_occ(self):
    """This function calculates the overlapping area fraction for a certain earth-angle and stores the data in arrays of the form: opening_ang, earth_occ\n
    Input:\n
    calc_earth_occ ( angle )\n
    Output:\n
    0 = angle of the detector-cone\n
    1 = area fraction of the earth-occulted area to the entire area of the detector-cone"""

    angles = np.arange(0, 180.5, .5)

    earth_occ_dic = {}
    angle_d = []
    area_frac = []
    free_area = []
    occ_area = []

    for angle in angles:

        # get the distance from the satellite to the center of the earth
        sat_dist = 6912000.

        # get the earth_radius at the satellite's position
        earth_radius = 6371000.8
        atmosphere = 12000.
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere

        # define the opening angles of the overlapping cones (earth and detector). The defined angles are just half of the opening angle, from the central line to the surface of the cone.
        theta = math.asin(r / sat_dist)  # earth-cone
        opening_ang = np.arange(math.pi / 36000., math.pi / 2. + math.pi / 36000., math.pi / 36000.)  # detector-cone

        # get the angle between the detector direction and the earth direction
        earth_ang = angle * 2. * math.pi / 360.  # input parameter

        # geometric considerations for the two overlapping spherical cap problem
        phi = math.pi / 2 - earth_ang  # angle between the earth-direction and the axis perpendicular to the detector-orientation on the earth-detector-plane
        f = (np.cos(theta) - np.cos(opening_ang) * np.sin(phi)) / (
            np.cos(phi))  # distance perpendicular to the detector-orientation to the intersection-plane of the spherical caps
        beta = np.arctan2(f, (np.cos(opening_ang)))  # angle of the intersection-plane to the detector orientation

        # same considerations for the earth-component
        f_e = (np.cos(opening_ang) - np.cos(theta) * np.sin(phi)) / (
            np.cos(phi))  # distance perpendicular to the detector-orientation to the intersection-plane of the spherical caps
        beta_e = np.arctan2(f_e, (np.cos(theta)))  # angle of the intersection-plane to the detector orientation

        # calculate one part of the overlapping area of the spherical caps. This area belongs to the detector-cone
        A_d_an2 = 2 * (np.arctan2(
            (np.sqrt(-(np.tan(beta)) ** 2 / ((np.sin(opening_ang)) ** 2) + (np.tan(beta)) ** 2 + 1) * np.sin(opening_ang)),
            np.tan(beta)) - np.cos(opening_ang) * np.arccos(np.tan(beta) / np.tan(opening_ang)) - (np.arctan2(
            (np.sqrt(-(np.tan(beta)) ** 2 / ((np.sin(beta)) ** 2) + (np.tan(beta)) ** 2 + 1) * np.sin(beta)),
            np.tan(beta)) - np.cos(beta) * np.arccos(np.tan(beta) / np.tan(beta))))

        # calculate the other part of the overlapping area. This area belongs to the earth-cone
        A_e_an2 = 2 * (
        np.arctan2((np.sqrt(-(np.tan(beta_e)) ** 2 / ((np.sin(theta)) ** 2) + (np.tan(beta_e)) ** 2 + 1) * np.sin(theta)),
                   np.tan(beta_e)) - np.cos(theta) * np.arccos(np.tan(beta_e) / np.tan(theta)) - (
        np.arctan2((np.sqrt(-(np.tan(beta_e)) ** 2 / ((np.sin(beta_e)) ** 2) + (np.tan(beta_e)) ** 2 + 1) * np.sin(beta_e)),
                   np.tan(beta_e)) - np.cos(beta_e) * np.arccos(np.tan(beta_e) / np.tan(beta_e))))

        # take the limitations of trignometric functions into account. -> Get rid of 2*pi jumps
        A_e_an2[np.where(earth_ang < beta)] = A_e_an2[np.where(earth_ang < beta)] - 2 * math.pi
        A_d_an2[np.where(f < 0)] = A_d_an2[np.where(f < 0)] - 2 * math.pi

        # combine the two area segments to get the total area
        A_an2 = A_d_an2 + A_e_an2

        # calculate the unocculted area of the detector cone
        free_area = 2 * math.pi * (1 - np.cos(opening_ang))

        # add values to the overlapping area, where either the detector-cone is completely embedded within the earth-cone or the other way around. Within this function both could be the case, because we are changing the angle of the detector-cone!
        A_an2[np.where(opening_ang <= theta - earth_ang)] = free_area
        A_an2[np.where(opening_ang >= theta + earth_ang)] = 2 * math.pi * (1 - np.cos(theta))
        A_an2[np.where(opening_ang <= earth_ang - theta)] = 0.

        # if the earth will never be within the detector-cone, the overlapping area will always be 0
        # if earth_ang > opening_ang[-1] + theta:
        #    A_an2 = np.zeros(len(opening_ang))

        # Apparently the numeric calculation of the analytic solution doesn't always return a value (probably because of runtime error). As a result there are several 'nan' entries in the A_an2 array. To get rid of those we interpolate over all the calculated solutions. We have chosen enough steps for the opening_ang to eliminate any errors due to this interpolation, because we get enough good results from the calculation.
        tck = interpolate.splrep(opening_ang[np.logical_not(np.isnan(A_an2))], A_an2[np.logical_not(np.isnan(A_an2))], s=0)
        A_an2 = interpolate.splev(opening_ang, tck, der=0)

        # calculate the fraction of the occulated area
        earth_occ = A_an2 / free_area

        angle_d.append(opening_ang * 180. / math.pi)
        area_frac.append(earth_occ)
        free_area.append(free_area)
        occ_area.append(A_an2)

    earth_occ_dic['angle_d'] = angle_d
    earth_occ_dic['area_frac'] = area_frac
    earth_occ_dic['free_area'] = free_area
    earth_occ_dic['occ_area'] = occ_area

    return earth_occ_dic
#!/usr/bin python2.7

import getpass
import math
import os
from datetime import datetime

import ephem
import matplotlib                          as mpl
import matplotlib.pyplot                   as plt
import numpy                               as np
import scipy.optimize                      as optimization
from astropy.io import fits
from astropy.time import Time
from numpy import linalg        as LA
from scipy import interpolate

from gbmbkgpy.external_prop import ExternalProps, writefile


class detector(object):
    """This class contains the orientations of all detectors of the GBM setup (azimuth and zenith):\n
    n0 -> azimuth, zenith, azimuthg, zenithg\n
    n1 -> azimuth, zenith, azimuthg, zenithg\n
    .\n
    .\n
    .\n
    b1 -> azimuth, zenith, azimuthg, zenithg\n\n\n"""

    class n0:
        azimuth = 45.8899994 * 2 * math.pi / 360.  # radians
        zenith = 20.5799999 * 2 * math.pi / 360.
        azimuthg = 45.88999943  # degrees
        zenithg = 20.5799999

    class n1:
        azimuth = 45.1100006 * 2 * math.pi / 360.
        zenith = 45.3100014 * 2 * math.pi / 360.
        azimuthg = 45.1100006
        zenithg = 45.3100014

    class n2:
        azimuth = 58.4399986 * 2 * math.pi / 360.
        zenith = 90.2099991 * 2 * math.pi / 360.
        azimuthg = 58.4399986
        zenithg = 90.2099991

    class n3:
        azimuth = 314.869995 * 2 * math.pi / 360.
        zenith = 45.2400017 * 2 * math.pi / 360.
        azimuthg = 314.869995
        zenithg = 45.2400017

    class n4:
        azimuth = 303.149994 * 2 * math.pi / 360.
        zenith = 90.2699966 * 2 * math.pi / 360.
        azimuthg = 303.149994
        zenithg = 90.2699966

    class n5:
        azimuth = 3.34999990 * 2 * math.pi / 360.
        zenith = 89.7900009 * 2 * math.pi / 360.
        azimuthg = 3.34999990
        zenithg = 89.7900009

    class n6:
        azimuth = 224.929993 * 2 * math.pi / 360.
        zenith = 20.4300003 * 2 * math.pi / 360.
        azimuthg = 224.929993
        zenithg = 20.4300003

    class n7:
        azimuth = 224.619995 * 2 * math.pi / 360.
        zenith = 46.1800003 * 2 * math.pi / 360.
        azimuthg = 224.619995
        zenithg = 46.1800003

    class n8:
        azimuth = 236.610001 * 2 * math.pi / 360.
        zenith = 89.9700012 * 2 * math.pi / 360.
        azimuthg = 236.610001
        zenithg = 89.9700012

    class n9:
        azimuth = 135.190002 * 2 * math.pi / 360.
        zenith = 45.5499992 * 2 * math.pi / 360.
        azimuthg = 135.190002
        zenithg = 45.5499992

    class na:
        azimuth = 123.730003 * 2 * math.pi / 360.
        zenith = 90.4199982 * 2 * math.pi / 360.
        azimuthg = 123.730003
        zenithg = 90.4199982

    class nb:
        azimuth = 183.740005 * 2 * math.pi / 360.
        zenith = 90.3199997 * 2 * math.pi / 360.
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

class calculate_geometry(object):
    def altitude(self, day):
        """This function calculates the satellite's altitude for one day and stores the data in arrays of the form: altitude, earth_radius, sat_time\n
        Input:\n
        calculate.altitude ( day = JJMMDD )\n
        Output:\n
        0 = altitude of the satellite above the WGS84 ellipsoid\n
        1 = radius of the earth at the position of the satellite\n
        2 = time (MET) in seconds"""

        poshist = rf.poshist(day)
        sat_time = poshist[0]
        sat_pos = poshist[1]
        sat_lat = poshist[2]
        sat_lon = poshist[3]

        ell_a = 6378137
        ell_b = 6356752.3142
        sat_lat_rad = sat_lat * 2 * math.pi / 360.

        earth_radius = ell_a * ell_b / (
            (ell_a ** 2 * np.sin(sat_lat_rad) ** 2 + ell_b ** 2 * np.cos(sat_lat_rad) ** 2) ** 0.5)

        altitude = (LA.norm(sat_pos, axis=0) - earth_radius) / 1000.

        return altitude, earth_radius, sat_time

    def ang_eff(self, ang, echan, data_type='ctime', detector_type='NaI'):
        """This function converts the angle of one detectortype to a certain source into an effective angle considering the angular dependence of the effective area and stores the data in an array of the form: ang_eff\n
        Input:\n
        calculate.ang_eff ( ang (in degrees), echan (integer in the range of 0-7 or 0-127), datatype='ctime' (or 'cspec'), detectortype='NaI' (or 'BGO') )\n
        Output:\n
        0 = effective angle\n
        1 = normalized photopeak effective area curve"""

        fitsname = 'peak_eff_area_angle_calib_GBM_all_DRM.fits'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/calibration/'
        fitsfilepath = os.path.join(fits_path, fitsname)
        fitsfile = fits.open(fitsfilepath, mode='update')
        data = fitsfile[1].data
        fitsfile.close()
        x = data.field(0)
        y0 = data.field(1)  # for NaI (4-12 keV) gewichteter Mittelwert = 10.76 keV
        y1 = data.field(2)  # for NaI (12-27 keV) gewichteter Mittelwert = 20.42 keV
        y2 = data.field(3)  # for NaI (27-50 keV) gewichteter Mittelwert = 38.80 keV
        y3 = data.field(4)  # for NaI (50-102 keV) gewichteter Mittelwert = 76.37 keV
        y4 = data.field(5)  # for NaI (102-295 keV) gewichteter Mittelwert = 190.19 keV
        y5 = data.field(6)  # for NaI (295-540 keV) gewichteter Mittelwert = 410.91 keV
        y6 = data.field(7)  # for NaI (540-985 keV) gewichteter Mittelwert = 751.94 keV
        y7 = data.field(8)  # for NaI (985-2000 keV) gewichteter Mittelwert = 1466.43 keV
        # y4 = data.field(4)#for BGO (898 keV)
        # y5 = data.field(5)#for BGO (1836 keV)

        y_all = np.array([y0, y1, y2, y3, y4, y5, y6, y7])
        ang_eff = []

        if detector_type == 'NaI':
            if data_type == 'ctime':
                # ctime linear-interpolation factors
                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 127./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 109./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 77./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 159./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # resulting effective area curve
                y = y_all[echan]

                # normalize
                # y = y/y1[90]

                # calculate the angle factors
                tck = interpolate.splrep(x, y)
                ang_eff = interpolate.splev(ang, tck, der=0)

                # convert the angle according to their factors
                # ang_eff = np.array(ang_fac*ang)

            else:
                print 'data_type cspec not yet implemented'

        else:
            print 'detector_type BGO not yet implemented'

        '''ang_rad = ang*(2.*math.pi)/360.
        ang_eff = 110*np.cos(ang_rad)
        ang_eff[np.where(ang > 90.)] = 0.'''
        return ang_eff, y

    def ang_to_earth(self, day, ra, dec):
        """This function calculates the angle between a source and the earth and stores the data in arrays of the form: ang_occ, src_pos, src_rad\n
        Input:\n
        calculate.ang_to_earth( day = YYMMDD, ra, dec)\n
        Output:\n
        0 = angle between the source and the earth\n
        1 = position of the source in J2000 coordinates\n
        2 = position of the source in right ascension and declination"""
        # get satellite data
        data = ExternalProps.poshist(ExternalProps(), day)
        sat_time = data[0]
        sat_pos = data[1]
        sat_lat = data[2]
        sat_lon = data[3]
        sat_q = data[4]

        # calculate the earth location unit-vector
        sat_dist = LA.norm(sat_pos, axis=0)  # get the altitude of the satellite (length of the position vector)
        sat_pos_unit = sat_pos / sat_dist  # convert the position vector into a unit-vector
        geo_pos_unit = -sat_pos_unit

        # convert the given angles into a unit-vector
        src_rad = np.array([ra, dec], float)
        src_pos = np.array([np.cos(ra * (2. * math.pi) / 360.) * np.cos(dec * (2. * math.pi) / 360.),
                            np.sin(ra * (2. * math.pi) / 360.) * np.cos(dec * (2. * math.pi) / 360.),
                            np.sin(dec * (2. * math.pi) / 360.)], float)

        # calculate the angle between the earth location and the detector orientation
        scalar_product = src_pos[0] * geo_pos_unit[0] + src_pos[1] * geo_pos_unit[1] + src_pos[2] * geo_pos_unit[2]
        ang_src_geo = np.arccos(scalar_product)
        ang_occ = ang_src_geo * 360. / (2. * math.pi)
        ang_occ = np.array(ang_occ)
        return ang_occ, src_pos, src_rad

    def ang_to_earth_bin(self, day, ra, dec, bin_time_mid=0, detector_name=0, data_type='ctime'):
        """This function calculates the angle between a source and the earth and stores the data in arrays of the form: ang_occ, src_pos, src_rad\n
        Input:\n
        calculate.ang_to_earth( day = YYMMDD, ra, dec)\n
        Output:\n
        0 = angle between the source and the earth\n
        1 = position of the source in J2000 coordinates\n
        2 = position of the source in right ascension and declination"""
        # get satellite data
        data = ExternalProps.poshist_bin(ExternalProps(), day, bin_time_mid, detector_name, data_type)
        sat_time = data[0]
        sat_pos = data[1]
        sat_lat = data[2]
        sat_lon = data[3]
        sat_q = data[4]
        bin_time_mid = data[5]

        # calculate the earth location unit-vector
        sat_dist = LA.norm(sat_pos, axis=0)  # get the altitude of the satellite (length of the position vector)
        sat_pos_unit = sat_pos / sat_dist  # convert the position vector into a unit-vector
        geo_pos_unit = -sat_pos_unit

        # convert the given angles into a unit-vector
        src_rad = np.array([ra, dec], float)
        src_pos = np.array([np.cos(ra * (2. * math.pi) / 360.) * np.cos(dec * (2. * math.pi) / 360.),
                            np.sin(ra * (2. * math.pi) / 360.) * np.cos(dec * (2. * math.pi) / 360.),
                            np.sin(dec * (2. * math.pi) / 360.)], float)

        # calculate the angle between the earth location and the detector orientation
        scalar_product = src_pos[0] * geo_pos_unit[0] + src_pos[1] * geo_pos_unit[1] + src_pos[2] * geo_pos_unit[2]
        ang_src_geo = np.arccos(scalar_product)
        ang_occ = ang_src_geo * 360. / (2. * math.pi)
        ang_occ = np.array(ang_occ)
        return ang_occ, src_pos, src_rad

    def burst_ang(self, detector_name, day, burst_ra, burst_dec):
        """This function calculates the burst orientation for one detector and stores the data in arrays of the form: burst_ang, sat_time\n
        Input:\n
        calculate.burst_ang ( detector, day = JJMMDD, burst_ra, burst_dec )\n
        Output:\n
        0 = angle between the burst and the detector\n
        1 = time (MET) in seconds"""

        # get the detector and satellite data
        data_det = calculate.det_or(calculate(), detector_name, day)
        det_coor = data_det[0]  # unit-vector of the detector orientation
        det_rad = data_det[1]  # detector orientation in right ascension and declination
        sat_pos = data_det[2]  # position of the satellite
        sat_time = np.array(data_det[3])  # time (MET) in seconds

        # convert the burst angles into a unit-vector
        burst_rad = np.array([burst_ra, burst_dec], float)
        burst_pos = np.array([np.cos(burst_ra * (2. * math.pi) / 360.) * np.cos(burst_dec * (2. * math.pi) / 360.),
                              np.sin(burst_ra * (2. * math.pi) / 360.) * np.cos(burst_dec * (2. * math.pi) / 360.),
                              np.sin(burst_dec * (2. * math.pi) / 360.)],
                             float)  # unit-vector pointing to the burst location

        # calculate the angle between the burst location and the detector orientation
        scalar_product = det_coor[0] * burst_pos[0] + det_coor[1] * burst_pos[1] + det_coor[2] * burst_pos[2]
        ang_det_burst = np.arccos(scalar_product)
        burst_ang = (ang_det_burst) * 360. / (2. * math.pi)  # convert to degrees
        burst_ang = np.array(burst_ang)
        return burst_ang, sat_time

    def burst_ang_bin(self, detector_name, day, burst_ra, burst_dec, bin_time_mid=0, data_type='ctime'):
        """This function calculates the binned burst orientation for one detector and stores the data in arrays of the form: burst_ang_bin, sat_time_bin, bin_time_mid\n
        Input:\n
        calculate.burst_ang_bin ( detector, day = JJMMDD, burst_ra, burst_dec, bin_time_mid = 0, detector = 0, data_type = 'ctime' )\n
        Output:\n
        0 = angle between the burst and the detector\n
        1 = time (MET) in seconds\n
        2 = bin_time_mid"""

        # get the detector and satellite data
        data_det = calculate.det_or_bin(calculate(), detector_name, day, bin_time_mid, data_type)
        det_coor_bin = data_det[0]  # unit-vector of the detector orientation
        det_rad_bin = data_det[1]  # detector orientation in right ascension and declination
        sat_pos_bin = data_det[2]  # position of the satellite
        sat_time_bin = np.array(data_det[3])  # time (MET) in seconds
        bin_time_mid = data_det[4]

        # convert the burst angles into a unit-vector
        burst_rad = np.array([burst_ra, burst_dec], float)
        burst_pos = np.array(
            [np.cos(burst_ra) * np.cos(burst_dec), np.sin(burst_ra) * np.cos(burst_dec), np.sin(burst_dec)],
            float)  # unit-vector pointing to the burst location

        # calculate the angle between the burst location and the detector orientation
        scalar_product = det_coor_bin[0] * burst_pos[0] + det_coor_bin[1] * burst_pos[1] + det_coor_bin[2] * burst_pos[
            2]
        ang_det_burst = np.arccos(scalar_product)
        burst_ang_bin = (ang_det_burst) * 360. / (2. * math.pi)  # convert to degrees
        burst_ang_bin = np.array(burst_ang_bin)
        return burst_ang_bin, sat_time_bin, bin_time_mid


class calculate(object):
    """This class contains all calculation functions needed for the GBM background model:\n
    altitude(self, day) -> altitude, earth_radius, sat_time\n
    ang_eff(self, ang, detector) -> ang_eff\n
    burst_ang(self, detector, day, burst_ra, burst_dec) -> burst_ang, sat_time\n
    burst_ang_bin(self, detector, day, burst_ra, burst_dec, bin_time_mid = 0, data_type = 'ctime') -> burst_ang_bin, sat_time_bin, bin_time_mid\n
    earth_occ_eff(self, earth_ang, echan, datatype = 'ctime', detectortype = 'NaI') -> eff_area_frac\n
    date_to_met(self, year, month = 01, day = 01, hour = 00, minute = 00, seconds = 00.) -> met, mjdutc\n
    day_to_met(self, day) -> met, mjdutc\n
    det_or(self, detector, day) -> det_coor, det_rad, sat_pos, sat_time\n
    det_or_bin(self, detector, day, bin_time_mid = 0, data_type = 'ctime') -> det_coor_bin, det_rad_bin, sat_pos_bin, sat_time_bin, bin_time_mid\n
    earth_ang(self, detector, day) -> earth_ang, sat_time\n
    earth_ang_bin(self, detector, day, bin_time_mid = 0, data_type = 'ctime') -> earth_ang_bin, sat_time_bin, bin_time_mid\n
    geo_to_sat(self, sat_q, geo_coor) -> sat_coor, sat_rad\n
    intpol(self, vector, day, direction = 0, sat_time = 0, bin_time_mid = 0, detector = 0, data_type = 'ctime') -> new_vector, sat_time, bin_time_mid\n
    met_to_date(self, met) -> mjdutc, mjdtt, isot, date, burst\n
    sat_to_geo(self, sat_q, sat_coor) -> geo_coor, geo_rad\n
    sun_ang(self, detector, day) -> sun_ang, sat_time\n
    sun_ang_bin(self, detector, day, bin_time_mid = 0, data_type = 'ctime') -> sun_ang_bin, sat_time_bin, bin_time_mid\n
    sun_pos(self, day) -> sun_pos, sun_rad\n
    sun_pos_bin(self, day, bin_time_mid = 0, detector = 0, data_type = 'ctime') -> sun_pos_bin, sun_rad_bin, bin_time_mid\n
    rigidity(self, day, bin_time_mid = 0) -> rigidity, sat_lon, sat_lat, sat_time\n\n\n"""



    def curve_fit_plots(self, day, detector_name, echan, data_type='ctime', plot='yes', write='no', burst_time='none'):
        """This function calculates the chi-square fit to the data of a given detector, day and energy channel and saves the figure in the appropriate folder\n
        Input:\n
        calculate.curve_fit_plots( day = YYMMDD, detector (f. e. 'n5'), echan (9 or 129 = total counts for ctime or cspec), data_type = 'ctime' (or 'cspec'), plot = 'yes' (input 'yes!' for creating new plots), write = 'no' (input 'yes' for creating new data_files))\n
        Output:\n
        0 = residuals\n
        1 = counts\n
        2 = fit_curve\n
        3 = cgb_fit\n
        4 = magnetic_fit\n
        5 = earth_ang_bin_fit\n
        6 = sun_ang_bin_fit\n
        7 = crab_ang_bin_fit\n
        8 = scox_ang_bin\n
        9 = cygx_ang_bin\n
        10 = plot_time_bin\n
        11 = plot_time_sat"""

        sources_data = ExternalProps.point_sources(ExternalProps())
        sources_names = sources_data[0]
        sources_coordinates = sources_data[1]
        sources_number = int(len(sources_names))

        ##### Pointsources testing #####
        # if data_type == 'ctime':
        #    if echan < 8:
        #        fitsname_data = '_test_s' + str(sources_number) + '_ctime_' + detector_name + '_e' + str(echan) + '_kt.fits'
        #    elif echan == 8:
        #        fitsname_data = '_test_s' + str(sources_number) + '_ctime_' + detector_name + '_tot_kt.fits'
        #    else:
        #        print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
        #        return
        # elif data_type == 'cspec':
        #    if echan < 128:
        #        fitsname_data = '_test_s' + str(sources_number) + '_cspec_' + detector_name + '_e' + str(echan) + '__kt.fits'
        #    elif echan == 128:
        #        fitsname_data = '_test_s' + str(sources_number) + '_cspec_' + detector_name + '_tot_kt.fits'
        #    else:
        #        print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
        #        return
        # else:
        #    print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
        #    return

        # See if Fit has already been calculated before
        if data_type == 'ctime':
            if echan < 8:
                fitsname_data = 'ctime_' + detector_name + '_e' + str(echan) + '_kt.fits'
            elif echan == 8:
                fitsname_data = 'ctime_' + detector_name + '_tot_kt.fits'
            else:
                print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
                return
        elif data_type == 'cspec':
            if echan < 128:
                fitsname_data = 'cspec_' + detector_name + '_e' + str(echan) + '__kt.fits'
            elif echan == 128:
                fitsname_data = 'cspec_' + detector_name + '_tot_kt.fits'
            else:
                print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
                return
        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
            return

        user = getpass.getuser()
        fits_path_data = '/home/' + user + '/Work/Fits_data/' + str(day) + '/'
        if not os.access(fits_path_data, os.F_OK):
            os.mkdir(fits_path_data)
        fitsfilepath_data = os.path.join(fits_path_data, fitsname_data)

        if write == 'no' and os.path.isfile(fitsfilepath_data) == True and plot != 'yes!':
            data = ExternalProps.fits_data(ExternalProps(), day, detector_name, echan, data_type)
            residuals = np.array(data[0].astype(np.float))
            counts = np.array(data[1].astype(np.float))
            fit_curve = np.array(data[2].astype(np.float))
            cgb = np.array(data[3].astype(np.float))
            magnetic = np.array(data[4].astype(np.float))
            earth_ang_bin = np.array(data[5].astype(np.float))
            sun_ang_bin = np.array(data[6].astype(np.float))
            sources_fit_curve = np.array(data[7].astype(np.float))
            plot_time_bin = np.array(data[8].astype(np.float))
            plot_time_sat = np.array(data[9].astype(np.float))
            fit_coeffs = np.array(data[10].astype(np.float))
            a = fit_coeffs[0]
            b = fit_coeffs[1]
            c = fit_coeffs[2]
            d = fit_coeffs[3]
            sources_ang_bin = data[11].astype(np.float)
            sources_names = data[12]
            sources_coeff = data[13].astype(np.float)

            return residuals, counts, fit_curve, a * cgb, b * magnetic, c * earth_ang_bin, d * sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat

        det = getattr(detector(), detector_name)
        year = int('20' + str(day)[0:2])

        # get the iso-date-format from the day
        date = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))

        # get the ordinal indicator for the date
        ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

        # read the measurement data
        if data_type == 'ctime':
            ctime_data = ExternalProps.ctime(ExternalProps(), detector_name, day)
            chan_energies = ctime_data[0]
            total_counts = ctime_data[1]
            echan_counts = ctime_data[2]
            total_rate = ctime_data[3]
            echan_rate = ctime_data[4]
            bin_time = ctime_data[5]
            good_time = ctime_data[6]
            exptime = ctime_data[7]
            bin_time_mid = np.array((bin_time[:, 0] + bin_time[:, 1]) / 2)

            total_rate = np.sum(echan_rate[1:-1], axis=0)

        elif data_type == 'cspec':
            cspec_data = ExternalProps.cspec(ExternalProps(), detector_name, day)
            chan_energies = cspec_data[0]
            total_counts = cspec_data[1]
            echan_counts = cspec_data[2]
            total_rate = cspec_data[3]
            echan_rate = cspec_data[4]
            bin_time = cspec_data[5]
            good_time = cspec_data[6]
            exptime = cspec_data[7]
            bin_time_mid = np.array((bin_time[:, 0] + bin_time[:, 1]) / 2)

            total_rate = np.sum(echan_rate[1:-1], axis=0)

        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'

        if data_type == 'ctime':
            if echan < 8:
                counts = echan_rate[echan]
            elif echan == 8:
                counts = total_rate
                echan = 0
            else:
                print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
                return
        elif data_type == 'cspec':
            if echan < 128:
                counts = echan_rate[echan]
            elif echan == 128:
                counts = total_rate
                echan = 0
            else:
                print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
                return
        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
            return

        # read the satellite data
        sat_data = ExternalProps.poshist_bin(ExternalProps(), day, bin_time_mid, detector_name, data_type)
        sat_time_bin = sat_data[0]
        sat_pos_bin = sat_data[1]
        sat_lat_bin = sat_data[2]
        sat_lon_bin = sat_data[3]
        sat_q_bin = sat_data[4]

        # calculate the sun data
        sun_data = calculate.sun_ang_bin(calculate(), detector_name, day, bin_time_mid, data_type)
        sun_ang_bin = sun_data[0]
        sun_ang_bin = calculate.ang_eff(calculate(), sun_ang_bin, echan)[0]
        sun_rad = sun_data[2]
        sun_ra = sun_rad[:, 0]
        sun_dec = sun_rad[:, 1]
        sun_occ = calculate.src_occultation_bin(calculate(), day, sun_ra, sun_dec, bin_time_mid)[0]

        # calculate the earth data
        earth_data = calculate.earth_ang_bin(calculate(), detector_name, day, bin_time_mid, data_type)
        earth_ang_bin = earth_data[0]
        # earth_ang_bin = calc.ang_eff(earth_ang_bin, echan)[0]
        earth_ang_bin = calculate.earth_occ_eff(calculate(), earth_ang_bin, echan)

        # calculate the pointsources data
        sources_ang_bin = np.array([np.zeros(len(sun_ang_bin))])

        for i in range(0, sources_number):
            src_ra = sources_coordinates[0, i]
            src_dec = sources_coordinates[1, i]

            src_data = calculate.burst_ang_bin(calculate(), detector_name, day, src_ra, src_dec, bin_time_mid,
                                               data_type)
            src_ang_bin = src_data[0]
            src_ang_bin = calculate.ang_eff(calculate(), src_ang_bin, echan, data_type)[0]
            src_occ = \
            calculate.src_occultation_bin(calculate(), day, src_ra, src_dec, bin_time_mid, detector_name, data_type)[0]

            src_ang_bin[np.where(src_occ == 0)] = 0
            src_ang_bin[np.where(total_rate == 0)] = 0

            # remove vertical movement from scaling the src_ang_bin functions
            src_ang_bin[src_ang_bin > 0] = src_ang_bin[src_ang_bin > 0] - np.min(src_ang_bin[src_ang_bin > 0])

            src_ang_bin = np.array(src_ang_bin)

            sources_ang_bin = np.concatenate((sources_ang_bin, [src_ang_bin]), axis=0)

        sources_ang_bin = np.delete(sources_ang_bin, 0, 0)

        # read the SFL data
        flares = ExternalProps.flares(ExternalProps(), year)
        flares_day = flares[0]
        flares_time = flares[1]
        flares_today = flares_time[:, np.where(flares_day == int(day))]
        flares_today = np.squeeze(flares_today, axis=(1,)) / 3600.

        # read the mcilwain parameter data
        sat_time = ExternalProps.poshist(ExternalProps(), day)[0]
        lat_data = ExternalProps.mcilwain(ExternalProps(), day)
        mc_b = lat_data[1]
        mc_l = lat_data[2]

        mc_b = calculate.intpol(calculate(), mc_b, day, 0, sat_time, bin_time_mid)[0]
        mc_l = calculate.intpol(calculate(), mc_l, day, 0, sat_time, bin_time_mid)[0]

        magnetic = mc_l
        magnetic = magnetic - np.mean(magnetic, dtype=np.float64)

        # constant function corresponding to the diffuse y-ray background
        cgb = np.ones(len(total_rate))

        # counts[120000:] = 0
        cgb[np.where(total_rate == 0)] = 0
        earth_ang_bin[np.where(total_rate == 0)] = 0
        sun_ang_bin[np.where(sun_occ == 0)] = 0
        sun_ang_bin[np.where(total_rate == 0)] = 0
        magnetic[np.where(total_rate == 0)] = 0

        # remove vertical movement from all functions but the CGB
        sun_ang_bin[sun_ang_bin > 0] = sun_ang_bin[sun_ang_bin > 0] - np.min(sun_ang_bin[sun_ang_bin > 0])
        earth_ang_bin[earth_ang_bin != 0] = earth_ang_bin[earth_ang_bin != 0] - np.mean(
            earth_ang_bin[earth_ang_bin != 0], dtype=np.float64)

        saa_exits = [0]
        for i in range(1, len(total_rate)):
            if np.logical_and(total_rate[i - 1] == 0, total_rate[i] != 0):
                # print i
                saa_exits.append(int(i))
        saa_exits = np.array(saa_exits)

        if saa_exits[1] - saa_exits[0] < 10:
            saa_exits = np.delete(saa_exits, 0)

        def exp_func(x, a, b, i, addition):
            #    if saa_exits[i] == 0:
            #        addition = 0
            #    elif saa_exits[i] < 200:
            #        addition = 10
            #    else:
            #        addition = 9
            #    for i in range(0, 30):
            #        if i > addition:
            #            addition = i
            #            break
            #    print addition
            #    addition = round(addition)
            x_func = x[int(saa_exits[i] + math.fabs(addition)):] - x[int(saa_exits[i] + math.fabs(addition))]
            func = math.fabs(a) * np.exp(-math.fabs(b) * x_func)
            zeros = np.zeros(len(x))
            zeros[int(saa_exits[i] + math.fabs(addition)):] = func
            zeros[np.where(total_rate == 0)] = 0
            return zeros

        deaths = len(saa_exits)

        exp = []
        for i in range(0, deaths):
            exp = np.append(exp, [40., 0.00001])

        sources_x0 = -np.ones(sources_number)

        x0_args = np.append(sources_x0, exp)

        def fit_function(x, a, b, c, d, addition, sources_coeff, exp1, exp2, deaths, sources_number):
            this_is_it = a * cgb + b * magnetic + c * earth_ang_bin + d * sun_ang_bin
            for i in range(0, sources_number):
                this_is_it = np.add(this_is_it, math.fabs(sources_coeff[i]) * sources_ang_bin[i])
            for j in range(0, deaths):
                this_is_it = np.add(this_is_it, exp_func(x, exp1[j], exp2[j], j, addition))
            return this_is_it

        def wrapper_fit_function(x, deaths, sources_number, addition, a, b, c, d, *args):
            sources_coeff = args[0:sources_number]
            exp1 = args[sources_number::2]
            exp2 = args[sources_number + 1::2]
            return fit_function(x, a, b, c, d, addition, sources_coeff, np.fabs(exp1), np.fabs(exp2), deaths,
                                sources_number)

        user = getpass.getuser()
        addition_path = '/home/' + user + '/Work/Fits/SAA_additions/' + str(day) + '/'
        if counts.all == total_rate.all:
            addition_name = det.__name__ + '_add_tot.dat'
        else:
            addition_name = det.__name__ + '_add_e' + str(echan) + '.dat'
        addition_file = os.path.join(addition_path, addition_name)

        if os.access(addition_file, os.F_OK):
            infos = open(addition_file, 'r')
            addition = int(infos.read())
            infos.close()

        else:
            if not os.access(addition_path, os.F_OK):
                print("Making New Directory")
                os.mkdir(addition_path)

            good_fit = []

            for addition in range(0, 21):
                # print addition
                x0 = np.append(np.array([1300., 20., -12., -1.]), x0_args)  ### add new sources ###
                sigma = np.array((counts + 1.) ** (0.5))

                try:
                    fit_results = optimization.curve_fit(
                        lambda x, a, b, c, d, *args: wrapper_fit_function(x, deaths, sources_number, addition, a, b, c,
                                                                          d, *args), bin_time_mid, counts, x0, sigma,
                        maxfev=10000)

                except RuntimeError:
                    print("Error - curve_fit failed for the value " + str(addition) + ".")
                    good_fit.append(float('Inf'))
                    continue

                coeff = fit_results[0]

                a = coeff[0]
                b = coeff[1]
                c = coeff[2]
                d = coeff[3]
                # addition = coeff[4]
                sources_coeff = np.fabs(coeff[4:sources_number + 4])
                exp1 = coeff[sources_number + 4::2]
                exp2 = coeff[sources_number + 5::2]

                fit_curve = fit_function(bin_time_mid, a, b, c, d, addition, sources_coeff, exp1, exp2, deaths,
                                         sources_number)

                residual_curve = counts - fit_curve

                chi_squared = np.sum(residual_curve ** 2)

                good_fit.append(chi_squared)

            good_fit = np.array(good_fit)

            # print np.argmin(good_fit)

            addition = np.argmin(good_fit)

            infos = open(addition_file, 'w')
            infos.write(str(addition))
            infos.close()

        x0 = np.append(np.array([1300., 20., -12., -1.]), x0_args)
        sigma = np.array((counts + 1) ** (0.5))

        fit_results = optimization.curve_fit(
            lambda x, a, b, c, d, *args: wrapper_fit_function(x, deaths, sources_number, addition, a, b, c, d, *args),
            bin_time_mid, counts, x0, sigma, maxfev=15000)
        coeff = fit_results[0]
        # pcov = fit_results[1]






        a = coeff[0]
        b = coeff[1]
        c = coeff[2]
        d = coeff[3]
        # addition = coeff[4]
        sources_coeff = np.fabs(coeff[4:sources_number + 4])
        exp1 = coeff[sources_number + 4::2]
        exp2 = coeff[sources_number + 5::2]
        saa_coeffs = np.array([exp1, exp2])

        fit_curve = fit_function(bin_time_mid, a, b, c, d, addition, sources_coeff, exp1, exp2, deaths, sources_number)

        sources_fit_curve = np.squeeze(np.sum(np.multiply(sources_coeff, sources_ang_bin.T).T, axis=0))

        #####plot-algorhythm#####
        # convert the x-axis into hours of the day
        plot_time_bin_date = calculate.met_to_date(calculate(), bin_time_mid)[0]
        plot_time_bin = (plot_time_bin_date - calculate.day_to_met(calculate(), day)[1]) * 24  # Time of day in hours
        plot_time_sat_date = calculate.met_to_date(calculate(), sat_time_bin)[0]
        plot_time_sat = (plot_time_sat_date - calculate.day_to_met(calculate(), day)[1]) * 24  # Time of day in hours

        mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

        ###plot each on the same axis as converted to counts###
        if plot == 'yes' or plot == 'yes!':
            fig, ax1 = plt.subplots()

            if burst_time != 'none':
                if len(burst_time) > 2:
                    for i in range(0, len(burst_time)):
                        if i == 0:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        elif i < 4:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                        else:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue', linestyle='-.')
                elif len(burst_time) == 2:
                    plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                    plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                else:
                    plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

            # plot vertical lines for the solar flares of the day
            if flares_today.size != 0:
                if len(flares_today[0]) > 1:
                    for i in range(0, len(flares_today[0])):
                        plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                else:
                    plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

            plot1 = ax1.step(plot_time_bin, counts, 'b-', label='Count rate')
            plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label='Fitted Background Curve')
            plot3 = ax1.plot(plot_time_sat, d * sun_ang_bin, 'y-', label='Sun effective angle')
            plot4 = ax1.plot(plot_time_sat, c * earth_ang_bin, 'c-', label='Earth occultation')
            plot5 = ax1.plot(plot_time_sat, b * magnetic, 'g-', label='McIlwain L-parameter')
            plot6 = ax1.plot(plot_time_sat, a * cgb, 'b--', label='CGB')
            plot7 = ax1.plot(plot_time_sat, sources_fit_curve, 'k-', label='Point-like sources')

            plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6 + plot7
            labels = [l.get_label() for l in plots]
            ax1.legend(plots, labels, loc=4, fontsize=20)

            ax1.grid()

            ax1.set_xlabel('Time of day in 24h', fontsize=40)
            ax1.set_ylabel('Count rate', fontsize=40)
            ax1.tick_params(axis='both', which='major', labelsize=30)

            # ax1.set_xlim([9.84, 9.85])
            ax1.set_xlim([-0.5, 24.5])
            if counts.all == total_rate.all:
                ax1.set_ylim([-500, 1500])
            else:
                ax1.set_ylim([-500, 750])
                # ax1.set_ylim([-200, 600])

            plt.title(str(data_type) + '-fit of the ' + detector_name + '-detector on the ' + ordinal(
                int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year), fontsize=45)

            figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
            if not os.access(figure_path, os.F_OK):
                os.mkdir(figure_path)
            if counts.all == total_rate.all:
                figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_tot.png'
            else:
                figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_e' + str(echan) + '.png'

            fig = plt.gcf()  # get current figure
            fig.set_size_inches(20, 12)

            figure = os.path.join(figure_path, figure_name)
            plt.savefig(figure, bbox_inches='tight', dpi=80)

            # plt.show()

            fig.clf()
            plt.clf()

            ###plot residual noise of the fitting algorithm###

            mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

            residuals = counts - fit_curve

            if burst_time != 'none':
                if len(burst_time) > 2:
                    for i in range(0, len(burst_time)):
                        if i == 0:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        elif i < 4:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                        else:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue', linestyle='-.')
                elif len(burst_time) == 2:
                    plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                    plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                else:
                    plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

            # plot vertical lines for the solar flares of the day
            if np.all(flares_today != -5):
                if len(flares_today[0]) > 1:
                    for i in range(0, len(flares_today[0])):
                        plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                else:
                    plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

            plt.step(plot_time_bin, residuals, 'b-')

            plt.xlabel('Time of day in 24h', fontsize=40)
            plt.ylabel('Residual noise', fontsize=40)
            plt.tick_params(axis='both', which='major', labelsize=30)

            plt.grid()

            plt.title(str(data_type) + '-fit residuals of the ' + detector_name + '-detector on the ' + ordinal(
                int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year), fontsize=45)

            plt.xlim([-0.5, 24.5])
            if counts.all == total_rate.all:
                plt.ylim([-600, 600])
            else:
                plt.ylim([-300, 300])

            figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
            if not os.access(figure_path, os.F_OK):
                os.mkdir(figure_path)
            if counts.all == total_rate.all:
                figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_tot_res.png'
            else:
                figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_e' + str(echan) + '_res.png'

            fig = plt.gcf()  # get current figure
            fig.set_size_inches(20, 12)

            figure = os.path.join(figure_path, figure_name)
            plt.savefig(figure, bbox_inches='tight', dpi=80)

            # plt.show()

            fig.clf()
            plt.clf()

        # print sources_names

        # write data to file if it not already exists.
        if os.path.isfile(fitsfilepath_data) == False:
            writefile.fits_data(writefile(), day, detector_name, echan, data_type, residuals, counts, fit_curve, cgb,
                                magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat,
                                a, b, c, d, sources_ang_bin, sources_names, sources_coeff, saa_coeffs)
        elif write == 'yes':
            writefile.fits_data(writefile(), day, detector_name, echan, data_type, residuals, counts, fit_curve, cgb,
                                magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat,
                                a, b, c, d, sources_ang_bin, sources_names, sources_coeff, saa_coeffs)

        return residuals, counts, fit_curve, a * cgb, b * magnetic, c * earth_ang_bin, d * sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat

    def curve_fit_analysis(self, day, detector_names, echans, data_type='ctime', analysis='energychannels',
                           echanstart='None', echanstop='None', show='no', timeframe='no', burst_time='none',
                           binning=1):
        """This function calculates the chi-square fit to the data of given detectors, days and energy channels and analyses it according to the Input parameters.\n
        Input:\n
        calculate.curve_fit_analysis( day = YYMMDD, detector_names (f. e. ['n5', 'n3']), echans (f. e. [1, 5, 7]), data_type = 'ctime' (or 'cspec'), analysis = 'energychannels' (or 'detectors' or 'both'), echanstart = 'None', echanstop = 'None', show = 'no', timeframe = 'no' (enter 24h array, e.g. [4.5, 6.73]))\n
        Output:\n
        None"""

        if type(day) == np.ndarray and day.size == 1:
            day = day[0]
        elif type(day) != int:
            print 'Invalid Input: Enter a single day in the form YYMMDD.'
            return

        if detector_names.size == 1:
            array_length = len(ExternalProps.poshist_bin(ExternalProps(), day, 0, detector_names[0], data_type)[0])
        else:
            array_length = len(ExternalProps.poshist_bin(ExternalProps(), day, 0, detector_names[0], data_type)[0])

        # get the ordinal indicator for the date
        ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

        # read the SFL data
        year = int('20' + str(day)[0:2])
        flares = ExternalProps.flares(ExternalProps(), year)
        flares_day = flares[0]
        flares_time = flares[1]
        flares_today = flares_time[:, np.where(flares_day == int(day))]
        flares_today = np.squeeze(flares_today, axis=(1,)) / 3600.

        sources_data = ExternalProps.point_sources(ExternalProps())
        sources_names = sources_data[0]
        sources_coordinates = sources_data[1]
        sources_number = len(sources_names)

        # get the iso-date-format from the day
        date = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))

        user = getpass.getuser()

        ##################################################################################################################
        #############################################energychannels#######################################################
        ##################################################################################################################

        if analysis == 'energychannels':
            # for each energy channel calculate all fits for the detectors and sum up the results. Save them seperately for each energy channel.
            if echanstart == 'None' and echanstop == 'None':
                data_channels = []
                if echans.size == 1:
                    if detector_names.size == 1:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echans[0], data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for detector_name in detector_names:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echans[0], data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_channels = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    for echan in echans:
                        if detector_names.size == 1:
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for detector_name in detector_names:
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_channels.append(temp_data)

                    data_channels = np.array(data_channels)




            elif echanstart != 'None' and echanstop != 'None':
                data_channels = []
                if echanstart == echanstop:
                    if detector_names.size == 1:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echanstart, data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for detector_name in detector_names:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echanstart, data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_channels = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    for echan in range(echanstart, echanstop + 1):
                        if detector_names.size == 1:
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for detector_name in detector_names:
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_channels.append(temp_data)

                    data_channels = np.array(data_channels)

            else:
                print 'Unknown energy channel Input. Please insert certain energy channels or choose a starting and an ending energy channel.'
                return

            combined_data = data_channels
            #####plot-algorithm#####

            mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

            if echanstart == 'None' and echanstop == 'None':
                length = echans.size
            else:
                length = echanstop + 1 - echanstart

            for j in range(0, length):
                if length == 1:
                    residuals = combined_data[0]
                    counts = combined_data[1]
                    fit_curve = combined_data[2]
                    cgb = combined_data[3]
                    magnetic = combined_data[4]
                    earth_ang_bin = combined_data[5]
                    sun_ang_bin = combined_data[6]
                    sources_fit_curve = combined_data[7]
                    plot_time_bin = combined_data[8]  # Time of day in hours
                    plot_time_sat = combined_data[9]  # Time of day in hours
                else:
                    residuals = combined_data[j][0]
                    counts = combined_data[j][1]
                    fit_curve = combined_data[j][2]
                    cgb = combined_data[j][3]
                    magnetic = combined_data[j][4]
                    earth_ang_bin = combined_data[j][5]
                    sun_ang_bin = combined_data[j][6]
                    sources_fit_curve = combined_data[j][7]
                    plot_time_bin = combined_data[j][8]  # Time of day in hours
                    plot_time_sat = combined_data[j][9]  # Time of day in hours

                ###plot each on the same axis as converted to counts###
                fig, ax1 = plt.subplots()

                if burst_time != 'none':
                    if len(burst_time) > 2:
                        for i in range(0, len(burst_time)):
                            if i == 0:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black',
                                            linestyle='--')
                            elif i < 4:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                            else:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue',
                                            linestyle='-.')
                    elif len(burst_time) == 2:
                        plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                    else:
                        plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

                # plot vertical lines for the solar flares of the day
                if flares_today.size != 0:
                    if len(flares_today[0]) > 1:
                        for i in range(0, len(flares_today[0])):
                            plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                            plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                    else:
                        plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

                plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
                plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

                # ax1.set_xlim([9.84, 9.85])
                if type(timeframe) == str:
                    ax1.set_xlim([-0.5, 24.5])
                else:
                    ax1.set_xlim(timeframe)
                    mask_low = plot_time_bin > timeframe[0]
                    mask_high = plot_time_bin < timeframe[1]
                    count_interval = counts[mask_low & mask_high]
                    high = np.amax(count_interval) + 200
                    ax1.set_ylim([-300, high])
                # ax1.set_ylim([-500, 750])

                binned_counts = counts[0::binning]
                binned_times_bin = plot_time_bin[int(binning / 2)::binning]
                # binned_times_sat = plot_time_sat[int(binning/2)::binning]

                for l in range(1, binning):
                    if len(binned_counts) != len(counts[l::binning]):
                        binned_counts = np.delete(binned_counts, -1)
                    binned_counts = np.add(binned_counts, counts[l::binning])

                if len(binned_counts) < len(binned_times_bin):
                    binned_times_bin = np.delete(binned_times_bin, -1)

                binned_counts = np.array(binned_counts) / binning

                plot1 = ax1.step(binned_times_bin, binned_counts, 'b-', label='Count rate')
                plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label='Fitted Background Curve')
                plot3 = ax1.plot(plot_time_sat, sun_ang_bin, 'y-', label='Sun effective angle')
                plot4 = ax1.plot(plot_time_sat, earth_ang_bin, 'c-', label='Earth occultation')
                plot5 = ax1.plot(plot_time_sat, magnetic, 'g-', label='McIlwain L-parameter')
                plot6 = ax1.plot(plot_time_sat, cgb, 'b--', label='CGB')
                plot7 = ax1.plot(plot_time_sat, sources_fit_curve, 'k-', label='Point-like sources')

                plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6 + plot7
                labels = [l.get_label() for l in plots]
                ax1.legend(plots, labels, loc=4, fontsize=20)

                ax1.grid()

                ax1.set_xlabel('Time of day in 24h', fontsize=40)
                ax1.set_ylabel('Combined count rate', fontsize=40)
                ax1.tick_params(axis='both', which='major', labelsize=30)

                plt.title('Combined detectors on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[
                                                                                             0:3] + ' ' + str(year),
                          fontsize=40)

                figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
                if not os.access(figure_path, os.F_OK):
                    os.mkdir(figure_path)

                if detector_names.size == 1:
                    detec_name = detector_names[0]
                else:
                    detec_name = detector_names[0] + '_to_' + detector_names[-1]
                if echanstart == 'None' and echanstop == 'None':
                    if echans.size == 1:
                        echan_name = str(echans[0])
                    else:
                        echan_name = str(echans[j])
                else:
                    echan_name = str(echanstart + j)

                figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(
                    data_type) + '_' + detec_name + '_e' + echan_name + '.png'

                fig = plt.gcf()  # get current figure
                fig.set_size_inches(20, 12)

                figure = os.path.join(figure_path, figure_name)
                plt.savefig(figure, bbox_inches='tight', dpi=80)

                if show == 'yes':
                    plt.show()

                fig.clf()
                plt.clf()

                ###plot residual noise of the fitting algorithm###

                residuals = counts - fit_curve

                mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

                if burst_time != 'none':
                    if len(burst_time) > 2:
                        for i in range(0, len(burst_time)):
                            if i == 0:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black',
                                            linestyle='--')
                            elif i < 4:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                            else:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue',
                                            linestyle='-.')
                    elif len(burst_time) == 2:
                        plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                    else:
                        plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

                # plot vertical lines for the solar flares of the day
                if np.all(flares_today != -5):
                    if len(flares_today[0]) > 1:
                        for i in range(0, len(flares_today[0])):
                            plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                            plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                    else:
                        plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

                plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
                plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

                binned_res = residuals[0::binning]
                binned_times_bin = plot_time_bin[int(binning / 2)::binning]
                # binned_times_sat = plot_time_sat[int(binning/2)::binning]

                for l in range(1, binning):
                    if len(binned_res) != len(residuals[l::binning]):
                        binned_res = np.delete(binned_res, -1)
                    binned_res = np.add(binned_res, residuals[l::binning])

                if len(binned_res) < len(binned_times_bin):
                    binned_times_bin = np.delete(binned_times_bin, -1)

                residuals = np.array(binned_res) / binning

                plt.step(binned_times_bin, residuals, 'b-')

                plt.xlabel('Time of day in 24h', fontsize=40)
                plt.ylabel('Residual noise', fontsize=40)
                plt.tick_params(axis='both', which='major', labelsize=30)

                plt.grid()

                plt.title(
                    'Combined detector residuals on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[
                                                                                                0:3] + ' ' + str(year),
                    fontsize=40)

                if type(timeframe) == str:
                    plt.xlim([-0.5, 24.5])
                    # plt.ylim([np.mean(residuals)-500, np.mean(residuals)+500])
                else:
                    plt.xlim(timeframe)
                    # plt.ylim([np.min(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-50, np.max(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+50])
                    # plt.ylim([np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-500, np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+500])

                plt.ylim([-600, 1100])

                figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
                if not os.access(figure_path, os.F_OK):
                    os.mkdir(figure_path)

                figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(data_type) + '_' + str(
                    detec_name) + '_e' + str(echan_name) + '_res.png'

                fig = plt.gcf()  # get current figure
                fig.set_size_inches(20, 12)

                figure = os.path.join(figure_path, figure_name)
                plt.savefig(figure, bbox_inches='tight', dpi=80)

                if show == 'yes':
                    plt.show()

                fig.clf()
                plt.clf()

            ##################################################################################################################
            #############################################detectors############################################################
            ##################################################################################################################


        elif analysis == 'detectors':
            # for each detector calculate all fits for the energy channels and sum up the results. Save them seperately for each detector.
            if echanstart == 'None' and echanstop == 'None':
                data_detectors = []
                if detector_names.size == 1:
                    if echans.size == 1:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echans[0], data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for echan in echans:
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_detectors = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    for detector_name in detector_names:
                        if echans.size == 1:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echans[0], data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for echan in echans:
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_detectors.append(temp_data)

                    data_detectors = np.array(data_detectors)




            elif echanstart != 'None' and echanstop != 'None':
                data_detectors = []
                if detector_names.size == 1:
                    if echanstart == echanstop:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echanstart, data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for echan in range(echanstart, echanstop + 1):
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_detectors = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    for detector_name in detector_names:
                        if echanstart == echanstop:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echanstart, data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for echan in range(echanstart, echanstop + 1):
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_detectors.append(temp_data)

                    data_detectors = np.array(data_detectors)

            else:
                print 'Unknown energy channel Input. Please insert certain energy channels or choose a starting and an ending energy channel.'
                return

            combined_data = data_detectors
            #####plot-algorithm#####

            mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

            for j in range(0, detector_names.size):
                if detector_names.size == 1:
                    residuals = combined_data[0]
                    counts = combined_data[1]
                    fit_curve = combined_data[2]
                    cgb = combined_data[3]
                    magnetic = combined_data[4]
                    earth_ang_bin = combined_data[5]
                    sun_ang_bin = combined_data[6]
                    sources_fit_curve = combined_data[7]
                    plot_time_bin = combined_data[8]  # Time of day in hours
                    plot_time_sat = combined_data[9]  # Time of day in hours
                else:
                    residuals = combined_data[j][0]
                    counts = combined_data[j][1]
                    fit_curve = combined_data[j][2]
                    cgb = combined_data[j][3]
                    magnetic = combined_data[j][4]
                    earth_ang_bin = combined_data[j][5]
                    sun_ang_bin = combined_data[j][6]
                    sources_fit_curve = combined_data[j][7]
                    plot_time_bin = combined_data[j][8]  # Time of day in hours
                    plot_time_sat = combined_data[j][9]  # Time of day in hours

                ###plot each on the same axis as converted to counts###
                fig, ax1 = plt.subplots()

                if burst_time != 'none':
                    if len(burst_time) > 2:
                        for i in range(0, len(burst_time)):
                            if i == 0:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black',
                                            linestyle='--')
                            elif i < 4:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                            else:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue',
                                            linestyle='-.')
                    elif len(burst_time) == 2:
                        plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                    else:
                        plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

                # plot vertical lines for the solar flares of the day
                if flares_today.size != 0:
                    if len(flares_today[0]) > 1:
                        for i in range(0, len(flares_today[0])):
                            plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                            plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                    else:
                        plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

                plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
                plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

                # ax1.set_xlim([9.84, 9.85])
                if type(timeframe) == str:
                    ax1.set_xlim([-0.5, 24.5])
                else:
                    ax1.set_xlim(timeframe)
                    mask_low = plot_time_bin > timeframe[0]
                    mask_high = plot_time_bin < timeframe[1]
                    count_interval = counts[mask_low & mask_high]
                    high = np.amax(count_interval) + 200
                    ax1.set_ylim([-300, high])
                # ax1.set_ylim([-500, 750])

                binned_counts = counts[0::binning]
                binned_times_bin = plot_time_bin[int(binning / 2)::binning]
                # binned_times_sat = plot_time_sat[int(binning/2)::binning]

                for l in range(1, binning):
                    if len(binned_counts) != len(counts[l::binning]):
                        binned_counts = np.delete(binned_counts, -1)
                    binned_counts = np.add(binned_counts, counts[l::binning])

                if len(binned_counts) < len(binned_times_bin):
                    binned_times_bin = np.delete(binned_times_bin, -1)

                binned_counts = np.array(binned_counts) / binning

                plot1 = ax1.step(binned_times_bin, binned_counts, 'b-', label='Count rate')
                plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label='Fitted Background Curve')
                plot3 = ax1.plot(plot_time_sat, sun_ang_bin, 'y-', label='Sun effective angle')
                plot4 = ax1.plot(plot_time_sat, earth_ang_bin, 'c-', label='Earth occultation')
                plot5 = ax1.plot(plot_time_sat, magnetic, 'g-', label='McIlwain L-parameter')
                plot6 = ax1.plot(plot_time_sat, cgb, 'b--', label='CGB')
                plot7 = ax1.plot(plot_time_sat, sources_fit_curve, 'k-', label='Point-like sources')

                plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6 + plot7
                labels = [l.get_label() for l in plots]
                ax1.legend(plots, labels, loc=4, fontsize=20)

                ax1.grid()

                ax1.set_xlabel('Time of day in 24h', fontsize=40)
                ax1.set_ylabel('Combined count rate', fontsize=40)
                ax1.tick_params(axis='both', which='major', labelsize=30)

                plt.title('Combined energy channels on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[
                                                                                                   0:3] + ' ' + str(
                    year), fontsize=40)

                figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
                if not os.access(figure_path, os.F_OK):
                    os.mkdir(figure_path)

                if detector_names.size == 1:
                    detec_name = detector_names[0]
                else:
                    detec_name = detector_names[j]
                if echanstart == 'None' and echanstop == 'None':
                    if echans.size == 1:
                        echan_name = str(echans[0])
                    else:
                        echan_name = str(echans[0]) + '_to_e' + str(echans[-1])
                else:
                    echan_name = str(echanstart) + '_to_e' + str(echanstop)

                figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(
                    data_type) + '_' + detec_name + '_e' + echan_name + '.png'

                fig = plt.gcf()  # get current figure
                fig.set_size_inches(20, 12)

                figure = os.path.join(figure_path, figure_name)
                plt.savefig(figure, bbox_inches='tight', dpi=80)

                if show == 'yes':
                    plt.show()

                fig.clf()
                plt.clf()

                ###plot residual noise of the fitting algorithm###

                residuals = counts - fit_curve

                mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

                if burst_time != 'none':
                    if len(burst_time) > 2:
                        for i in range(0, len(burst_time)):
                            if i == 0:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black',
                                            linestyle='--')
                            elif i < 4:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                            else:
                                plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue',
                                            linestyle='-.')
                    elif len(burst_time) == 2:
                        plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                    else:
                        plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

                # plot vertical lines for the solar flares of the day
                if np.all(flares_today != -5):
                    if len(flares_today[0]) > 1:
                        for i in range(0, len(flares_today[0])):
                            plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                            plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                    else:
                        plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

                plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
                plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

                binned_res = residuals[0::binning]
                binned_times_bin = plot_time_bin[int(binning / 2)::binning]
                # binned_times_sat = plot_time_sat[int(binning/2)::binning]

                for l in range(1, binning):
                    if len(binned_res) != len(residuals[l::binning]):
                        binned_res = np.delete(binned_res, -1)
                    binned_res = np.add(binned_res, residuals[l::binning])

                if len(binned_res) < len(binned_times_bin):
                    binned_times_bin = np.delete(binned_times_bin, -1)

                residuals = np.array(binned_res) / binning

                plt.step(binned_times_bin, residuals, 'b-')

                plt.xlabel('Time of day in 24h', fontsize=40)
                plt.ylabel('Residual noise', fontsize=40)
                plt.tick_params(axis='both', which='major', labelsize=30)

                plt.grid()

                plt.title(
                    'Combined energy channel residuals on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime(
                        '%B')[0:3] + ' ' + str(year), fontsize=40)

                if type(timeframe) == str:
                    plt.xlim([-0.5, 24.5])
                    # plt.ylim([np.mean(residuals)-500, np.mean(residuals)+500])
                else:
                    plt.xlim(timeframe)
                    # plt.ylim([np.min(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-50, np.max(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+50])
                    # plt.ylim([np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-500, np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+500])

                plt.ylim([-400, 500])

                figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
                if not os.access(figure_path, os.F_OK):
                    os.mkdir(figure_path)

                figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(
                    data_type) + '_' + detec_name + '_e' + echan_name + '_res.png'

                fig = plt.gcf()  # get current figure
                fig.set_size_inches(20, 12)

                figure = os.path.join(figure_path, figure_name)
                plt.savefig(figure, bbox_inches='tight', dpi=80)

                if show == 'yes':
                    plt.show()

                fig.clf()
                plt.clf()



            ##################################################################################################################
            ################################################both#############################################################
            ##################################################################################################################

        elif analysis == 'both':
            # for all detectors and energy channels calculate the fits and sum up the results. Combine all of them.
            if echanstart == 'None' and echanstop == 'None':
                data_both = []
                if detector_names.size == 1:
                    if echans.size == 1:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echans[0], data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for echan in echans:
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_both = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    data_detectors = []
                    for detector_name in detector_names:
                        if echans.size == 1:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echans[0], data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for echan in echans:
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_detectors.append(temp_data)

                    data_detectors = np.array(data_detectors)
                    data_both = np.squeeze(np.sum(data_detectors, axis=0))

            if echanstart != 'None' and echanstop != 'None':
                data_detectors = []
                if detector_names.size == 1:
                    if echanstart == echanstop:
                        data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echanstart, data_type)
                        residuals = data[0]
                        counts = data[1]
                        fit_curve = data[2]
                        cgb = data[3]
                        magnetic = data[4]
                        earth_ang_bin = data[5]
                        sun_ang_bin = data[6]
                        sources_fit_curve = data[7]
                        plot_time_bin = data[8]
                        plot_time_sat = data[9]
                    else:
                        residuals = np.zeros(array_length)
                        counts = np.zeros(array_length)
                        fit_curve = np.zeros(array_length)
                        cgb = np.zeros(array_length)
                        magnetic = np.zeros(array_length)
                        earth_ang_bin = np.zeros(array_length)
                        sun_ang_bin = np.zeros(array_length)
                        sources_fit_curve = np.zeros(array_length)
                        for echan in range(echanstart, echanstop + 1):
                            data = calculate.curve_fit_plots(calculate(), day, detector_names[0], echan, data_type)
                            residuals = np.add(residuals, data[0])
                            counts = np.add(counts, data[1])
                            fit_curve = np.add(fit_curve, data[2])
                            cgb = np.add(cgb, data[3])
                            magnetic = np.add(magnetic, data[4])
                            earth_ang_bin = np.add(earth_ang_bin, data[5])
                            sun_ang_bin = np.add(sun_ang_bin, data[6])
                            sources_fit_curve = np.add(sources_fit_curve, data[7])
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                    data_both = np.array(
                        [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                         plot_time_bin, plot_time_sat])

                else:
                    data_detectors = []
                    for detector_name in detector_names:
                        if echanstart == echanstop:
                            data = calculate.curve_fit_plots(calculate(), day, detector_name, echanstart, data_type)
                            residuals = data[0]
                            counts = data[1]
                            fit_curve = data[2]
                            cgb = data[3]
                            magnetic = data[4]
                            earth_ang_bin = data[5]
                            sun_ang_bin = data[6]
                            sources_fit_curve = data[7]
                            plot_time_bin = data[8]
                            plot_time_sat = data[9]
                        else:
                            residuals = np.zeros(array_length)
                            counts = np.zeros(array_length)
                            fit_curve = np.zeros(array_length)
                            cgb = np.zeros(array_length)
                            magnetic = np.zeros(array_length)
                            earth_ang_bin = np.zeros(array_length)
                            sun_ang_bin = np.zeros(array_length)
                            sources_fit_curve = np.zeros(array_length)
                            for echan in range(echanstart, echanstop + 1):
                                data = calculate.curve_fit_plots(calculate(), day, detector_name, echan, data_type)
                                residuals = np.add(residuals, data[0])
                                counts = np.add(counts, data[1])
                                fit_curve = np.add(fit_curve, data[2])
                                cgb = np.add(cgb, data[3])
                                magnetic = np.add(magnetic, data[4])
                                earth_ang_bin = np.add(earth_ang_bin, data[5])
                                sun_ang_bin = np.add(sun_ang_bin, data[6])
                                sources_fit_curve = np.add(sources_fit_curve, data[7])
                                plot_time_bin = data[8]
                                plot_time_sat = data[9]

                        temp_data = np.array(
                            [residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve,
                             plot_time_bin, plot_time_sat])
                        data_detectors.append(temp_data)

                    data_detectors = np.array(data_detectors)
                    data_both = np.squeeze(np.sum(data_detectors, axis=0))

            combined_data = data_both

            #####plot-algorithm#####

            mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

            residuals = combined_data[0]
            counts = combined_data[1]
            fit_curve = combined_data[2]
            cgb = combined_data[3]
            magnetic = combined_data[4]
            earth_ang_bin = combined_data[5]
            sun_ang_bin = combined_data[6]
            sources_fit_curve = combined_data[7]

            ###plot each on the same axis as converted to counts###
            fig, ax1 = plt.subplots()

            if burst_time != 'none':
                if len(burst_time) > 2:
                    for i in range(0, len(burst_time)):
                        if i == 0:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        elif i < 4:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                        else:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue', linestyle='-.')
                elif len(burst_time) == 2:
                    plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                    plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                else:
                    plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

            # plot vertical lines for the solar flares of the day
            if flares_today.size != 0:
                if len(flares_today[0]) > 1:
                    for i in range(0, len(flares_today[0])):
                        plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                else:
                    plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

            plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
            plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

            # ax1.set_xlim([9.84, 9.85])
            if type(timeframe) == str:
                ax1.set_xlim([-0.5, 24.5])
            else:
                ax1.set_xlim(timeframe)
                mask_low = plot_time_bin > timeframe[0]
                mask_high = plot_time_bin < timeframe[1]
                count_interval = counts[mask_low & mask_high]
                high = np.amax(count_interval) + 200
                ax1.set_ylim([-300, high])
            # ax1.set_ylim([-500, 750])

            binned_counts = counts[0::binning]
            binned_times_bin = plot_time_bin[int(binning / 2)::binning]
            # binned_times_sat = plot_time_sat[int(binning/2)::binning]

            for l in range(1, binning):
                if len(binned_counts) != len(counts[l::binning]):
                    binned_counts = np.delete(binned_counts, -1)
                binned_counts = np.add(binned_counts, counts[l::binning])

            if len(binned_counts) < len(binned_times_bin):
                binned_times_bin = np.delete(binned_times_bin, -1)

            binned_counts = np.array(binned_counts) / binning

            plot1 = ax1.step(binned_times_bin, binned_counts, 'b-', label='Count rate')
            plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label='Fitted Background Curve')
            plot3 = ax1.plot(plot_time_sat, sun_ang_bin, 'y-', label='Sun effective angle')
            plot4 = ax1.plot(plot_time_sat, earth_ang_bin, 'c-', label='Earth occultation')
            plot5 = ax1.plot(plot_time_sat, magnetic, 'g-', label='McIlwain L-parameter')
            plot6 = ax1.plot(plot_time_sat, cgb, 'b--', label='CGB')
            plot7 = ax1.plot(plot_time_sat, sources_fit_curve, 'k-', label='Point-like sources')

            plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6 + plot7
            labels = [l.get_label() for l in plots]
            ax1.legend(plots, labels, loc=4, fontsize=20)

            ax1.grid()

            ax1.set_xlabel('Time of day in 24h', fontsize=40)
            ax1.set_ylabel('Combined count rate', fontsize=40)
            ax1.tick_params(axis='both', which='major', labelsize=30)

            plt.title(
                'Combined fit on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year),
                fontsize=40)

            figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
            if not os.access(figure_path, os.F_OK):
                os.mkdir(figure_path)

            if detector_names.size == 1:
                detec_name = detector_names[0]
            else:
                detec_name = detector_names[0] + '_to_' + detector_names[-1]
            if echanstart == 'None' and echanstop == 'None':
                if echans.size == 1:
                    echan_name = str(echans[0])
                else:
                    echan_name = str(echans[0]) + '_to_e' + str(echans[-1])
            else:
                echan_name = str(echanstart) + '_to_e' + str(echanstop)

            figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(
                data_type) + '_' + detec_name + '_e' + echan_name + '.png'

            fig = plt.gcf()  # get current figure
            fig.set_size_inches(20, 12)

            figure = os.path.join(figure_path, figure_name)
            plt.savefig(figure, bbox_inches='tight', dpi=80)

            if show == 'yes':
                plt.show()

            fig.clf()
            plt.clf()

            ###plot residual noise of the fitting algorithm###

            mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

            residuals = counts - fit_curve

            if burst_time != 'none':
                if len(burst_time) > 2:
                    for i in range(0, len(burst_time)):
                        if i == 0:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                        elif i < 4:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='red', linestyle='--')
                        else:
                            plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='blue', linestyle='-.')
                elif len(burst_time) == 2:
                    plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                    plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
                else:
                    plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

            # plot vertical lines for the solar flares of the day
            if np.all(flares_today != -5):
                if len(flares_today[0]) > 1:
                    for i in range(0, len(flares_today[0])):
                        plt.axvline(x=flares_today[0, i], ymin=0., ymax=1., linewidth=2, color='grey')
                        plt.axvline(x=flares_today[1, i], ymin=0., ymax=1., color='grey', linestyle='--')
                else:
                    plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

            plt.axvline(x=7.090, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')
            plt.axvline(x=7.587, ymin=0., ymax=1., linewidth=2, color='green', linestyle='-.')

            binned_res = residuals[0::binning]
            binned_times_bin = plot_time_bin[int(binning / 2)::binning]
            # binned_times_sat = plot_time_sat[int(binning/2)::binning]

            for l in range(1, binning):
                if len(binned_res) != len(residuals[l::binning]):
                    binned_res = np.delete(binned_res, -1)
                binned_res = np.add(binned_res, residuals[l::binning])

            if len(binned_res) < len(binned_times_bin):
                binned_times_bin = np.delete(binned_times_bin, -1)

            residuals = np.array(binned_res) / binning

            plt.step(binned_times_bin, residuals, 'b-')

            plt.xlabel('Time of day in 24h', fontsize=40)
            plt.ylabel('Residual noise', fontsize=40)
            plt.tick_params(axis='both', which='major', labelsize=30)

            plt.grid()

            plt.title(
                'Combined residuals on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(
                    year), fontsize=40)

            if type(timeframe) == str:
                plt.xlim([-0.5, 24.5])
                # plt.ylim([np.mean(residuals)-500, np.mean(residuals)+500])
            else:
                plt.xlim(timeframe)
                # plt.ylim([np.min(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-50, np.max(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+50])
                # plt.ylim([np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])-500, np.mean(residuals[np.where(plot_time_bin>timeframe[0])[0][0]:np.where(plot_time_bin<timeframe[1])[0][-1]])+500])

            plt.ylim([-2000, 3000])

            figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
            if not os.access(figure_path, os.F_OK):
                os.mkdir(figure_path)

            figure_name = str(day) + '_combined_' + str(analysis) + '_' + str(
                data_type) + '_' + detec_name + '_e' + echan_name + '_res.png'

            fig = plt.gcf()  # get current figure
            fig.set_size_inches(20, 12)

            figure = os.path.join(figure_path, figure_name)
            plt.savefig(figure, bbox_inches='tight', dpi=80)

            if show == 'yes':
                plt.show()

            fig.clf()
            plt.clf()


        else:
            print 'Unknown analysis input. Please choose between energychannels, detectors or both.'
            return

        return

    def date_to_met(self, year, month=01, day=01, hour=00, minute=00, seconds=00.):
        """This function converts a date into the MET format and stores it in the form: met, mjdutc.\n
        Input:\n
        calculate.date_to_met ( year, month = 01, day = 01, hour = 00, minute = 00, seconds = 00. )\n
        Output:\n
        0 = met\n
        1 = mjdutc"""

        times = str(year) + '-' + str(month) + '-' + str(day) + 'T' + str(hour) + ':' + str(minute) + ':' + str(
            seconds)  # create date-string in the isot-format (YYYY-MM-DDTHH:MM:SS)
        t = Time(times, format='isot', scale='utc')  # use the Time-function to calculate the utc-time
        mjdutc = t.mjd  # get the Mid-Julian-Date attribute

        if mjdutc <= 54832.00000000:
            utc_tt_diff = 65.184
        elif mjdutc <= 56109.00000000:
            utc_tt_diff = 66.184
        elif mjdutc <= 57204.00000000:
            utc_tt_diff = 67.184
        elif mjdutc <= 57754.00000000:
            utc_tt_diff = 68.184
        else:
            utc_tt_diff = 69.184

        met = (mjdutc - 51910 - 0.0007428703703) * 86400.0 + utc_tt_diff  # convert it into MET
        return met, mjdutc

    def day_to_met(self, day):
        """This function converts a day into the MET format and stores it in the form: met, mjdutc.\n
        Input:\n
        calculate.day_to_met ( day = YYMMDD )\n
        Output:\n
        0 = met\n
        1 = mjdutc"""

        times = '20' + str(day)[0:2] + '-' + str(day)[2:4] + '-' + str(day)[
                                                                   4:6] + 'T00:00:00.0'  # create date-string in the isot-format (YYYY-MM-DDTHH:MM:SS)
        t = Time(times, format='isot', scale='utc')  # use the Time-function to calculate the utc-time
        mjdutc = t.mjd  # get the Mid-Julian-Date attribute

        if mjdutc <= 54832.00000000:
            utc_tt_diff = 65.184
        elif mjdutc <= 56109.00000000:
            utc_tt_diff = 66.184
        elif mjdutc <= 57204.00000000:
            utc_tt_diff = 67.184
        elif mjdutc <= 57754.00000000:
            utc_tt_diff = 68.184
        else:
            utc_tt_diff = 69.184

        met = (mjdutc - 51910 - 0.0007428703703) * 86400.0 + utc_tt_diff  # convert it into MET
        return met, mjdutc

    def det_or(self, detector_name, day):
        """This function reads a posthist file and the detector assembly table to calculate the detector orientation and stores it in arrays of the form: det_coor, det_rad, sat_pos, sat_time\n
        Input:\n
        calculate.det_or( detector = n0/n1/b0.., day = YYMMDD )\n
        Output:\n
        0 = detector coordinates[x[], y[], z[]]\n
        1 = detector geocentric angles [(right ascension, declination)]\n
        2 = satellite position [x[], y[], z[]]\n
        3 = time (MET) in seconds"""

        # get satellite data for the convertion
        sat_data = ExternalProps.poshist(ExternalProps(), day)
        sat_time = sat_data[0]
        sat_pos = sat_data[1]
        sat_q = sat_data[4]

        # get detector orientation data (in sat-coordinates) from the defined detector-class
        det = getattr(detector(), detector_name)
        az = det.azimuth
        zen = det.zenith
        det_pos = np.array([math.cos(az) * math.sin(zen), math.sin(az) * math.sin(zen),
                            math.cos(zen)])  # convert into unit-vector in the satellite coordinate system

        # convert the orientation in geo-coordinates
        det_geo = calculate.sat_to_geo(calculate(), sat_q, det_pos)
        det_coor = det_geo[0]  # unit-vector
        det_rad = det_geo[1]
        return det_coor, det_rad, sat_pos, sat_time

    def det_or_bin(self, detector_name, day, bin_time_mid=0, data_type='ctime'):
        """This function reads a posthist file and the detector assembly table to calculate the binned detector orientation and stores it in arrays of the form: det_coor_bin, det_rad_bin, sat_pos_bin, sat_time_bin, bin_time_mid\n
        Input:\n
        calculate.det_or_bin( detector = n0/n1/b0.., day = YYMMDD, bin_time_mid = 0, detector = 0, data_type = 'ctime' )\n
        Output:\n
        0 = detector coordinates[x[], y[], z[]]\n
        1 = detector geocentric angles [(right ascension, declination)]\n
        2 = satellite position [x[], y[], z[]]\n
        3 = time (MET) in seconds\n
        4 = bin_time_mid"""

        # get satellite data for the convertion
        sat_data = ExternalProps.poshist_bin(ExternalProps(), day, bin_time_mid, detector_name, data_type)
        sat_time_bin = sat_data[0]
        sat_pos_bin = sat_data[1]
        sat_q_bin = sat_data[4]
        bin_time_mid = sat_data[5]

        # get detector orientation data (in sat-coordinates) from the defined detector-class
        det = getattr(detector(), detector_name)
        az = det.azimuth
        zen = det.zenith
        det_pos = np.array([math.cos(az) * math.sin(zen), math.sin(az) * math.sin(zen),
                            math.cos(zen)])  # convert into unit-vector in the satellite coordinate system

        # convert the orientation in geo-coordinates
        det_geo = calculate.sat_to_geo(calculate(), sat_q_bin, det_pos)
        det_coor_bin = det_geo[0]  # unit-vector
        det_rad_bin = det_geo[1]
        return det_coor_bin, det_rad_bin, sat_pos_bin, sat_time_bin, bin_time_mid

    def earth_ang(self, detector_name, day):
        """This function calculates the earth occultation for one detector and stores the data in arrays of the form: earth_ang, sat_time\n
        Input:\n
        calculate.earth_ang ( detector, day = JJMMDD )\n
        Output:\n
        0 = angle between the detector orientation and the earth position\n
        1 = time (MET) in seconds"""

        # get the detector and satellite data
        data = calculate.det_or(calculate(), detector_name, day)
        det_coor = data[0]  # unit-vector of the detector orientation
        det_rad = data[1]  # detector orientation in right ascension and declination
        sat_pos = data[2]  # position of the satellite
        sat_time = np.array(data[3])  # time (MET) in seconds

        # calculate the earth location unit-vector
        sat_dist = LA.norm(sat_pos, axis=0)  # get the altitude of the satellite (length of the position vector)
        sat_pos_unit = sat_pos / sat_dist  # convert the position vector into a unit-vector
        geo_pos_unit = -sat_pos_unit

        # calculate the angle between the earth location and the detector orientation
        scalar_product = det_coor[0] * geo_pos_unit[0] + det_coor[1] * geo_pos_unit[1] + det_coor[2] * geo_pos_unit[2]
        ang_det_geo = np.arccos(scalar_product)
        earth_ang = ang_det_geo * 360. / (2. * math.pi)
        earth_ang = np.array(earth_ang)
        return earth_ang, sat_time

    def earth_ang_bin(self, detector_name, day, bin_time_mid=0, data_type='ctime'):
        """This function calculates the binned earth occultation for one detector and stores the data in arrays of the form: earth_ang_bin, sat_time_bin, bin_time_mid\n
        Input:\n
        calculate.earth_ang_bin ( detector, day = JJMMDD, bin_time_mid = 0, detector = 0, data_type = 'ctime' )\n
        Output:\n
        0 = angle between the detector orientation and the earth position\n
        1 = time (MET) in seconds
        2 = bin_time_mid"""

        # get the detector and satellite data
        data = calculate.det_or_bin(calculate(), detector_name, day, bin_time_mid, data_type)
        det_coor_bin = data[0]  # unit-vector of the detector orientation
        det_rad_bin = data[1]  # detector orientation in right ascension and declination
        sat_pos_bin = data[2]  # position of the satellite
        sat_time_bin = np.array(data[3])  # time (MET) in seconds
        bin_time_mid = data[4]

        # calculate the earth location unit-vector
        sat_dist = LA.norm(sat_pos_bin, axis=0)  # get the altitude of the satellite (length of the position vector)
        sat_pos_unit = sat_pos_bin / sat_dist  # convert the position vector into a unit-vector
        geo_pos_unit = -sat_pos_unit

        # calculate the angle between the earth location and the detector orientation
        scalar_product = det_coor_bin[0] * geo_pos_unit[0] + det_coor_bin[1] * geo_pos_unit[1] + det_coor_bin[2] * \
                                                                                                 geo_pos_unit[2]
        ang_det_geo = np.arccos(scalar_product)
        earth_ang_bin = ang_det_geo * 360. / (2. * math.pi)
        earth_ang_bin = np.array(earth_ang_bin)
        return earth_ang_bin, sat_time_bin, bin_time_mid

    def earth_occ_eff(self, earth_ang, echan, datatype='ctime', detectortype='NaI'):
        """This function converts the earth angle into an effective earth occultation considering the angular dependence of the effective area and stores the data in an array of the form: earth_occ_eff\n
        Input:\n
        calculate.earth_occ_eff ( earth_ang (in degrees), echan (integer in the range of 0-7 or 0-127), datatype='ctime' (or 'cspec'), detectortype='NaI' (or 'BGO') )\n
        Output:\n
        0 = effective unocculted detector area"""

        fitsname = 'peak_eff_area_angle_calib_GBM_all_DRM.fits'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/calibration/'
        fitsfilepath = os.path.join(fits_path, fitsname)
        fitsfile = fits.open(fitsfilepath, mode='update')
        data = fitsfile[1].data
        fitsfile.close()
        x = data.field(0)
        y0 = data.field(1)  # for NaI (4-12 keV) gewichteter Mittelwert = 10.76 keV
        y1 = data.field(2)  # for NaI (12-27 keV) gewichteter Mittelwert = 20.42 keV
        y2 = data.field(3)  # for NaI (27-50 keV) gewichteter Mittelwert = 38.80 keV
        y3 = data.field(4)  # for NaI (50-102 keV) gewichteter Mittelwert = 76.37 keV
        y4 = data.field(5)  # for NaI (102-295 keV) gewichteter Mittelwert = 190.19 keV
        y5 = data.field(6)  # for NaI (295-540 keV) gewichteter Mittelwert = 410.91 keV
        y6 = data.field(7)  # for NaI (540-985 keV) gewichteter Mittelwert = 751.94 keV
        y7 = data.field(8)  # for NaI (985-2000 keV) gewichteter Mittelwert = 1466.43 keV
        # y4 = data.field(4)#for BGO (898 keV)
        # y5 = data.field(5)#for BGO (1836 keV)

        y_all = np.array([y0, y1, y2, y3, y4, y5, y6, y7])

        data = ExternalProps.earth_occ(ExternalProps())
        earth_ang_0 = data[0]
        angle_d = data[1][0]
        area_frac = data[2]
        free_area = data[3][0]
        occ_area = data[4]

        if detectortype == 'NaI':
            if datatype == 'ctime':
                # ctime linear-interpolation factors
                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 127./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 109./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 77./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 159./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # resulting effective area curve
                y = y_all[echan]

                # normalize
                # y = y/y[90]

                # calculate the angle factors
                tck = interpolate.splrep(x, y)
                ang_fac = interpolate.splev(angle_d, tck, der=0)

            else:
                print 'datatype cspec not yet implemented'

        else:
            print 'detectortype BGO not yet implemented'

        free_circ_eff = [free_area[0] * ang_fac[0]]

        for i in range(1, len(free_area)):
            circ_area = free_area[i] - free_area[i - 1]
            circ_area_eff = circ_area * ang_fac[i]
            free_circ_eff.append(circ_area_eff)

        free_circ_eff = np.array(free_circ_eff)

        occ_circ_eff = []

        for j in range(0, len(earth_ang_0)):
            occ_circ_eff_0 = [occ_area[j][0] * ang_fac[0]]
            for i in range(1, len(occ_area[j])):
                circ_area = occ_area[j][i] - occ_area[j][i - 1]
                circ_area_eff = circ_area * ang_fac[i]
                occ_circ_eff_0.append(circ_area_eff)

            occ_circ_eff.append(occ_circ_eff_0)

        occ_circ_eff = np.array(occ_circ_eff)
        # eff_area_frac = np.sum(occ_circ_eff)/np.sum(free_circ_eff)
        eff_area_frac_0 = np.sum(occ_circ_eff, axis=1) / np.sum(free_circ_eff)

        tck = interpolate.splrep(earth_ang_0, eff_area_frac_0, s=0)
        eff_area_frac = interpolate.splev(earth_ang, tck, der=0)

        return eff_area_frac

    def geo_to_sat(self, sat_q, geo_coor):
        """This function converts the geographical coordinates into satellite coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: sat_coor, sat_rad\n
        Input:\n
        calculate.geo_to_sat ( sat_q = quaternion-matrix, geo_coor = 3D-array(x, y, z) )\n
        Output:\n
        0 = satellite coordinates[x[], y[], z[]]\n
        1 = satellite angle[(azimuth, zenith)]"""

        # calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
        nt = np.size(sat_q[0])
        scx = np.zeros((nt, 3), float)
        scx[:, 0] = (sat_q[0] ** 2 - sat_q[1] ** 2 - sat_q[2] ** 2 + sat_q[3] ** 2)
        scx[:, 1] = 2. * (sat_q[0] * sat_q[1] + sat_q[3] * sat_q[2])
        scx[:, 2] = 2. * (sat_q[0] * sat_q[2] - sat_q[3] * sat_q[1])
        scy = np.zeros((nt, 3), float)
        scy[:, 0] = 2. * (sat_q[0] * sat_q[1] - sat_q[3] * sat_q[2])
        scy[:, 1] = (-sat_q[0] ** 2 + sat_q[1] ** 2 - sat_q[2] ** 2 + sat_q[3] ** 2)
        scy[:, 2] = 2. * (sat_q[1] * sat_q[2] + sat_q[3] * sat_q[0])
        scz = np.zeros((nt, 3), float)
        scz[:, 0] = 2. * (sat_q[0] * sat_q[2] + sat_q[3] * sat_q[1])
        scz[:, 1] = 2. * (sat_q[1] * sat_q[2] - sat_q[3] * sat_q[0])
        scz[:, 2] = (-sat_q[0] ** 2 - sat_q[1] ** 2 + sat_q[2] ** 2 + sat_q[3] ** 2)

        # create geo_to_sat rotation matrix
        sat_mat = np.array([scx, scy, scz])

        # convert geo_coordinates to sat_coordinates
        geo_coor = np.array(geo_coor)
        sat_coor = np.zeros((3, nt), float)
        sat_coor[0] = sat_mat[0, :, 0] * geo_coor[0] + sat_mat[0, :, 1] * geo_coor[1] + sat_mat[0, :, 2] * geo_coor[2]
        sat_coor[1] = sat_mat[1, :, 0] * geo_coor[0] + sat_mat[1, :, 1] * geo_coor[1] + sat_mat[1, :, 2] * geo_coor[2]
        sat_coor[2] = sat_mat[2, :, 0] * geo_coor[0] + sat_mat[2, :, 1] * geo_coor[1] + sat_mat[2, :, 2] * geo_coor[2]

        # calculate the azimuth and zenith
        sat_az = np.arctan2(-sat_coor[1], -sat_coor[0]) * 360. / (2. * math.pi) + 180.
        sat_zen = 90. - np.arctan((sat_coor[2] / (sat_coor[0] ** 2 + sat_coor[1] ** 2) ** 0.5)) * 360. / (2. * math.pi)

        # put azimuth and zenith together in one array as [:,0] and [:,1]
        sat_rad = np.zeros((nt, 2), float)
        sat_rad[:, 0] = np.array(sat_az)
        sat_rad[:, 1] = np.array(sat_zen)
        return sat_coor, sat_rad

    def intpol(self, vector, day, direction=0, sat_time=0, bin_time_mid=0, detector_name=0, data_type='ctime'):
        """This function interpolates a vector (from poshist- or count-files) and adjusts the length to the arrays of the other source-file and stores the data in an array of the form: vector\n
        Input:\n
        calc_intpol ( vector, \n
        day = JJMMDD, \n
        direction = 0(from poshist- to count-file(0) or the other way(1); default: 0), \n
        sat_time = 0(input sat_time if available; default: 0), \n
        bin_time_mid = 0(input bin_time_mid if available; default: 0), \n
        detector = 0(input the detector in the form det.n0; default: 0), \n
        data_type = 'ctime'(input ctime or cspec as string; default: 'ctime') )\n
        Output:\n
        0 = vector\n
        1 = sat_time\n
        2 = bin_time_mid"""

        sat_time = np.array(sat_time)
        # get the missing satellite and measurement data, if needed
        if sat_time.all() == 0:
            sat_data = ExternalProps.poshist(ExternalProps(), day)
            sat_time = np.array(sat_data[0])  # time (MET) in seconds

        bin_time_mid = np.array(bin_time_mid)
        if bin_time_mid.all() == 0:
            if detector_name != 0:
                if data_type == 'ctime':
                    bin_data = ExternalProps.ctime(ExternalProps(), detector_name, day)
                    bin_time = np.array(bin_data[5])  # time (MET) in seconds
                    bin_time_mid = np.array((bin_time[:, 0] + bin_time[:,
                                                              1]) / 2)  # convert bin_time into 1-D array. Take the medium of start and stop time of the bin.
                elif data_type == 'cspec':
                    bin_data = ExternalProps.cspec(ExternalProps(), detector_name, day)
                    bin_time = np.array(bin_data[5])  # time (MET) in seconds
                    bin_time_mid = np.array((bin_time[:, 0] + bin_time[:,
                                                              1]) / 2)  # convert bin_time into 1-D array. Take the medium of start and stop time of the bin.
                else:
                    print "Invalid data_type input. Please insert 'ctime' or 'cspec' for the data_type. See .__doc__ for further information."
                    return vector, sat_time, bin_time_mid
            else:
                print "Missing or false detector input. Please insert the chosen detector (f. e. det.n0). See .__doc__ for further information."
                return vector, sat_time, bin_time_mid

        # define x-values depending on the direction of the interpolation
        if direction == 0:
            x1 = np.array(sat_time)
            x2 = np.array(bin_time_mid)
        elif direction == 1:
            x1 = np.array(bin_time_mid)
            x2 = np.array(sat_time)
        else:
            print 'Invalid direction input. Please insert 0 or 1 for the direction. See .__doc__ for further information.'
            return vector, sat_time, bin_time_mid

        vector_shape = np.shape(vector)
        vector_dim = vector_shape[0]

        # interpolate all subvectors of the input vector with splines and evaluate the splines at the new x-values
        if len(vector_shape) == 1:
            tck = interpolate.splrep(x1, vector, s=0)
            new_vector = interpolate.splev(x2, tck, der=0)
        else:
            new_vector = np.zeros((vector_dim, len(x2)), float)
            for i in range(0, vector_dim):
                tck = interpolate.splrep(x1, vector[i], s=0)
                new_vector[i] = interpolate.splev(x2, tck, der=0)

        return new_vector, sat_time, bin_time_mid

    def met_to_date(self, met):
        """This function converts a MET to other times and stores it in the form: mjdutc, mjdtt, isot, date, decimal.\n
        Input:\n
        calculate.met_to_date ( met )\n
        Output:\n
        0 = mjdutc\n
        1 = mjdtt\n
        2 = isot\n
        3 = date\n
        4 = decimal"""

        if isinstance(met, list) or isinstance(met, np.ndarray):
            if met[-1] <= 252460801.000:
                utc_tt_diff = 65.184
            elif met[-1] <= 362793602.000:
                utc_tt_diff = 66.184
            elif met[-1] <= 457401603.000:
                utc_tt_diff = 67.184
            elif met[-1] <= 504921604.000:
                utc_tt_diff = 68.184
            else:
                utc_tt_diff = 69.184
        else:
            if met <= 252460801.000:
                utc_tt_diff = 65.184
            elif met <= 362793602.000:
                utc_tt_diff = 66.184
            elif met <= 457401603.000:
                utc_tt_diff = 67.184
            elif met <= 504921604.000:
                utc_tt_diff = 68.184
            else:
                utc_tt_diff = 69.184

        mjdutc = ((
                  met - utc_tt_diff) / 86400.0) + 51910 + 0.0007428703703  # -68.184 added to account for diff between TT and UTC and the 4 leapseconds since 2001
        mjdtt = ((met) / 86400.0) + 51910 + 0.00074287037037
        mjdtt = Time(mjdtt, scale='tt', format='mjd')
        isot = Time(mjdtt, scale='utc', format='isot')
        date = Time(mjdtt, scale='utc', format='iso')
        decimal = Time(mjdtt, scale='utc', format='decimalyear')
        return mjdutc, mjdtt, isot, date, decimal

    def plots(self, day, detector_name, echan, data_type='ctime', burst_time='none', plot_time='none'):
        """This function plots the chi-square fit to the data of a given detector, day and energy channel and saves the figure in the appropriate folder\n
        Input:\n
        calculate.plots( day = YYMMDD, detector (f. e. 'n5'), echan (9 or 129 = total counts for ctime or cspec), data_type = 'ctime' (or 'cspec'))\n
        Output:\n
        None"""

        sources_data = ExternalProps.point_sources(ExternalProps())
        sources_names = sources_data[0]
        sources_coordinates = sources_data[1]
        sources_number = int(len(sources_names))

        if data_type == 'ctime':
            if echan < 8:
                fitsname_data = 'ctime_' + detector_name + '_e' + str(echan) + '_kt.fits'
            elif echan == 8:
                fitsname_data = 'ctime_' + detector_name + '_tot_kt.fits'
            else:
                print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
                return
        elif data_type == 'cspec':
            if echan < 128:
                fitsname_data = 'cspec_' + detector_name + '_e' + str(echan) + '__kt.fits'
            elif echan == 128:
                fitsname_data = 'cspec_' + detector_name + '_tot_kt.fits'
            else:
                print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
                return
        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
            return

        user = getpass.getuser()
        fits_path_data = '/home/' + user + '/Work/Fits_data/' + str(day) + '/'
        if not os.access(fits_path_data, os.F_OK):
            os.mkdir(fits_path_data)
        fitsfilepath_data = os.path.join(fits_path_data, fitsname_data)

        data = ExternalProps.fits_data(ExternalProps(), day, detector_name, echan, data_type)
        residuals = np.array(data[0].astype(np.float))
        counts = np.array(data[1].astype(np.float))
        fit_curve = np.array(data[2].astype(np.float))
        cgb = np.array(data[3].astype(np.float))
        magnetic = np.array(data[4].astype(np.float))
        earth_ang_bin = np.array(data[5].astype(np.float))
        sun_ang_bin = np.array(data[6].astype(np.float))
        sources_fit_curve = np.array(data[7].astype(np.float))
        plot_time_bin = np.array(data[8].astype(np.float))
        plot_time_sat = np.array(data[9].astype(np.float))
        fit_coeffs = np.array(data[10].astype(np.float))
        a = fit_coeffs[0]
        b = fit_coeffs[1]
        c = fit_coeffs[2]
        d = fit_coeffs[3]
        sources_ang_bin = data[11].astype(np.float)
        sources_names = data[12]
        sources_coeff = data[13].astype(np.float)

        det = getattr(detector(), detector_name)
        year = int('20' + str(day)[0:2])

        # get the iso-date-format from the day
        date = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))

        # get the ordinal indicator for the date
        ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

        # read the SFL data
        flares = ExternalProps.flares(ExternalProps(), year)
        flares_day = np.array(flares[0]).astype(np.int)
        flares_time = np.array(flares[1])
        # if np.any(flares_day == day) == True:
        flares_today = flares_time[:, np.where(flares_day == int(day))]
        flares_today = np.squeeze(flares_today, axis=(1,)) / 3600.
        # else:
        #    flares_today = np.array([-5])
        # print flares_time.shape
        # print flares_day
        # print year
        # print day
        # print flares_day[1333]
        # print repr(day), flares_day.dtype
        # print np.where(flares_day == day)
        # print flares_day[flares_day == day]
        # print flares_day[np.where(flares_day == day)]



        #####plot-algorhythm####
        # mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

        ###plot each on the same axis as converted to counts###
        fig, ax1 = plt.subplots()

        if burst_time != 'none':
            if len(burst_time) > 2:
                for i in range(0, len(burst_time)):
                    plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
            elif len(burst_time) == 2:
                plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
            else:
                plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

        # plot vertical lines for the solar flares of the day
        if flares_today.size != 0:
            if len(flares_today[0]) > 1:
                for j in range(0, len(flares_today[0])):
                    plt.axvline(x=flares_today[0, j], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1, j], ymin=0., ymax=1., color='grey', linestyle='--')
            else:
                plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

        plot1 = ax1.step(plot_time_bin, counts, 'b-', label='Count rate')
        plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label='Fitted Background Curve')
        plot3 = ax1.plot(plot_time_sat, d * sun_ang_bin, 'y-', label='Sun effective angle')
        plot4 = ax1.plot(plot_time_sat, c * earth_ang_bin, 'c-', label='Earth occultation')
        plot5 = ax1.plot(plot_time_sat, b * magnetic, 'g-', label='McIlwain L-parameter')
        plot6 = ax1.plot(plot_time_sat, a * cgb, 'b--', label='CGB')
        plot7 = ax1.plot(plot_time_sat, sources_fit_curve, 'k-', label='Point-like sources')

        plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6 + plot7
        labels = [l.get_label() for l in plots]
        ax1.legend(plots, labels, loc=1, fontsize=20)

        ax1.grid()

        ax1.set_xlabel('Time of day in 24h', fontsize=40)
        ax1.set_ylabel('Count rate', fontsize=40)
        ax1.tick_params(axis='both', which='major', labelsize=30)

        # ax1.set_xlim([9.84, 9.85])
        if plot_time != 'none':
            ax1.set_xlim(plot_time)
        else:
            ax1.set_xlim([-0.5, 24.5])
        # if counts.all == total_rate.all:
        #    ax1.set_ylim([-500, 1500])
        # else:
        # ax1.set_ylim([-500, 750])
        ax1.set_ylim([-200, 600])

        plt.title(str(data_type) + '-fit of the ' + detector_name + '-detector on the ' + ordinal(
            int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year), fontsize=45)

        figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
        if not os.access(figure_path, os.F_OK):
            os.mkdir(figure_path)
        # if counts.all == total_rate.all:
        #    figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_tot.png'
        # else:
        figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_e' + str(echan) + '.png'

        fig = plt.gcf()  # get current figure
        fig.set_size_inches(20, 12)

        figure = os.path.join(figure_path, figure_name)
        plt.savefig(figure, bbox_inches='tight', dpi=80)

        # plt.show()

        fig.clf()
        plt.clf()

        ###plot residual noise of the fitting algorithm###

        mpl.pyplot.rcParams['agg.path.chunksize'] = 20000

        residuals = counts - fit_curve

        if burst_time != 'none':
            if len(burst_time) > 2:
                for i in range(0, len(burst_time)):
                    plt.axvline(x=burst_time[i], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
            elif len(burst_time) == 2:
                plt.axvline(x=burst_time[0], ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')
                plt.axvline(x=burst_time[1], ymin=0., ymax=1., linewidth=2, color='black', linestyle='-.')
            else:
                plt.axvline(x=burst_time, ymin=0., ymax=1., linewidth=2, color='black', linestyle='--')

        # plot vertical lines for the solar flares of the day
        if flares_today.size != 0:
            if len(flares_today[0]) > 1:
                for j in range(0, len(flares_today[0])):
                    plt.axvline(x=flares_today[0, j], ymin=0., ymax=1., linewidth=2, color='grey')
                    plt.axvline(x=flares_today[1, j], ymin=0., ymax=1., color='grey', linestyle='--')
            else:
                plt.axvline(x=flares_today[0], ymin=0., ymax=1., linewidth=2, color='grey')
                plt.axvline(x=flares_today[1], ymin=0., ymax=1., color='grey', linestyle='--')

        plt.step(plot_time_bin, residuals, 'b-')

        plt.xlabel('Time of day in 24h', fontsize=40)
        plt.ylabel('Residual noise', fontsize=40)
        plt.tick_params(axis='both', which='major', labelsize=30)

        plt.grid()

        plt.title(str(data_type) + '-fit residuals of the ' + detector_name + '-detector on the ' + ordinal(
            int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year), fontsize=45)

        if plot_time != 'none':
            plt.xlim(plot_time)
        else:
            plt.xlim([-0.5, 24.5])
        # if counts.all == total_rate.all:
        #    ax1.set_ylim([-500, 1500])
        # else:
        ax1.set_ylim([-500, 750])
        # if counts.all == total_rate.all:
        #    plt.ylim([-600, 600])
        # else:
        plt.ylim([-300, 300])

        figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
        if not os.access(figure_path, os.F_OK):
            os.mkdir(figure_path)
        # if counts.all == total_rate.all:
        #    figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_tot_res.png'
        # else:
        figure_name = str(day) + '_' + str(data_type) + '_' + detector_name + '_e' + str(echan) + '_res.png'

        fig = plt.gcf()  # get current figure
        fig.set_size_inches(20, 12)

        figure = os.path.join(figure_path, figure_name)
        plt.savefig(figure, bbox_inches='tight', dpi=80)

        # plt.show()

        fig.clf()
        plt.clf()

        return

    def sat_to_geo(self, sat_q, sat_coor):
        """This function converts the satellite coordinates into geographical coordinates depending on the quaternion-rotation of the satellite and stores the data in arrays of the form: geo_coor, geo_rad\n
        Input:\n
        calculate.sat_to_geo ( sat_q = quaternion-matrix, sat_coor = 3D-array(x, y, z) )\n
        Output:\n
        0 = geocentric coordinates[x[], y[], z[]]\n
        1 = geocentric angles[(right ascension, declination)]"""

        # calculate the rotation matrix for the satellite coordinate system as compared to the geographical coordinate system (J2000)
        nt = np.size(sat_q[0])
        scx = np.zeros((nt, 3), float)
        scx[:, 0] = (sat_q[0] ** 2 - sat_q[1] ** 2 - sat_q[2] ** 2 + sat_q[3] ** 2)
        scx[:, 1] = 2. * (sat_q[0] * sat_q[1] + sat_q[3] * sat_q[2])
        scx[:, 2] = 2. * (sat_q[0] * sat_q[2] - sat_q[3] * sat_q[1])
        scy = np.zeros((nt, 3), float)
        scy[:, 0] = 2. * (sat_q[0] * sat_q[1] - sat_q[3] * sat_q[2])
        scy[:, 1] = (-sat_q[0] ** 2 + sat_q[1] ** 2 - sat_q[2] ** 2 + sat_q[3] ** 2)
        scy[:, 2] = 2. * (sat_q[1] * sat_q[2] + sat_q[3] * sat_q[0])
        scz = np.zeros((nt, 3), float)
        scz[:, 0] = 2. * (sat_q[0] * sat_q[2] + sat_q[3] * sat_q[1])
        scz[:, 1] = 2. * (sat_q[1] * sat_q[2] - sat_q[3] * sat_q[0])
        scz[:, 2] = (-sat_q[0] ** 2 - sat_q[1] ** 2 + sat_q[2] ** 2 + sat_q[3] ** 2)

        # create geo_to_sat rotation matrix
        sat_mat = np.array([scx, scy, scz])

        # transpose into sat_to_geo rotation matrix
        geo_mat = np.transpose(sat_mat)

        # convert satellite coordinates into geocentric coordinates
        sat_coor = np.array(sat_coor)
        geo_coor = np.zeros((3, nt), float)
        geo_coor[0] = geo_mat[0, :, 0] * sat_coor[0] + geo_mat[0, :, 1] * sat_coor[1] + geo_mat[0, :, 2] * sat_coor[2]
        geo_coor[1] = geo_mat[1, :, 0] * sat_coor[0] + geo_mat[1, :, 1] * sat_coor[1] + geo_mat[1, :, 2] * sat_coor[2]
        geo_coor[2] = geo_mat[2, :, 0] * sat_coor[0] + geo_mat[2, :, 1] * sat_coor[1] + geo_mat[2, :, 2] * sat_coor[2]

        # calculate the right ascension and declination
        geo_ra = np.arctan2(-geo_coor[1], -geo_coor[0]) * 360. / (2. * math.pi) + 180.
        geo_dec = np.arctan(geo_coor[2] / (geo_coor[0] ** 2 + geo_coor[1] ** 2) ** 0.5) * 360. / (2. * math.pi)

        # put the right ascension and declination together in one array as [:,0] and [:,1]
        geo_rad = np.zeros((nt, 2), float)
        geo_rad[:, 0] = geo_ra
        geo_rad[:, 1] = geo_dec
        return geo_coor, geo_rad

    def src_occultation(self, day, ra, dec):
        """This function calculates the angle between a source and the earth and stores the data in arrays of the form: ang_occ, src_pos, src_rad\n
        Input:\n
        calculate.src_occultation( day = YYMMDD, ra, dec)\n
        Output:\n
        0 = angle between the source and the earth\n
        1 = position of the source in J2000 coordinates\n
        2 = position of the source in right ascension and declination"""
        # get the source to earth angle
        data = calculate.ang_to_earth(calculate(), day, ra, dec)
        ang_occ = data[0]
        src_pos = data[1]
        src_rad = data[2]

        # define the size of the earth
        earth_radius = 6371000.8  # geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
        atmosphere = 12000.  # the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere
        sat_dist = 6912000.
        earth_opening_angle = math.asin(r / sat_dist) * 360. / (2. * math.pi)  # earth-cone

        ang_occ[ang_occ < earth_opening_angle] = 0.
        return ang_occ, src_pos, src_rad

    def src_occultation_bin(self, day, ra, dec, bin_time_mid=0, detector_name=0, data_type='ctime'):
        """This function calculates the angle between a source and the earth and stores the data in arrays of the form: ang_occ, src_pos, src_rad\n
        Input:\n
        calculate.src_occultation_bin( day = YYMMDD, ra, dec, bin_time_mid = 0, detector = 0, data_type = 'ctime' (or 'scpec'))\n
        Output:\n
        0 = angle between the source and the earth\n
        1 = position of the source in J2000 coordinates\n
        2 = position of the source in right ascension and declination"""
        # get the source to earth angle
        data = calculate.ang_to_earth_bin(calculate(), day, ra, dec, bin_time_mid, detector_name, data_type)
        ang_occ = data[0]
        src_pos = data[1]
        src_rad = data[2]

        # define the size of the earth
        earth_radius = 6371000.8  # geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
        atmosphere = 12000.  # the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere
        sat_dist = 6912000.
        earth_opening_angle = math.asin(r / sat_dist) * 360. / (2. * math.pi)  # earth-cone

        ang_occ[ang_occ < earth_opening_angle] = 0.
        return ang_occ, src_pos, src_rad

    def sun_ang(self, detector_name, day):
        """This function calculates the sun orientation for one detector and stores the data in arrays of the form: sun_ang, sat_time\n
        Input:\n
        calculate.sun_ang ( detector, day = JJMMDD )\n
        Output:\n
        0 = angle between the sun location and the detector orientation\n
        1 = time (MET) in seconds"""

        # get the detector and satellite data
        data_det = calculate.det_or(calculate(), detector_name, day)
        det_coor = data_det[0]  # unit-vector of the detector orientation
        det_rad = data_det[1]  # detector orientation in right ascension and declination
        sat_pos = data_det[2]  # position of the satellite
        sat_time = np.array(data_det[3])  # time (MET) in seconds

        # get the sun data
        data_sun = calculate.sun_pos(calculate(), day)
        sun_pos = data_sun[0]
        sun_rad = data_sun[1]

        # calculate the angle between the sun location and the detector orientation
        scalar_product = det_coor[0] * sun_pos[0] + det_coor[1] * sun_pos[1] + det_coor[2] * sun_pos[2]
        ang_det_sun = np.arccos(scalar_product)
        sun_ang = (ang_det_sun) * 360. / (2. * math.pi)
        sun_ang = np.array(sun_ang)
        return sun_ang, sat_time, sun_rad

    def sun_ang_bin(self, detector_name, day, bin_time_mid=0, data_type='ctime'):
        """This function calculates the binned sun orientation for one detector and stores the data in arrays of the form: sun_ang_bin, sat_time_bin, bin_time_mid\n
        Input:\n
        calculate.sun_ang_bin ( detector, day = JJMMDD, bin_time_mid = 0, detector = 0, data_type = 'ctime' )\n
        Output:\n
        0 = angle between the sun location and the detector orientation\n
        1 = time (MET) in seconds\n
        2 = bin_time_mid"""

        # get the detector and satellite data
        data_det = calculate.det_or_bin(calculate(), detector_name, day, bin_time_mid, data_type)
        det_coor_bin = data_det[0]  # unit-vector of the detector orientation
        det_rad_bin = data_det[1]  # detector orientation in right ascension and declination
        sat_pos_bin = data_det[2]  # position of the satellite
        sat_time_bin = np.array(data_det[3])  # time (MET) in seconds
        bin_time_mid = data_det[4]

        # get the sun data
        data_sun = calculate.sun_pos_bin(calculate(), day, bin_time_mid, detector_name, data_type)
        sun_pos_bin = data_sun[0]
        sun_rad_bin = data_sun[1]

        # calculate the angle between the sun location and the detector orientation
        scalar_product = det_coor_bin[0] * sun_pos_bin[0] + det_coor_bin[1] * sun_pos_bin[1] + det_coor_bin[2] * \
                                                                                               sun_pos_bin[2]
        ang_det_sun = np.arccos(scalar_product)
        sun_ang_bin = (ang_det_sun) * 360. / (2. * math.pi)
        sun_ang_bin = np.array(sun_ang_bin)
        return sun_ang_bin, sat_time_bin, sun_rad_bin, bin_time_mid

    def sun_pos(self, day):
        """This function calculates the course of the sun during a certain day and stores the data in arrays of the form: sun_pos, sun_rad\n
        Input:\n
        calculate.sun_pos ( day = YYMMDD )\n
        Output:\n
        0 = unit-vector of the sun position[x[], y[], z[]]\n
        1 = geocentric angles of the sun position[(right ascension, declination)]"""

        # get the satellite data
        data = ExternalProps.poshist(ExternalProps(), day)
        sat_time = np.array(data[0])

        if sat_time[-1] <= 252460801.000:
            utc_tt_diff = 65.184
        elif sat_time[-1] <= 362793602.000:
            utc_tt_diff = 66.184
        elif sat_time[-1] <= 457401603.000:
            utc_tt_diff = 67.184
        elif sat_time[-1] <= 504921604.000:
            utc_tt_diff = 68.184
        else:
            utc_tt_diff = 69.184

        sat_time = (sat_time - utc_tt_diff) / (3600 * 24) + 36890.50074287037037037
        sat_pos = np.array(data[1])

        # calculate the geocentric angles of the sun for each time-bin
        sun = ephem.Sun()
        sun_ra = []
        sun_dec = []
        for i in range(0, len(sat_time)):
            sun.compute(sat_time[i])  # generate the sun information from the ephem module for the sat_time[i]
            sun_ra.append(sun.ra)  # add to the right ascension vector
            sun_dec.append(sun.dec)  # add to the declination vector

        # put the right ascension and declination together in one array as [:,0] and [:,1]
        sun_rad = np.zeros((len(sun_ra), 2), float)
        sun_rad[:, 0] = sun_ra
        sun_rad[:, 1] = sun_dec
        sun_rad = np.array(sun_rad)

        # derive the unit-vector of the sun location in geocentric coordinates
        sun_pos = np.array([np.cos(sun_ra) * np.cos(sun_dec), np.sin(sun_ra) * np.cos(sun_dec), np.sin(sun_dec)])
        return sun_pos, sun_rad

    def sun_pos_bin(self, day, bin_time_mid=0, detector_name=0, data_type='ctime'):
        """This function calculates the course of the sun during a certain day and stores the data in arrays of the form: sun_pos_bin, sun_rad_bin, bin_time_mid\n
        Input:\n
        calculate.sun_pos ( day = YYMMDD, bin_time_mid = 0, detector = 0, data_type = 'ctime' )\n
        Output:\n
        0 = unit-vector of the sun position[x[], y[], z[]]\n
        1 = geocentric angles of the sun position[(right ascension, declination)]
        2 = bin_time_mid"""

        # get the satellite data
        data = ExternalProps.poshist_bin(ExternalProps(), day, bin_time_mid, detector_name, data_type)
        sat_time_bin = np.array(data[0])

        if sat_time_bin[-1] <= 252460801.000:
            utc_tt_diff = 65.184
        elif sat_time_bin[-1] <= 362793602.000:
            utc_tt_diff = 66.184
        elif sat_time_bin[-1] <= 457401603.000:
            utc_tt_diff = 67.184
        elif sat_time_bin[-1] <= 504921604.000:
            utc_tt_diff = 68.184
        else:
            utc_tt_diff = 69.184

        sat_time_bin = (sat_time_bin - utc_tt_diff) / (3600 * 24) + 36890.50074287037037037
        sat_pos_bin = np.array(data[1])
        bin_time_mid = data[5]

        # calculate the geocentric angles of the sun for each time-bin
        sun = ephem.Sun()
        sun_ra_bin = []
        sun_dec_bin = []
        for i in range(0, len(sat_time_bin)):
            sun.compute(sat_time_bin[i])  # generate the sun information from the ephem module for the sat_time[i]
            sun_ra_bin.append(sun.ra)  # add to the right ascension vector
            sun_dec_bin.append(sun.dec)  # add to the declination vector

        # put the right ascension and declination together in one array as [:,0] and [:,1]
        sun_rad_bin = np.zeros((len(sun_ra_bin), 2), float)
        sun_rad_bin[:, 0] = sun_ra_bin
        sun_rad_bin[:, 1] = sun_dec_bin
        sun_rad_bin = np.array(sun_rad_bin)

        # derive the unit-vector of the sun location in geocentric coordinates
        sun_pos_bin = np.array(
            [np.cos(sun_ra_bin) * np.cos(sun_dec_bin), np.sin(sun_ra_bin) * np.cos(sun_dec_bin), np.sin(sun_dec_bin)])
        return sun_pos_bin, sun_rad_bin, bin_time_mid

        def rigidity(self, day, bin_time_mid=0):
            """This function interpolates the rigidity of the lookup table from Humble et al., 1979, evaluates the values for the satellite position on a given day and stores the data in an array of the form: rigidity\n
            Input:\n
            calculate.rigidity ( day = JJMMDD )\n
            Output:\n
            0 = rigidity\n
            1 = sat_lon_bin\n
            2 = sat_lat_bin\n
            3 = sat_time_bin"""

            # data from the paper mentioned in the docstring. Rigidity at 400 km altitude
            lon = np.arange(0., 360.001, 30.)
            lat = np.arange(40., -40.001, -10.)
            rigidity_matrix = [[6.1, 6.94, 7.66, 8.47, 9.08, 9.17, 7.44, 5., 3.26, 2.08, 2.44, 4.43, 6.1],
                               [10.54, 11.07, 11.87, 12.73, 12.73, 11.99, 10.75, 9.08, 6.28, 4.39, 4.80, 8.81, 10.54],
                               [12.48, 13.25, 14.2, 14.93, 14.49, 13.65, 12.48, 11.52, 10.13, 7.01, 8.9, 11.3, 12.48],
                               [12.99, 13.92, 14.93, 15.53, 15.23, 14.34, 13.51, 12.86, 11.87, 10.75, 10.96, 12.23,
                                12.99],
                               [12.36, 13.25, 14.2, 14.93, 14.78, 14.2, 13.92, 13.38, 12.73, 11.87, 11.64, 12.11,
                                12.36],
                               [10.96, 11.41, 12.36, 12.99, 13.12, 13.12, 13.12, 13.12, 12.73, 11.99, 11.52, 11.07,
                                10.96],
                               [8.72, 8.81, 9.35, 9.63, 9.83, 10.33, 11.3, 11.99, 11.99, 11.52, 10.86, 9.93, 8.72],
                               [6.34, 5.8, 5.2, 4.75, 4.85, 5.41, 7.01, 9.35, 10.64, 10.54, 9.83, 8.3, 6.34],
                               [3.97, 3.42, 2.72, 2.06, 2.02, 2.42, 3.71, 5.2, 7.97, 9.08, 8.55, 6.1, 3.97]]

            # interpolation
            tck = interpolate.interp2d(lon, lat, rigidity_matrix, kind='cubic')

            # get the satellite data
            sat_data = ExternalProps.poshist_bin(ExternalProps(), day, bin_time_mid)
            sat_lon = sat_data[3]
            sat_lat = sat_data[2]
            sat_time = sat_data[0]

            # evaluate the rigidity -> for-loop is slow, but otherwise the tck-function doesn't seem to accept the sat_lon and sat_lat arrays
            rigidity = []
            for i in range(0, len(sat_lon)):
                rig = tck(sat_lon[i], sat_lat[i])
                rigidity.append(rig)
            rigidity = np.array(rigidity)

            return rigidity, sat_lon, sat_lat, sat_time



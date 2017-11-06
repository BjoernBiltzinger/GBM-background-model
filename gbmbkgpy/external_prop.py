#!/usr/bin python2.7

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from datetime import date
from datetime import datetime
import ephem
import fileinput
import getpass
import math
import matplotlib.pyplot                   as plt
import matplotlib                          as mpl
import numpy                               as np
from numpy import linalg        as LA
import os
import pyfits
import scipy.optimize                      as optimization
from scipy import integrate
from scipy import interpolate
from scipy.optimize import curve_fit
import subprocess
from subprocess import Popen, PIPE
import urllib2

"""
This class contains all functions for reading the files needed for the GBM background model:\n
earth_occ(self) -> earth_ang, angle_d, area_frac, free_area, occ_area\n
flares(self, year) -> day, time\n
lat_spacecraft (self, week) -> lat_time, mc_b, mc_l\n
magfits (self, day) -> t_magn, h_magn, x_magn, y_magn, z_magn\n
mcilwain (self, day) -> sat_time, mc_b, mc_l\n
saa(self) -> saa\n\n\n
"""


class ExternalProps(object):
    def __init__(self, day):
        self._day = day

        self._read_saa()
        self._earth_occ()
        self._fits_data()
        self._flares()
        self._magfits()
        self._mcilwain()
        self._point_sources()

    def _read_saa(self):
        """
        This function reads the saa.dat file and returns the polygon in the form: saa[lat][lon]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = saa[latitude][longitude]\n
        """

        # TODO: path symbols on different OS's can change. Use os.path.join(<path1,path2>)

        user = getpass.getuser()
        saa_path = os.path.join('/home', user, 'Work', 'saa')
        filepath = os.path.join(saa_path, 'saa.dat')

        # context managers  allow for quick handling of files open/close
        with open(filepath, 'r') as poly:
            lines = poly.readlines()

        saa_lat = []
        saa_lon = []  # define latitude and longitude arrays
        # write file data into the arrays
        for line in lines:
            p = line.split()
            saa_lat.append(float(p[0]))
            saa_lon.append(float(p[1]))  # (float(p[1]) + 360.)%360)
        saa = np.array([saa_lat, saa_lon])  # merge the arrays

        self._saa_properties = saa

        @property
        def saa(self):
            return self._saa_properties

    def _earth_occ(self):
        """This function reads the earth occultation fits file and stores the data in arrays of the form: earth_ang, angle_d, area_frac, free_area, occ_area.\n
        Input:\n
        readfile.earth_occ ( )\n
        Output:\n
        0 = angle between detector direction and the earth in 0.5 degree intervals\n
        1 = opening angles of the detector (matrix)\n
        2 = fraction of the occulted area to the FOV area of the detector (matrix)\n
        3 = FOV area of the detector (matrix)\n
        4 = occulted area (matrix)"""

        # read the file
        user = getpass.getuser()
        fits_path = os.path.join('/home/', user, '/Work/earth_occultation/')
        fitsname = 'earth_occ_calc_total_kt.fits'
        fitsfilepath = os.path.join(fits_path, fitsname)

        e_occ_fits = fits.open(fitsfilepath)
        angle_d = []
        area_frac = []
        free_area = []
        occ_area = []
        earth_occ_dic = {}
        for i in range(1, len(e_occ_fits)):
            data = e_occ_fits[i].data
            angle_d.append(data.angle_d)
            area_frac.append(data.area_frac)
            free_area.append(data.free_area)
            occ_area.append(data.occ_area)
        e_occ_fits.close()

        # Store Values in diccionary
        earth_occ_dic['angle_d'] = np.array(angle_d, dtype='f8')
        earth_occ_dic['area_frac'] = np.array(area_frac, dtype='f8')
        earth_occ_dic['free_area'] = np.array(free_area, dtype='f8')
        earth_occ_dic['occ_area'] = np.array(occ_area, dtype='f8')
        earth_occ_dic['earth_ang'] = np.arange(0, 180.5, .5)

        self._earth_occ_properties = earth_occ_dic

        @property
        def earth_occ(self):
            return self._earth_occ_properties

    def _fits_data(self, day, detector_name, echan, data_type='ctime', sources_number='None'):
        """This function reads a Fits-data file and stores the data in arrays of the form: residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, crab_ang_bin, plot_time_bin, plot_time_sat.\n
        Input:\n
        readfile.fits_data ( day = YYMMDD, detector_name, echan, data_type = 'ctime' (or 'cspec'), sources_number = 'None' (takes number from current sources_file or input another number) )\n
        Output:\n
        0 = residuals\n
        1 = counts\n
        2 = fit_curve\n
        3 = cgb\n
        4 = magnetic\n
        5 = earth_ang_bin\n
        6 = sun_ang_bin\n
        7 = sources_fit_curve\n
        8 = plot_time_bin\n
        9 = plot_time_sat\n
        10 = fit_coeffs\n
        11 = sources_ang_bin\n
        12 = sources_names\n
        13 = sources_coeff"""

        fits_data_dic = {}
        sources_data = self._point_sources()  # old: point_sources(readfile()) Not sure what the readfile() stands for!!!!
        sources_names = sources_data[0]
        # sources_coordinates = sources_data[1]
        if sources_number == 'None':
            sources_number = len(sources_names)

        # read the file
        if data_type == 'ctime':
            if echan < 8:
                fitsname = os.path.join('ctime_', detector_name, '_e', str(echan), '_kt.fits')
            elif echan == 8:
                fitsname = os.path.join('ctime_', detector_name, '_tot_kt.fits')
            else:
                print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
                return
        elif data_type == 'cspec':
            if echan < 128:
                fitsname = os.path.join('cspec_', detector_name, '_e', str(echan), '__kt.fits')
            elif echan == 128:
                fitsname = os.path.join('cspec_', detector_name, '_tot_kt.fits')
            else:
                print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
                return
        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
            return

        user = getpass.getuser()
        fits_path = os.path.join('/home/', user, '/Work/Fits_data/', str(day), '/')
        if not os.access(fits_path, os.F_OK):
            os.mkdir(fits_path)
        fitsfilepath = os.path.join(fits_path, fitsname)

        plot_fits = fits.open(fitsfilepath)
        fit_data = plot_fits[1].data
        pointsources_data = plot_fits[2].data
        plot_fits.close()

        # extract the data
        fits_data_dic['residuals'] = fit_data.Residuals
        fits_data_dic['counts'] = fit_data.Count_Rate
        fits_data_dic['fit_curve'] = fit_data.Fitting_curve
        fits_data_dic['cgb'] = fit_data.CGB_curve
        fits_data_dic['magnetic'] = fit_data.Mcilwain_L_curve
        fits_data_dic['earth_ang_bin'] = fit_data.Earth_curve
        fits_data_dic['sun_ang_bin'] = fit_data.Sun_curve
        fits_data_dic['sources_fit_curve'] = fit_data.Pointsources_curve
        # fits_data_dic['crab_ang_bin'] = fit_data.Crab_curve
        # fits_data_dic['scox_ang_bin'] = fit_data.Scox_curve
        # fits_data_dic['cygx_ang_bin'] = fit_data.Cygx_curve
        fits_data_dic['plot_time_bin'] = fit_data.Data_time
        fits_data_dic['plot_time_sat'] = fit_data.Parameter_time
        fits_data_dic['fit_coeffs'] = fit_data.FitCoefficients

        # sources_ang_bin = []
        fits_data_dic['sources_names'] = sources_names.tolist()
        # for i in range(0, len(plot_time_sat)):
        # for i in range(0, sources_number):
        # src_array = pointsources_data[sources_names[i]]
        #    src_array = pointsources_data[i]
        #    sources_ang_bin.append(src_array)
        # sources_ang_bin = np.array(sources_ang_bin)
        sources_ang_bin = np.array(pointsources_data)
        fits_data_dic['sources_ang_bin'] = \
            sources_ang_bin.view(sources_ang_bin.dtype[0]).reshape(sources_ang_bin.shape + (-1,)).astype(np.float).T

        # sources_ang_bin = np.swapaxes(sources_ang_bin,0,1)

        fits_data_dic['sources_coeff'] = pointsources_data.FitCoefficients_Pointsources

        # old return:  residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin,
        #             sources_fit_curve, plot_time_bin, plot_time_sat, fit_coeffs, sources_ang_bin, sources_names, sources_coeff

        self._fits_data_properties = fits_data_dic

        @property
        def fits_data(self):
            return self._fits_data_properties

    def _flares(self, year):
        """This function reads the YYYY.txt file containing the GOES solar flares of the corresponding year and returns the data in arrays of the form: day, time\n
        Input:\n
        year = YYYY\n
        Output\n
        0 = day ('YYMMDD')
        1 = time[start][stop] (in seconds on that day -> accuracy ~ 1 minute)\n"""
        filename = os.path.join(str(year), '.dat')
        user = getpass.getuser()
        fits_path = os.path.join('/home/', user, '/Work/flares/')
        filepath = os.path.join(fits_path, str(filename))
        while os.path.isfile(filepath) == False:
            download.flares(download(), year)

        with open(filepath, 'r') as flares:
            lines = flares.readlines()

        day = []  # define day, start & stop arrays
        start = []
        stop = []
        flares_dic = {}
        for line in lines:  # write file data into the arrays
            p = line.split()
            # print p[0]
            day.append(int(p[0][5:]))
            start.append(int(p[1][0:2]) * 3600. + int(p[1][2:4]) * 60.)
            stop.append(int(p[2][0:2]) * 3600. + int(p[2][2:4]) * 60.)

        # create numpy arrays
        flares_dic['day'] = np.array(day)  # array of days when solar flares occured
        start = np.array(start)
        stop = np.array(stop)
        flares_dic['time'] = np.array(
            [start, stop])  # combine the start and stop times of the solar flares into one matrix

        self._flares_properties = flares_dic

        @property
        def flares(self):
            return self._flares_properties

    def lat_spacecraft(self, week):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file
        filename = os.path.join('lat_spacecraft_weekly_w', str(week), '_p202_v001.fits')
        user = getpass.getuser()
        fits_path = os.path.join('/home/', user, '/Work/lat/')
        filepath = os.path.join(fits_path, str(filename))
        while os.path.isfile(filepath) == False:
            download.lat_spacecraft(download(), week)

        lat_dic = {}
        lat_fits = fits.open(filepath)
        data = lat_fits[1].data
        lat_fits.close()

        # extract the data
        lat_dic[
            'lat_time'] = data.START  # Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
        lat_dic['mc_b'] = data.B_MCILWAIN  # Position in J2000 equatorial coordinates
        lat_dic['mc_l'] = data.L_MCILWAIN

        self._lat_properties = lat_dic

        @property
        def lat_spacecraft(self):
            return self._lat_properties

    def _magfits(self, day):
        """This function reads a magnetic field fits file and stores the data in arrays of the form: t_magn, h_magn, x_magn, y_magn, z_magn.\n
        Input:\n
        readfile.magfits ( day = YYMMDD )\n
        Output:\n
        0 = t_magn\n
        1 = h_magn\n
        2 = x_magn\n
        3 = y_magn\n
        4 = z_magn"""

        # read the file
        fitsname = os.path.join('magn_', str(day), '_kt.fits')
        user = getpass.getuser()
        path = os.path.join('/home/', user, '/Work/magnetic_field/', str(day), '/')
        filepath = os.path.join(path, str(fitsname))
        mag_fits = fits.open(filepath)
        data = mag_fits[1].data
        mag_fits.close()
        magfits_dic = {}

        # extract the data
        altitude = data.Altitude  # altitude of the satellite above the WGS 84 ellipsoid
        magfits_dic['t_magn'] = data.F_nT  # total intensity of the geomagnetic field
        magfits_dic['h_magn'] = data.H_nT  # horizontal intensity of the geomagnetic field
        magfits_dic['x_magn'] = data.X_nT  # north component of the geomagnetic field
        magfits_dic['y_magn'] = data.Y_nT  # east component of the geomagnetic field
        magfits_dic['z_magn'] = data.Z_nT  # vertical component of the geomagnetic field

        self._magfits_properties = magfits_dic

        @property
        def magfits(self):
            return self._magfits_properties

    def _mcilwain(self, day):
        """This function reads a mcilwain file and stores the data in arrays of the form: sat_time, mc_b, mc_l.\n
        Input:\n
        readfile.mcilwain ( day = YYMMDD )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file
        filename = os.path.join('glg_mcilwain_all_', str(day), '_kt.fits')
        user = getpass.getuser()
        fits_path = os.path.join('/home/', user, '/Work/mcilwain/')
        filepath = os.path.join(fits_path, str(filename))
        while os.path.isfile(filepath) == False:
            begin_date = date(2008, 8, 7)  # first complete lat_spacecraft weekly-file
            year = int('20' + str(day)[:2])
            month = int(str(day)[2:4])
            this_day = int(str(day)[4:])
            this_date = date(year, month, this_day)
            delta = this_date - begin_date
            days_diff = delta.days
            week_diff = int(days_diff / 7)
            week = 10 + week_diff  # the first complete file is saved as week 010.
            if week < 10:
                week = '00' + str(week)
            elif week < 100:
                week = '0' + str(week)
            else:
                week = str(week)
            writefile.mcilwain_fits(writefile(), week, day)

        mc_fits = fits.open(filepath)
        data = mc_fits[1].data
        mc_fits.close()
        mcilwain_dic = {}

        # extract the data
        mcilwain_dic[
            'sat_time'] = data.col1  # Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
        mcilwain_dic['mc_b'] = data.col2  # Position in J2000 equatorial coordinates
        mcilwain_dic['mc_l'] = data.col3

        self._mcilwain_properties = mcilwain_dic

        @property
        def mcilwain(self):
            return self._mcilwain_properties

    def _point_sources(self):
        """This function reads the point_sources.dat file and returns the sources in the form: names, coordinates[ra][dec]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = source_names
        1 = coordinates\n"""
        user = getpass.getuser()
        saa_path = os.path.join('/home/', user, '/Work/background_point_sources/')
        filepath = os.path.join(saa_path, 'point_sources.dat')
        with open(filepath, 'r') as poly:
            lines = poly.readlines()
        source_names = []
        source_ra = []
        source_dec = []
        point_sources_dic = {}
        # write file data into the arrays
        for line in lines:
            p = line.split()
            source_names.append(p[0])
            source_ra.append(float(p[1]))
            source_dec.append(float(p[2]))
        point_sources_dic['source_names'] = np.array(source_names)
        point_sources_dic['coordinates'] = np.array([source_ra, source_dec])  # merge the arrays

        self._point_sources_properties = point_sources_dic

        @property
        def point_sources(self):
            return self._point_sources_properties


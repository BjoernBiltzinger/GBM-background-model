import os
import pandas as pd
import subprocess
import urllib2
from datetime import date
from subprocess import Popen

from astropy.coordinates import SkyCoord


import numpy as np
from astropy.io import fits

from gbmbkgpy.io.file_utils import file_existing_and_readable

from gbmbkgpy.point_source import PointSource


from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file

# TODO: Add fits_data and earth_occulation to data folder or different filepath and adjust path-handling

class ExternalProps(object):




    def __init__(self, day, data_in = 'data_in'):
        """
        Build the external properties for a given day
        :param day: YYMMDD
        """

        assert isinstance(day,str), 'Day must be a string'
        assert len(day) == 6, 'Day must be in format YYMMDD'


        self._day = day
        self._year = '20%s'%day[:2]
        self._data_in = data_in

        self._build_flares()

        self._read_saa()

        self._build_point_sources()

        # self._point_sources()

        # self._earth_occ()
        # self._fits_data()

        # self._magfits()
        # self._mcilwain()


    def _read_saa(self):
        """
        This function reads the saa.dat file and returns the polygon in the form: saa[lat][lon]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = saa[latitude][longitude]\n
        """

        filepath = get_path_of_data_file('saa', 'saa.dat')

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
        fitsname = 'earth_occ_calc_total_kt.fits'
        fitsfilepath = get_path_of_data_file('earth_occulation', fitsname)

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


        data_path = get_path_of_data_dir()
        fits_path = os.path.join(data_path, 'Fits_data/', str(day), '/')
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

    def _build_flares(self):
        """This function reads the YYYY.txt file containing the GOES solar flares of the corresponding year and returns the data in arrays of the form: day, time\n
        Input:\n
        year = YYYY\n
        Output\n
        0 = day ('YYMMDD')
        1 = time[start][stop] (in seconds on that day -> accuracy ~ 1 minute)\n"""
        filename =  '%s.dat' % self._year
        filepath = get_path_of_data_file('flares', str(filename))


        #while os.path.isfile(filepath) == False:
        if not file_existing_and_readable(filepath):

            _download_flares(self._year)

        with open(filepath, 'r') as flares:
            lines = flares.readlines()

        day = []  # define day, start & stop arrays
        start = []
        stop = []
        flares_dic = {}
        for line in lines:  # write file data into the arrays
            p = line.split()
            # print p[0]

            day.append(p[0][5:])
            start.append(int(p[1][0:2]) * 3600. + int(p[1][2:4]) * 60.)
            stop.append(int(p[2][0:2]) * 3600. + int(p[2][2:4]) * 60.)

        # create numpy arrays
        flares_dic['day'] = np.array(map(str,day))  # array of days when solar flares occured
        start = np.array(start)
        stop = np.array(stop)
        flares_dic['tstart'] = np.array(start)
        flares_dic['tstop'] = np.array(stop)


        self._flares_properties = pd.DataFrame(flares_dic)

        self._flare_idx = self._flares_properties['day'] == self._day



    @property
    def flares(self):
        return self._flares_properties[self._flare_idx]

    def _lat_spacecraft(self, week):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file
        filename = os.path.join('lat_spacecraft_weekly_w', str(week), '_p202_v001.fits')
        filepath = get_path_of_data_file('lat', str(filename))

        #while os.path.isfile(filepath) == False:
        if not file_existing_and_readable(filepath):
            _download_lat_spacecraft(week)

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
        path = os.path.join(get_path_of_data_dir(), 'magnetic_field/', str(day), '/')
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

    # def _mcilwain(self, day):
    #     """This function reads a mcilwain file and stores the data in arrays of the form: sat_time, mc_b, mc_l.\n
    #     Input:\n
    #     readfile.mcilwain ( day = YYMMDD )\n
    #     Output:\n
    #     0 = time\n
    #     1 = mcilwain parameter B\n
    #     2 = mcilwain parameter L"""
    #
    #     # read the file
    #     filename = os.path.join('glg_mcilwain_all_', str(day), '_kt.fits')
    #     filepath = get_path_of_data_file('mcilwain/', str(filename))
    #     while os.path.isfile(filepath) == False:
    #         begin_date = date(2008, 8, 7)  # first complete lat_spacecraft weekly-file
    #         year = int('20' + str(day)[:2])
    #         month = int(str(day)[2:4])
    #         this_day = int(str(day)[4:])
    #         this_date = date(year, month, this_day)
    #         delta = this_date - begin_date
    #         days_diff = delta.days
    #         week_diff = int(days_diff / 7)
    #         week = 10 + week_diff  # the first complete file is saved as week 010.
    #         if week < 10:
    #             week = '00' + str(week)
    #         elif week < 100:
    #             week = '0' + str(week)
    #         else:
    #             week = str(week)
    #         writefile.mcilwain_fits(writefile(), week, day)
    #
    #     mc_fits = fits.open(filepath)
    #     data = mc_fits[1].data
    #     mc_fits.close()
    #     mcilwain_dic = {}
    #
    #     # extract the data
    #     mcilwain_dic[
    #         'sat_time'] = data.col1  # Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
    #     mcilwain_dic['mc_b'] = data.col2  # Position in J2000 equatorial coordinates
    #     mcilwain_dic['mc_l'] = data.col3
    #
    #     self._mcilwain_properties = mcilwain_dic

    @property
    def mcilwain(self):
        return self._mcilwain_properties

    def _build_point_sources(self):
        """This function reads the point_sources.dat file and returns the sources in the form: names, coordinates[ra][dec]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = source_names
        1 = coordinates\n"""
        file_path = get_path_of_data_file('background_point_sources/', 'point_sources.dat')

        self._ps_df = pd.read_table(file_path,names=['name','ra','dec'])

        # instantiate dic of point source objects
        self._point_sources_dic = {}
        for row in self._ps_df.itertuples():
            self._point_sources_dic[row[1]] = PointSource(row[1], row[2], row[3], self._data_in)


        # with open(filepath, 'r') as poly:
        #     lines = poly.readlines()
        # source_names = []
        # source_ra = []
        # source_dec = []
        # point_sources_dic = {}
        # # write file data into the arrays
        # for line in lines:
        #     p = line.split()
        #     source_names.append(p[0])
        #     source_ra.append(float(p[1]))
        #     source_dec.append(float(p[2]))
        # point_sources_dic['source_names'] = np.array(source_names)
        # point_sources_dic['coordinates'] = np.array([source_ra, source_dec])  # merge the arrays

        #self._point_sources_properties = point_sources_dic

    @property
    def point_sources(self):
        return self._point_sources_dic



def _download_flares(year):
    """This function downloads a yearly solar flar data file and stores it in the appropriate folder\n
    Input:\n
    download.flares ( year = YYYY )\n"""

    # create the appropriate folder if it doesn't already exist and switch to it
    file_path = os.path.join(get_path_of_data_dir(), 'flares/')
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    os.chdir(file_path)
    if year == 2016:
        url = (
        'ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_' + str(
            year) + 'ytd.txt')
    else:
        url = (
        'ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_' + str(
            year) + '.txt')
    file_name = '%s.dat' % str(year)

    u = urllib2.urlopen(url)


    with open(file_name, 'wb') as f:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print status,



    return

def _download_lat_spacecraft(week):
    """This function downloads a weekly lat-data file and stores it in the appropriate folder\n
    Input:\n
    download.lat_spacecraft ( week = XXX )\n"""

    # create the appropriate folder if it doesn't already exist and switch to it
    file_path = os.path.join(get_path_of_data_dir(), 'lat/')
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    os.chdir(file_path)

    url = ('http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w' + str(
        week) + '_p202_v001.fits')
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        print status,

    f.close()

    return
#
#
# class writefile(object):
#     """This class contains all functions for writing files needed for the GBM background model:\n
#     coord_file(self, day) -> filepaths, directory\n
#     magn_file(self, day) -> out_paths\n
#     magn_fits_file(self, day) -> fitsfilepath\n
#     mcilwain_fits(self, week, day) -> fitsfilepath\n\n\n"""
#
#     def coord_file(self, day):
#         """This function writes four coordinate files of the satellite for one day and returns the following information about the files: filepaths, directory\n
#         Input:\n
#         writefile.coord_file ( day = JJMMDD )\n
#         Output:\n
#         0 = filepaths\n
#         1 = directory"""
#
#         poshist = rf.poshist(day)
#         sat_time = poshist[0]
#         sat_pos = poshist[1]
#         sat_lat = poshist[2]
#         sat_lon = poshist[3] - 180.
#
#         geometrie = calc_altitude(day)
#         altitude = geometrie[0]
#         earth_radius = geometrie[1]
#
#         decimal_year = calc.met_to_date(sat_time)[4]
#
#         user = getpass.getuser()
#         directory = os.path.join(get_path_of_data_dir(), 'magnetic_field/' + str(day))
#         fits_path = os.path.join(os.path.dirname(__dir__), directory)
#
#         filename1 = 'magn_coor_' + str(day) + '_kt_01.txt'
#         filename2 = 'magn_coor_' + str(day) + '_kt_02.txt'
#         filename3 = 'magn_coor_' + str(day) + '_kt_03.txt'
#         filename4 = 'magn_coor_' + str(day) + '_kt_04.txt'
#
#         filepath1 = os.path.join(fits_path, str(filename1))
#         filepath2 = os.path.join(fits_path, str(filename2))
#         filepath3 = os.path.join(fits_path, str(filename3))
#         filepath4 = os.path.join(fits_path, str(filename4))
#         filepaths = [filepath1, filepath2, filepath3, filepath4]
#
#         if not os.path.exists(fits_path):
#             try:
#                 os.makedirs(fits_path)
#             except OSError as exc:  # Guard against race condition
#                 if exc.errno != errno.EEXIST:
#                     raise
#
#         emm_file = open(filepath1, 'w')
#         for i in range(0, len(sat_time) / 4):
#             emm_file.write(
#                 str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
#         emm_file.close()
#
#         emm_file = open(filepath2, 'w')
#         for i in range(len(sat_time) / 4, len(sat_time) / 2):
#             emm_file.write(
#                 str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
#         emm_file.close()
#
#         emm_file = open(filepath3, 'w')
#         for i in range(len(sat_time) / 2, len(sat_time) * 3 / 4):
#             emm_file.write(
#                 str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
#         emm_file.close()
#
#         emm_file = open(filepath4, 'w')
#         for i in range(len(sat_time) * 3 / 4, len(sat_time)):
#             emm_file.write(
#                 str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
#         emm_file.close()
#
#         return filepaths, directory
#     #
#     # def fits_data(self, day, detector_name, echan, data_type, residuals, counts, fit_curve, cgb, magnetic,
#     #               earth_ang_bin, sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat, coeff_cgb, coeff_magn,
#     #               coeff_earth, coeff_sun, sources_ang_bin, sources_names, sources_coeff, saa_coeffs):
#     #     """This function writes Fits_data files for the fits of a specific detector and energy channel on a given day.\n
#     #     Input:\n
#     #     writefile.fits_data(day, detector_name, echan, data_type, residuals, counts, fit_curve, cgb, magnetic, earth_ang_bin, sun_ang_bin, sources_fit_curve, plot_time_bin, plot_time_sat, coeff_cgb, coeff_magn, coeff_earth, coeff_sun, sources_ang_bin, sources_names, sources_coeff)\n
#     #     Output:\n
#     #     None"""
#     #
#     #     sources_number = len(sources_names)
#     #
#     #     ##### Pointsources testing #####
#     #     # if data_type == 'ctime':
#     #     #    if echan < 9:
#     #     #        fitsname = '_test_s' + str(sources_number) + '_ctime_' + detector_name + '_e' + str(echan) + '_kt.fits'
#     #     #    elif echan == 9:
#     #     #        fitsname = '_test_s' + str(sources_number) + '_ctime_' + detector_name + '_tot_kt.fits'
#     #     #    else:
#     #     #        print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
#     #     #        return
#     #     # elif data_type == 'cspec':
#     #     #    if echan < 129:
#     #     #        fitsname = '_test_s' + str(sources_number) + '_cspec_' + detector_name + '_e' + str(echan) + '__kt.fits'
#     #     #    elif echan == 129:
#     #     #        fitsname = '_test_s' + str(sources_number) + '_cspec_' + detector_name + '_tot_kt.fits'
#     #     #    else:
#     #     #        print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
#     #     #        return
#     #     # else:
#     #     #    print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
#     #     #    return
#     #
#     #     if data_type == 'ctime':
#     #         if echan < 9:
#     #             fitsname = 'ctime_' + detector_name + '_e' + str(echan) + '_kt.fits'
#     #         elif echan == 9:
#     #             fitsname = 'ctime_' + detector_name + '_tot_kt.fits'
#     #         else:
#     #             print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
#     #             return
#     #     elif data_type == 'cspec':
#     #         if echan < 129:
#     #             fitsname = 'cspec_' + detector_name + '_e' + str(echan) + '__kt.fits'
#     #         elif echan == 129:
#     #             fitsname = 'cspec_' + detector_name + '_tot_kt.fits'
#     #         else:
#     #             print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
#     #             return
#     #     else:
#     #         print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
#     #         return
#     #
#     #     fits_path = os.path.join(get_path_of_data_dir(), 'Fits_data/' + str(day) + '/')
#     #     if not os.access(fits_path, os.F_OK):
#     #         os.mkdir(fits_path)
#     #     fitsfilepath = os.path.join(fits_path, fitsname)
#     #
#     #     prihdu = fits.PrimaryHDU()
#     #     hdulist = [prihdu]
#     #
#     #     col1 = fits.Column(name='Residuals', format='E', array=residuals, unit='counts/s')
#     #     col2 = fits.Column(name='Count_Rate', format='E', array=counts, unit='counts/s')
#     #     col3 = fits.Column(name='Fitting_curve', format='E', array=fit_curve, unit='counts/s')
#     #     col4 = fits.Column(name='CGB_curve', format='E', array=cgb)
#     #     col5 = fits.Column(name='Mcilwain_L_curve', format='E', array=magnetic)
#     #     col6 = fits.Column(name='Earth_curve', format='E', array=earth_ang_bin)
#     #     col7 = fits.Column(name='Sun_curve', format='E', array=sun_ang_bin)
#     #     col8 = fits.Column(name='Pointsources_curve', format='E', array=sources_fit_curve)
#     #     col9 = fits.Column(name='Data_time', format='E', array=plot_time_bin, unit='24h')
#     #     col10 = fits.Column(name='Parameter_time', format='E', array=plot_time_sat, unit='24h')
#     #     col11 = fits.Column(name='FitCoefficients', format='E', array=[coeff_cgb, coeff_magn, coeff_earth, coeff_sun])
#     #     cols1 = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11])
#     #
#     #     hdulist.append(fits.TableHDU.from_columns(cols1, name='Data'))
#     #
#     #     sources_cols = []
#     #     for i in range(0, len(sources_names)):
#     #         col = fits.Column(name=sources_names[i], format='E', array=sources_ang_bin[i])
#     #         sources_cols.append(col)
#     #
#     #     col12 = fits.Column(name='FitCoefficients_Pointsources', format='E', array=sources_coeff)
#     #     sources_cols.append(col12)
#     #     cols2 = fits.ColDefs(sources_cols)
#     #     hdulist.append(fits.TableHDU.from_columns(cols2, name='Pointsources'))
#     #
#     #     col13 = fits.Column(name='Prefactor_coefficient', format='E', array=saa_coeffs[0])
#     #     col14 = fits.Column(name='Exponent_coefficient', format='E', array=saa_coeffs[1])
#     #     cols3 = fits.ColDefs([col13, col14])
#     #     hdulist.append(fits.TableHDU.from_columns(cols3, name='SAA_Exit_Coefficients'))
#     #
#     #     thdulist = fits.HDUList(hdulist)
#     #     thdulist.writeto(fitsfilepath)
#     #
#     #     return fitsfilepath
#     # #
#     # # def magn_file(self, day):
#     # #     """This function calls the c-programme of the EMM-2015 magnetic field model to calculate and write the magnetic field data for the four given coordinate files for one day and returns the paths of the magnetic field files: out_paths\n
#     # #     Input:\n
#     # #     writefile.magn_file ( day = JJMMDD )\n
#     # #     Output:\n
#     # #     0 = out_paths"""
#     # #
#     # #     coord_files = write_coord_file(day)
#     # #     filepaths = coord_files[0]
#     # #     directory = coord_files[1]
#     # #
#     # #     user = getpass.getuser()
#     # #     fits_path_emm = '/home/' + user + '/Work/EMM2015_linux/'
#     # #     emm_file = os.path.join(fits_path_emm, 'emm_sph_fil')
#     # #
#     # #     out_paths = []
#     # #     for i in range(0, len(filepaths)):
#     # #         __dir__ = os.path.dirname(os.path.abspath(__file__))
#     # #         path = os.path.join(os.path.dirname(__dir__), directory)
#     # #         out_name = 'magn_' + str(day) + '_kt_0' + str(i + 1) + '.txt'
#     # #         out_file = os.path.join(path, out_name)
#     # #         out_paths.append(out_file)
#     # #
#     # #     for i in range(0, len(filepaths)):
#     # #         cmd = str(emm_file) + ' f ' + str(filepaths[i]) + ' ' + str(out_paths[i])
#     # #         result = Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=fits_path_emm)
#     # #     return out_paths
#     # #
#     # # def magn_fits_file(self, day):
#     # #     """This function reads the magnetic field files of one day, writes the data into a fit file and returns the filepath: fitsfilepath\n
#     # #     Input:\n
#     # #     writefile.magn_fits_file ( day = JJMMDD )\n
#     # #     Output:\n
#     # #     0 = fitsfilepath"""
#     # #
#     # #     user = getpass.getuser()
#     # #     directory = os.path.join(get_path_of_data_dir(),'magnetic_field/' + str(day))
#     # #     path = os.path.join(os.path.dirname(__dir__), directory)
#     # #     name1 = 'magn_' + str(day) + '_kt_01.txt'
#     # #     filename1 = os.path.join(path, name1)
#     # #     name2 = 'magn_' + str(day) + '_kt_02.txt'
#     # #     filename2 = os.path.join(path, name2)
#     # #     name3 = 'magn_' + str(day) + '_kt_03.txt'
#     # #     filename3 = os.path.join(path, name3)
#     # #     name4 = 'magn_' + str(day) + '_kt_04.txt'
#     # #     filename4 = os.path.join(path, name4)
#     # #     filenames = [filename1, filename2, filename3, filename4]
#     # #
#     # #     outname = 'magn_' + str(day) + '_kt.txt'
#     # #     outfilename = os.path.join(path, outname)
#     # #
#     # #     with open(outfilename, 'w') as outfile:
#     # #         with open(filenames[0]) as infile:
#     # #             for line in infile:
#     # #                 outfile.write(line)
#     # #         for fname in filenames[1:]:
#     # #             with open(fname) as infile:
#     # #                 for i, line in enumerate(infile):
#     # #                     if i > 0:
#     # #                         outfile.write(line)
#     # #
#     # #     content = Table.read(outfilename, format='ascii')
#     # #     fitsname = 'magn_' + str(day) + '_kt.fits'
#     # #     fitsfilepath = os.path.join(path, fitsname)
#     # #     content.write(fitsfilepath, overwrite=True)
#     # #
#     # #     return fitsfilepath
#     # #
#     # # def mcilwain_fits(self, week, day):
#     # #     """This function extracts the data from a LAT-spacecraft file and writes it into a fit file and returns the filepath: fitsfilepath\n
#     # #     Input:\n
#     # #     writefile.mcilwain_fits ( week = WWW, day = JJMMDD )\n
#     # #     Output:\n
#     # #     0 = fitsfilepath"""
#     # #
#     # #     # get the data from the file
#     # #     datum = '20' + str(day)[:2] + '-' + str(day)[2:4] + '-' + str(day)[4:]
#     # #     lat_data = ExternalProps.lat_spacecraft(ExternalProps(),
#     # #                                             week)  # pay attention to the first and last day of the weekly file, as they are split in two!
#     # #     lat_time = lat_data[0]
#     # #     mc_b = lat_data[1]
#     # #     mc_l = lat_data[2]
#     # #
#     # #     # convert the time of the files into dates
#     # #     date = calculate.met_to_date(calculate(), lat_time)[3]
#     # #     date = np.array(date)
#     # #
#     # #     # extract the indices where the dates match the chosen day
#     # #     x = []
#     # #     for i in range(0, len(date)):
#     # #         date[i] = str(date[i])
#     # #         if date[i][:10] == datum:
#     # #             x.append(i)
#     # #
#     # #     x = np.array(x)
#     # #
#     # #     if x[0] == 0:
#     # #         week2 = str(int(week) - 1)
#     # #         lat_data2 = ExternalProps.lat_spacecraft(ExternalProps(),
#     # #                                                  week2)  # pay attention to the first and last day of the weekly file, as they are split in two!
#     # #         lat_time2 = lat_data2[0]
#     # #         mc_b2 = lat_data2[1]
#     # #         mc_l2 = lat_data2[2]
#     # #
#     # #         lat_time = np.append(lat_time2, lat_time)
#     # #         mc_b = np.append(mc_b2, mc_b)
#     # #         mc_l = np.append(mc_l2, mc_l)
#     # #
#     # #         date = calculate.met_to_date(calculate(), lat_time)[3]
#     # #         date = np.array(date)
#     # #
#     # #         x_new = []
#     # #         for i in range(0, len(date)):
#     # #             date[i] = str(date[i])
#     # #             if date[i][:10] == datum:
#     # #                 x_new.append(i)
#     # #
#     # #         x = np.array(x_new)
#     # #
#     # #     if x[-1] == len(date):
#     # #         week2 = str(int(week) + 1)
#     # #         lat_data2 = ExternalProps.lat_spacecraft(ExternalProps(),
#     # #                                                  week2)  # pay attention to the first and last day of the weekly file, as they are split in two!
#     # #         lat_time2 = lat_data2[0]
#     # #         mc_b2 = lat_data2[1]
#     # #         mc_l2 = lat_data2[2]
#     # #
#     # #         lat_time = np.append(lat_time, lat_time2)
#     # #         mc_b = np.append(mc_b, mc_b2)
#     # #         mc_l = np.append(mc_l, mc_l2)
#     # #
#     # #         date = calculate.met_to_date(calculate(), lat_time)[3]
#     # #         date = np.array(date)
#     # #
#     # #         x_new = []
#     # #         for i in range(0, len(date)):
#     # #             date[i] = str(date[i])
#     # #             if date[i][:10] == datum:
#     # #                 x_new.append(i)
#     # #
#     # #         x = np.array(x_new)
#     # #
#     # #     x1 = x[0] - 1
#     # #     x2 = x[-1] + 2
#     # #
#     # #     # limit the date to the chosen day, however take one additional datapoint before and after
#     # #     lat_time = lat_time[x1:x2]
#     # #     mc_b = mc_b[x1:x2]
#     # #     mc_l = mc_l[x1:x2]
#     # #
#     # #     # interpolate the data to get the datapoints with respect to the GBM sat_time and not the LAT_time
#     # #     interpol1 = calculate.intpol(calculate(), mc_b, day, 1, 0, lat_time)
#     # #     mc_b = interpol1[0]
#     # #     sat_time = interpol1[1]
#     # #
#     # #     interpol2 = calculate.intpol(calculate(), mc_l, day, 1, 0, lat_time)
#     # #     mc_l = interpol2[0]
#     # #
#     # #     # first write the data into an ascii file and the convert it into a fits file
#     # #     filename = 'glg_mcilwain_all_' + str(day) + '_kt.txt'
#     # #     filepath = get_path_of_data_file('mcilwain/', str(filename))
#     # #
#     # #     mc_file = open(filepath, 'w')
#     # #     for i in range(0, len(sat_time)):
#     # #         mc_file.write(str(sat_time[i]) + ' ' + str(mc_b[i]) + ' ' + str(mc_l[i]) + '\n')
#     # #     mc_file.close()
#     # #
#     # #     content = Table.read(filepath, format='ascii')
#     # #     fitsname = 'glg_mcilwain_all_' + str(day) + '_kt.fits'
#     # #     fitsfilepath = get_path_of_data_file('mcilwain/', fitsname)
#     # #     content.write(fitsfilepath, overwrite=True)
#     # #     return fitsfilepath
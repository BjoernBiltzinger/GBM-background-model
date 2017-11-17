#!/usr/bin python2.7

import getpass
import os

import numpy                               as np
from astropy.io import fits

from gbmbkgpy.work_module_refactor import calculate, detector


class readfile(object):
    """This class contains all functions for reading the files needed for the GBM background model:\n
    cspec(self, detector, day, seconds = 0) -> echan, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime\n
    ctime(self, detector, day, seconds = 0) -> echan, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime\n
    earth_occ(self) -> earth_ang, angle_d, area_frac, free_area, occ_area\n
    flares(self, year) -> day, time\n
    lat_spacecraft (self, week) -> lat_time, mc_b, mc_l\n
    magfits (self, day) -> t_magn, h_magn, x_magn, y_magn, z_magn\n
    mcilwain (self, day) -> sat_time, mc_b, mc_l\n
    poshist(self, day) -> sat_time, sat_pos, sat_lat, sat_lon, sat_q\n
    poshist_bin(self, day, bin_time_mid = 0, detector = 0, data_type = 'ctime') -> sat_time_bin, sat_pos_bin, sat_lat_bin, sat_lon_bin, sat_q_bin, bin_time_mid\n
    saa(self) -> saa\n\n\n"""

    def cspec(self, detector_name, day, seconds=0):
        """This function reads a cspec file and stores the data in arrays of the form: echan[emin, emax], total_counts, echan_counts[echan], exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n
        Input:\n
        readfile.cspec ( detector, day = YYMMDD, seconds = SSS )\n
        Output:\n
        0 = energy channel interval\n
        1 = total number of counts\n
        2 = number of counts per energy channel\n
        3 = total count rate\n
        4 = count rate per energy channel\n
        5 = bin time interval[start, end]\n
        6 = good time interval[start, end]\n
        7 = time of exposure\n"""

        det = getattr(detector(), detector_name)

        # read the file. Check if one wants to read a specific trigger file or a daily file
        if seconds == 0:
            filename = 'glg_cspec_' + detector_name + '_' + str(day) + '_v00.pha'
        else:
            filename = 'glg_cspec_' + detector_name + '_bn' + str(day) + str(seconds) + '_v00.pha'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/cspec/' + str(day) + '/'
        filepath = os.path.join(fits_path, str(filename))
        if os.path.isfile(filepath) == False:
            if seconds == 0:
                while os.path.isfile(filepath) == False:
                    download.data(download(), detector_name, day, 'cspec')

            else:
                print 'Please download the file ' + filename + ' first and save it in the appropriate directory.'
                return

        cspec_fits = fits.open(filepath)
        energy = cspec_fits[1].data
        spectrum = cspec_fits[2].data
        goodtime = cspec_fits[3].data
        cspec_fits.close()

        # extract the data
        emin = energy['E_MIN']  # lower limit of the energy channels
        emax = energy['E_MAX']  # upper limit of the energy channels
        chan_energies = np.zeros((len(emin), 2),
                                 float)  # combine the energy limits of the energy channels in one matrix
        chan_energies[:, 0] = emin
        chan_energies[:, 1] = emax
        counts = spectrum['COUNTS']
        total_counts = np.sum(counts, axis=1)  # total number of counts for each time intervall
        echan_counts = np.vstack(([counts[:, 0].T], [counts[:, 1].T], [counts[:, 2].T], [counts[:, 3].T],
                                  [counts[:, 4].T], [counts[:, 5].T], [counts[:, 6].T], [counts[:, 7].T],
                                  [counts[:, 8].T], [counts[:, 9].T], [counts[:, 10].T], [counts[:, 11].T],
                                  [counts[:, 12].T], [counts[:, 13].T], [counts[:, 14].T], [counts[:, 15].T],
                                  [counts[:, 16].T], [counts[:, 17].T], [counts[:, 18].T], [counts[:, 19].T],
                                  [counts[:, 20].T], [counts[:, 21].T], [counts[:, 22].T], [counts[:, 23].T],
                                  [counts[:, 24].T], [counts[:, 25].T], [counts[:, 26].T], [counts[:, 27].T],
                                  [counts[:, 28].T], [counts[:, 29].T], [counts[:, 30].T], [counts[:, 31].T],
                                  [counts[:, 32].T], [counts[:, 33].T], [counts[:, 34].T], [counts[:, 35].T],
                                  [counts[:, 36].T], [counts[:, 37].T], [counts[:, 38].T], [counts[:, 39].T],
                                  [counts[:, 40].T], [counts[:, 41].T], [counts[:, 42].T], [counts[:, 43].T],
                                  [counts[:, 44].T], [counts[:, 45].T], [counts[:, 46].T], [counts[:, 47].T],
                                  [counts[:, 48].T], [counts[:, 49].T], [counts[:, 50].T], [counts[:, 51].T],
                                  [counts[:, 52].T], [counts[:, 53].T], [counts[:, 54].T], [counts[:, 55].T],
                                  [counts[:, 56].T], [counts[:, 57].T], [counts[:, 58].T], [counts[:, 59].T],
                                  [counts[:, 60].T], [counts[:, 61].T], [counts[:, 62].T], [counts[:, 63].T],
                                  [counts[:, 64].T], [counts[:, 65].T], [counts[:, 66].T], [counts[:, 67].T],
                                  [counts[:, 68].T], [counts[:, 69].T], [counts[:, 70].T], [counts[:, 71].T],
                                  [counts[:, 72].T], [counts[:, 73].T], [counts[:, 74].T], [counts[:, 75].T],
                                  [counts[:, 76].T], [counts[:, 77].T], [counts[:, 78].T], [counts[:, 79].T],
                                  [counts[:, 80].T], [counts[:, 81].T], [counts[:, 82].T], [counts[:, 83].T],
                                  [counts[:, 84].T], [counts[:, 85].T], [counts[:, 86].T], [counts[:, 87].T],
                                  [counts[:, 88].T], [counts[:, 89].T], [counts[:, 90].T], [counts[:, 91].T],
                                  [counts[:, 92].T], [counts[:, 93].T], [counts[:, 94].T], [counts[:, 95].T],
                                  [counts[:, 96].T], [counts[:, 97].T], [counts[:, 98].T], [counts[:, 99].T],
                                  [counts[:, 100].T], [counts[:, 101].T], [counts[:, 102].T], [counts[:, 103].T],
                                  [counts[:, 104].T], [counts[:, 105].T], [counts[:, 106].T], [counts[:, 107].T],
                                  [counts[:, 108].T], [counts[:, 109].T], [counts[:, 110].T], [counts[:, 111].T],
                                  [counts[:, 112].T], [counts[:, 113].T], [counts[:, 114].T], [counts[:, 115].T],
                                  [counts[:, 116].T], [counts[:, 117].T], [counts[:, 118].T], [counts[:, 119].T],
                                  [counts[:, 120].T], [counts[:, 121].T], [counts[:, 122].T], [counts[:, 123].T],
                                  [counts[:, 124].T], [counts[:, 125].T], [counts[:, 126].T], [counts[:,
                                                                                               127].T]))  # number of counts as a table with respect to the energy channel -> echan_counts[0] are the counts for the first energy channel
        exptime = spectrum['EXPOSURE']  # length of the time intervall
        quality = spectrum['QUALITY']  # bad measurement indicator
        bad = np.where(quality == 1)  # create indices to delete bad measurements from data
        echan_counts = np.delete(echan_counts, bad, 1)  # remove bad datapoints
        total_counts = np.delete(total_counts, bad)
        exptime = np.delete(exptime, bad)
        total_rate = np.divide(total_counts, exptime)  # total count rate for each time intervall
        total_rate = np.array(total_rate)
        echan_rate = np.divide(echan_counts, exptime)  # count rate per time intervall for each energy channel
        echan_rate = np.array(echan_rate)
        cstart = spectrum['TIME']  # start time of the time intervall
        cstop = spectrum['ENDTIME']  # end time of the time intervall
        cstart = np.delete(cstart, bad)  # remove bad datapoints
        cstop = np.delete(cstop, bad)
        bin_time = np.zeros((len(cstart), 2), float)  # combine the time limits of the counting intervals in one matrix
        bin_time[:, 0] = cstart
        bin_time[:, 1] = cstop
        gtstart = goodtime['START']  # start time of data collecting times (exiting SAA)
        gtstop = goodtime['STOP']  # end time of data collecting times (entering SAA)
        # times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
        good_time = np.zeros((len(gtstart), 2),
                             float)  # combine the time limits of the goodtime intervals in one matrix
        good_time[:, 0] = gtstart
        good_time[:, 1] = gtstop
        return chan_energies, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime

    def ctime(self, detector_name, day, seconds=0):
        """This function reads a cspec file and stores the data in arrays of the form: echan[emin, emax], total_counts, echan_counts[echan], exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n
        Input:\n
        readfile.ctime ( detector, day = YYMMDD, seconds = SSS )\n
        Output:\n
        0 = energy channel interval\n
        1 = total number of counts\n
        2 = number of counts per energy channel\n
        3 = total count rate\n
        4 = count rate per energy channel\n
        5 = bin time interval[start, end]\n
        6 = good time interval[start, end]\n
        7 = time of exposure\n"""

        det = getattr(detector(), detector_name)

        # read the file. Check if one wants to read a specific trigger file or a daily file
        if seconds == 0:
            filename = 'glg_ctime_' + detector_name + '_' + str(day) + '_v00.pha'
        else:
            filename = 'glg_ctime_' + detector_name + '_bn' + str(day) + str(seconds) + '_v00.pha'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/ctime/' + str(day) + '/'
        filepath = os.path.join(fits_path, str(filename))
        if os.path.isfile(filepath) == False:
            if seconds == 0:
                while os.path.isfile(filepath) == False:
                    download.data(download(), detector_name, day, 'ctime')

            else:
                print 'Please download the file ' + filename + ' first and save it in the appropriate directory.'
                return

        ctime_fits = fits.open(filepath)
        energy = ctime_fits[1].data
        spectrum = ctime_fits[2].data
        goodtime = ctime_fits[3].data
        ctime_fits.close()

        # extract the data
        emin = energy['E_MIN']  # lower limit of the energy channels
        emax = energy['E_MAX']  # upper limit of the energy channels
        chan_energies = np.zeros((len(emin), 2),
                                 float)  # combine the energy limits of the energy channels in one matrix
        chan_energies[:, 0] = emin
        chan_energies[:, 1] = emax
        counts = spectrum['COUNTS']
        total_counts = np.sum(counts, axis=1)  # total number of counts for each time intervall
        echan_counts = np.vstack(([counts[:, 0].T], [counts[:, 1].T], [counts[:, 2].T], [counts[:, 3].T],
                                  [counts[:, 4].T], [counts[:, 5].T], [counts[:, 6].T], [counts[:,
                                                                                         7].T]))  # number of counts as a table with respect to the energy channel -> echan_counts[0] are the counts for the first energy channel
        exptime = spectrum['EXPOSURE']  # length of the time intervall
        quality = spectrum['QUALITY']  # bad measurement indicator
        bad = np.where(quality == 1)  # create indices to delete bad measurements from data
        echan_counts = np.delete(echan_counts, bad, 1)  # remove bad datapoints
        total_counts = np.delete(total_counts, bad)
        exptime = np.delete(exptime, bad)
        total_rate = np.divide(total_counts, exptime)  # total count rate for each time intervall
        total_rate = np.array(total_rate)
        echan_rate = np.divide(echan_counts, exptime)  # count rate per time intervall for each energy channel
        echan_rate = np.array(echan_rate)
        cstart = spectrum['TIME']  # start time of the time intervall
        cstop = spectrum['ENDTIME']  # end time of the time intervall
        cstart = np.delete(cstart, bad)  # remove bad datapoints
        cstop = np.delete(cstop, bad)
        bin_time = np.zeros((len(cstart), 2), float)  # combine the time limits of the counting intervals in one matrix
        bin_time[:, 0] = cstart
        bin_time[:, 1] = cstop
        gtstart = goodtime['START']  # start time of data collecting times (exiting SAA)
        gtstop = goodtime['STOP']  # end time of data collecting times (entering SAA)
        # times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
        good_time = np.zeros((len(gtstart), 2),
                             float)  # combine the time limits of the goodtime intervals in one matrix
        good_time[:, 0] = gtstart
        good_time[:, 1] = gtstop
        return chan_energies, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime

    def poshist(self, day):
        """This function reads a posthist file and stores the data in arrays of the form: sat_time, sat_pos, sat_lat, sat_lon, sat_q.\n
        Input:\n
        readfile.poshist ( day = YYMMDD )\n
        Output:\n
        0 = time\n
        1 = position (x, y, z)\n
        2 = latitude\n
        3 = longitude\n
        4 = quaternion matrix (q1, q2, q3, q4)"""

        # read the file
        filename = 'glg_poshist_all_' + str(day) + '_v00.fit'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/poshist/'
        filepath = os.path.join(fits_path, str(filename))
        while os.path.isfile(filepath) == False:
            # download.poshist(download(), day)
            try:
                download.poshist(download(), day)
            except urllib2.HTTPError as err:
                if err.code == 404:
                    filename = 'glg_poshist_all_' + str(day) + '_v01.fit'
                    filepath = os.path.join(fits_path, str(filename))
                    download.poshist(download(), day, 'v01')
                else:
                    raise

        pos_fits = fits.open(filepath)
        data = pos_fits[1].data
        pos_fits.close()

        # extract the data
        sat_time = data.SCLK_UTC  # Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
        sat_pos = np.array([data.POS_X, data.POS_Y, data.POS_Z])  # Position in J2000 equatorial coordinates
        sat_lat = data.SC_LAT
        sat_lon = data.SC_LON  # Earth-angles -> considers earth rotation (needed for SAA)
        sat_q = np.array([data.QSJ_1, data.QSJ_2, data.QSJ_3,
                          data.QSJ_4])  # Quaternionen -> 4D-space with which one can describe rotations (rocking motion); regarding the satellite system with respect to the J2000 geocentric coordinate system
        return sat_time, sat_pos, sat_lat, sat_lon, sat_q

    def poshist_bin(self, day, bin_time_mid=0, detector_name=0, data_type='ctime'):
        """This function reads a posthist file, converts the data-arrays into the form of the bin_time-arrays from the 'ctime' or 'cspec' files and stores the data in arrays of the form: sat_time_bin, sat_pos_bin, sat_lat_bin, sat_lon_bin, sat_q_bin, bin_time_mid.\n
        Input:\n
        read_poshist ( day = YYMMDD,\n
        bin_time_mid = 0(input bin_time_mid if available; default: 0), \n
        detector = 0(input the detector in the form det.n0; default: 0), \n
        data_type = 'ctime'(input ctime or cspec as string; default: 'ctime') )\n
        Output:\n
        0 = time\n
        1 = position (x, y, z)\n
        2 = latitude\n
        3 = longitude\n
        4 = quaternion matrix (q1, q2, q3, q4)\n
        5 = bin_time_mid"""

        # read the file
        filename = 'glg_poshist_all_' + str(day) + '_v00.fit'
        user = getpass.getuser()
        fits_path = '/home/' + user + '/Work/poshist/'
        filepath = os.path.join(fits_path, str(filename))
        while os.path.isfile(filepath) == False:
            # download.poshist(download(), day)
            try:
                download.poshist(download(), day)
            except urllib2.HTTPError as err:
                if err.code == 404:
                    filename = 'glg_poshist_all_' + str(day) + '_v01.fit'
                    filepath = os.path.join(fits_path, str(filename))
                    download.poshist(download(), day, 'v01')
                else:
                    raise

        pos_fits = fits.open(filepath)
        data = pos_fits[1].data
        pos_fits.close()

        # extract the data
        sat_time = data.SCLK_UTC  # Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
        sat_pos = np.array([data.POS_X, data.POS_Y, data.POS_Z])  # Position in J2000 equatorial coordinates
        sat_lat = data.SC_LAT
        sat_lon = data.SC_LON  # Earth-angles -> considers earth rotation (needed for SAA)
        sat_q = np.array([data.QSJ_1, data.QSJ_2, data.QSJ_3,
                          data.QSJ_4])  # Quaternionen -> 4D-space with which one can describe rotations (rocking motion); regarding the satellite system with respect to the J2000 geocentric coordinate system

        # convert the poshist-data-arrays to the binning of the measurement-data
        # sat_time
        sat_time_conv = calculate.intpol(calculate(), sat_time, day, 0, sat_time, bin_time_mid, detector_name,
                                         data_type)
        sat_time_bin = np.array(sat_time_conv[0])
        bin_time_mid = np.array(sat_time_conv[2])
        # sat_pos
        sat_pos_bin = calculate.intpol(calculate(), sat_pos, day, 0, sat_time, bin_time_mid, detector_name, data_type)[
            0]
        # sat_lat
        sat_lat_bin = calculate.intpol(calculate(), sat_lat, day, 0, sat_time, bin_time_mid, detector_name, data_type)[
            0]
        # sat_lon
        sat_lon_bin = calculate.intpol(calculate(), sat_lon, day, 0, sat_time, bin_time_mid, detector_name, data_type)[
            0]
        # sat_q
        sat_q_bin = calculate.intpol(calculate(), sat_q, day, 0, sat_time, bin_time_mid, detector_name, data_type)[0]
        return sat_time_bin, sat_pos_bin, sat_lat_bin, sat_lon_bin, sat_q_bin, bin_time_mid


class download(object):
    """This class contains all functions for downloading the files needed for the GBM background model:\n
    data(self, detector, day, data_type = 'ctime', seconds = 0)\n
    flares(self, year)\n
    lat_spacecraft (self, week)\n
    poshist(self, day)\n\n\n"""

    def data(self, detector_name, day, data_type='ctime'):
        """This function downloads a daily data file and stores it in the appropriate folder\n
        Input:\n
        download.data ( detector, day = YYMMDD, data_type = 'ctime' (or 'cspec') )\n"""

        user = getpass.getuser()

        # create the appropriate folder if it doesn't already exist and switch to it
        file_path = '/home/' + user + '/Work/' + str(data_type) + '/' + str(day) + '/'
        if not os.access(file_path, os.F_OK):
            print("Making New Directory")
            os.mkdir(file_path)

        os.chdir(file_path)

        url = (
        'http://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/20' + str(day)[:2] + '/' + str(day)[2:4] + '/' + str(
            day)[4:] + '/current/glg_' + str(data_type) + '_' + detector_name + '_' + str(day) + '_v00.pha')
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

    def poshist(self, day, version='v00'):
        """This function downloads a daily poshist file and stores it in the appropriate folder\n
        Input:\n
        download.poshist ( day = YYMMDD, version = 'v00' )\n"""

        user = getpass.getuser()

        # create the appropriate folder if it doesn't already exist and switch to it
        file_path = '/home/' + user + '/Work/poshist/'
        if not os.access(file_path, os.F_OK):
            print("Making New Directory")
            os.mkdir(file_path)

        os.chdir(file_path)

        url = (
        'http://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/20' + str(day)[:2] + '/' + str(day)[2:4] + '/' + str(
            day)[4:] + '/current/glg_poshist_all_' + str(day) + '_' + str(version) + '.fit')
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
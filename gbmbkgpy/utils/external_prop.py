import astropy.io.fits as fits
import astropy.time as astro_time
import astropy.units as u
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from gbmgeometry import GBMTime
import os
from gbmbkgpy.io.downloading import download_flares, download_lat_spacecraft
from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.package_data import get_path_of_data_file
from gbmbkgpy.modeling.point_source import PointSrc


class ExternalProps(object):

    def __init__(self, day):
        """
        Build the external properties for a given day
        :param day: YYMMDD
        """

        assert isinstance(day,str), 'Day must be a string'
        assert len(day) == 6, 'Day must be in format YYMMDD'



        # compute all the quantities that are needed for making date calculations

        self._day = day
        self._year = '20%s'%day[:2]
        self._month = day[2:-2]
        self._dd = day[-2:]

        day_at = astro_time.Time("%s-%s-%s" % (self._year, self._month, self._dd))

        self._min_met = GBMTime(day_at).met

        self._max_met = GBMTime(day_at + u.day).met

        # now read external files which can be downloaded
        # as needed

        #self._build_flares()
        self._read_saa()
        #self._build_point_sources()
        self._build_lat_spacecraft()

        # self._earth_occ()

    def build_point_sources(self, data):
        self._data = data
        self._build_point_sources()

    def build_some_source(self,data, source_list):
        self._data=data
        self._build_some_source(source_list)

    def mc_l(self, met):
        """
        Get MC L for a given MET
        :param met: 
        :return: 
        """

        return self._mc_l_interp(met)


    def mc_b(self, met):
        """
        Get MC B for a given MET
        :param met: 
        :return: 
        """

        return self._mc_b_interp(met)

    @property
    def saa(self):
        return self._saa_properties

    @property
    def flares(self):
        return self._flares_properties[self._flare_idx]

    @property
    def earth_occ(self):
        return self._earth_occ_properties

    @property
    def point_sources(self):
        return self._point_sources_dic

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

        del saa, saa_lat, saa_lon

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


    def _build_flares(self):
        """This function reads the YYYY.txt file containing the GOES solar flares of the corresponding year and returns the data in arrays of the form: day, time\n
        Input:\n
        year = YYYY\n
        Output\n
        0 = day ('YYMMDD')
        1 = time[start][stop] (in seconds on that day -> accuracy ~ 1 minute)\n"""
        filename =  '%s.dat' % self._year
        filepath = get_path_of_data_file('flares', str(filename))

        if not file_existing_and_readable(filepath):

            download_flares(self._year)

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

        del flares_dic, day, start, stop

    def _build_lat_spacecraft(self):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file

        day = astro_time.Time("%s-%s-%s" %(self._year, self._month, self._dd))

        gbm_time = GBMTime(day)

        mission_week = np.floor(gbm_time.mission_week.value)


        filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % mission_week
        filepath = get_path_of_data_file('lat', filename)


        if not file_existing_and_readable(filepath):

            download_lat_spacecraft(mission_week)


        # lets check that this file has the right info

        week_before = False
        week_after = False

        with fits.open(filepath) as f:

            if (f['PRIMARY'].header['TSTART'] >= self._min_met):

                # we need to get week before

                week_before = True

                before_filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week - 1)
                before_filepath = get_path_of_data_file('lat', before_filename)

                if not file_existing_and_readable(before_filepath):
                    download_lat_spacecraft(mission_week - 1)


            if (f['PRIMARY'].header['TSTOP'] <= self._max_met):

                # we need to get week after

                week_after = True

                after_filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week + 1)
                after_filepath = get_path_of_data_file('lat', after_filename)

                if not file_existing_and_readable( after_filepath):
                    download_lat_spacecraft(mission_week + 1)


            # first lets get the primary file

            lat_time = np.mean( np.vstack( (f['SC_DATA'].data['START'],f['SC_DATA'].data['STOP'])),axis=0)
            mc_l = f['SC_DATA'].data['L_MCILWAIN']
            mc_b = f['SC_DATA'].data['B_MCILWAIN']


        # if we need to append anything to make up for the
        # dates not being included in the files
        # do it here... thanks Fermi!
        if week_before:

            with fits.open(before_filepath) as f:

                lat_time_before = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                mc_l_before = f['SC_DATA'].data['L_MCILWAIN']
                mc_b_before = f['SC_DATA'].data['B_MCILWAIN']


            mc_b = np.append(mc_b_before, mc_b)
            mc_l = np.append(mc_l_before, mc_l)
            lat_time = np.append(lat_time_before, lat_time)

        if week_after:

            with fits.open(after_filepath) as f:
                lat_time_after = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                mc_l_after = f['SC_DATA'].data['L_MCILWAIN']
                mc_b_after = f['SC_DATA'].data['B_MCILWAIN']

            mc_b = np.append(mc_b, mc_b_after)
            mc_l = np.append(mc_l, mc_l_after)
            lat_time = np.append(lat_time, lat_time_after)

        """
        # save them
        #TODO: do we need use the mean here?
        self._mc_l = mc_l
        self._mc_b = mc_b
        self._mc_time = lat_time

        # interpolate them

        self._mc_b_interp = interpolate.interp1d(self._mc_time, self._mc_b)
        self._mc_l_interp = interpolate.interp1d(self._mc_time, self._mc_l)
        """
        #remove the self-variables for memory saving
        self._mc_b_interp = interpolate.interp1d(lat_time, mc_b)
        self._mc_l_interp = interpolate.interp1d(lat_time, mc_l)

        del mc_l, mc_b, lat_time

    def _build_point_sources(self):
        """This function reads the point_sources.dat file and returns the sources in the form: names, coordinates[ra][dec]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = source_names
        1 = coordinates\n"""
        file_path = get_path_of_data_file('background_point_sources/', 'point_sources.dat')

        self._ps_df = pd.read_table(file_path, names=['name', 'ra', 'dec'])

        # instantiate dic of point source objects
        self._point_sources_dic = {}

        # TODO: investigate why calculation runs non stop, print ps_df and look

        ###Single core calc###
        for row in self._ps_df.itertuples():
            self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._data)


        """
        ###multicore calc###
        from pathos.multiprocessing import ProcessPool
        # define function for multiprocess calculation
        def calc_pointsources(x):
            return self._ps_df.loc[x][0], PointSource(self._ps_df.loc[x][0], self._ps_df.loc[x][1],
                                                      self._ps_df.loc[x][2], self._data)

        # Initialize Process pool with 8 threads       
        pool = ProcessPool(8)

        results = pool.map(calc_pointsources, range(len(self._ps_df)))

        # instantiate dic of point source objects
        for i in range(len(results)):
            self._point_sources_dic[results[i][0]] = results[i][1]

        del results
        """

    #function to build only some pointsources, which are specified in the source_list

    def _build_some_source(self, source_list):
        file_path = get_path_of_data_file('background_point_sources/', 'point_sources.dat')

        self._ps_df = pd.read_table(file_path, names=['name', 'ra', 'dec'])

        # instantiate dic of point source objects
        self._point_sources_dic = {}

        ###Single core calc###
        for row in self._ps_df.itertuples():
            for element in source_list:
                if row[1]==element:
                    print(element)
                    print(row[2], row[3])
                    self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._data)

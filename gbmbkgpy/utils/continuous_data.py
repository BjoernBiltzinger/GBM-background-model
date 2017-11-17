import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list


import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable

import astropy.time as astro_time
import astropy.coordinates as coord



from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file
from gbmbkgpy.utils.progress_bar import progress_bar

class ContinuousData(object):
    def __init__(self, file_name, position_history):




        _, self._data_type, self._det, self._day, _ = file_name.split('_')

        assert 'ctime' in self._data_type, 'currently only working for CTIME data'
        assert 'n' in self._det, 'currently only working NAI detectors'

        self._pos_hist = position_history

        _, _,_,pos_hist_day,_ = self._pos_hist.split('_')

        assert pos_hist_day == self._day, 'Position history file does not match data file day'




        with fits.open(file_name) as f:
            self._counts = f['SPECTRUM'].data['COUNTS']
            self._bin_start = f['SPECTRUM'].data['TIME']
            self._bin_stop = f['SPECTRUM'].data['ENDTIME']

            self._n_entries = len(self._bin_start)

            self._exposure = f['SPECTRUM'].data['EXPOSURE']
            self._bin_start = f['SPECTRUM'].data['TIME']

        self._n_time_bins, self._n_channels  = self._counts.shape

        self._setup_geometery()



        #self._calculate_ang_eff()


    @property
    def day(self):
        return self._day

    @property
    def data_type(self):
        return self._data_type


    @property
    def detector_id(self):

        return self._det[-1]

    @property
    def n_channels(self):

        return self._n_channels

    @property
    def n_time_bins(self):

        return self._n_time_bins

    @property
    def rates(self):
        return self._counts / self._exposure.reshape((self._n_entries, 1))

    @property
    def counts(self):
        return self._counts

    @property
    def exposure(self):
        return self._exposure

    @property
    def time_bins(self):
        return np.vstack((self._bin_start, self._bin_stop)).T

    @property
    def mean_time(self):
        return np.mean(self.time_bins, axis=1)


    def get_quaternion(self, met):

        return self._position_interpolator.quaternion(met)


    def effective_angle(self,channel, angle):
        """
        
        :param channel: 
        :param angle: 
        :return: 
        """

        assert isinstance(channel,int), 'channel must be an integer'
        assert channel in range(self._n_channels), 'Invalid channel'


        return  interpolate.splev(angle, self._tck[channel], der=0)


    def _calculate_ang_eff(self):
        """This function converts the angle of one detectortype to a certain source into an effective angle considering the angular dependence of the effective area and stores the data in an array of the form: ang_eff\n
        Input:\n
        calculate.ang_eff ( ang (in degrees), echan (integer in the range of 0-7 or 0-127), datatype='ctime' (or 'cspec'), detectortype='NaI' (or 'BGO') )\n
        Output:\n
        0 = effective angle\n
        1 = normalized photopeak effective area curve"""

        fitsname = 'peak_eff_area_angle_calib_GBM_all_DRM.fits'

        fitsfilepath = get_path_of_data_file('calibration', fitsname)

        with fits.open(fitsfilepath) as fits_file:
            data = fits_file[1].data

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


        self._tck = collections.OrderedDict()

        if self._det[0] == 'n':

            #if self._data_type == 'ctime':
                # ctime linear-interpolation factors
                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 127./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 109./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # y1_fac = np.array([1.2, 1.08, 238./246., 196./246., 77./246., 0., 0., 0.])
                # y2_fac = np.array([0., 0., 5./246., 40./246., 159./246., 230./383., 0., 0.])
                # y3_fac = np.array([0., 0., 0., 0., 0., 133./383., .7, .5])

                # resulting effective area curve


            for echan in range(self._n_channels):

                y = y_all[echan]

                # normalize
                # y = y/y1[90]

                # calculate the angle factors
                self._tck[echan] = interpolate.splrep(x, y)
                #ang_eff = interpolate.splev(ang, tck, der=0)

                # convert the angle according to their factors
                # ang_eff = np.array(ang_fac*ang)
        else:

            raise NotImplementedError('BGO not implemented yet')

        #
        # else:
        #     print 'detector_type BGO not yet implemented'
        #
        # '''ang_rad = ang*(2.*math.pi)/360.
        # ang_eff = 110*np.cos(ang_rad)
        # ang_eff[np.where(ang > 90.)] = 0.'''
        # return ang_eff, y

    def _setup_geometery(self):


        n_bins_to_calculate = 800.

        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # ok we need to get the sun angle

        n_skip = int(np.ceil(self._n_time_bins/n_bins_to_calculate))

        sun_angle = []
        sun_time = []
        earth_angle = []
        pointing = []

        # go thru a subset of times and calculate the sun angle with GBM geometry

        with progress_bar(n_bins_to_calculate,title='Calculating sun position') as p:

            for mean_time in self.mean_time[::n_skip]:

                det = gbm_detector_list[self._det]( quaternion = self._position_interpolator.quaternion(mean_time),
                                                    sc_pos=self._position_interpolator.sc_pos(mean_time),
                                                    time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_time.append(mean_time)
                earth_angle.append(det.earth_angle.value)
                pointing.append(det.center.icrs)


                p.increase()

        # get the last data point

        mean_time = self.mean_time[-2]

        det = gbm_detector_list[self._det](quaternion=self._position_interpolator.quaternion(mean_time),
                                           sc_pos=self._position_interpolator.sc_pos(mean_time),
                                           time=astro_time.Time(self._position_interpolator.utc(mean_time)))

        sun_angle.append(det.sun_angle.value)
        sun_time.append(mean_time)
        earth_angle.append(det.earth_angle.value)


        self._pointing = np.array(pointing)#coord.concatenate(pointing)



        self._sun_angle = sun_angle
        self._sun_time = sun_time
        self._earth_angle = earth_angle


        # interpolate it

        self._sun_angle_interpolator = interpolate.interp1d(self._sun_time,self._sun_angle)
        self._earth_angle_interpolator = interpolate.interp1d(self._sun_time, self._earth_angle)


    @property
    def pointing(self):

        return self._pointing

    @property
    def interpolation_time(self):

        return self._sun_time

    def sun_angle(self, met):

        return self._sun_angle_interpolator(met)


    def earth_angle(self, met):

        return self._earth_angle_interpolator(met)



    def plot_angle(self):

        fig, ax = plt.subplots()

        ax.plot(self.mean_time[:-1],self.sun_angle(self.mean_time[:-1]))
        ax.plot(self.mean_time[:-1], self.earth_angle(self.mean_time[:-1]))

        ax.set_xlabel('MET')
        ax.set_ylabel('Angle (deg)')

        return fig


    def plot_eff_angle(self):

        fig, ax = plt.subplots()


        x_grid = np.linspace(-180,180,200)

        for i in range(self._n_channels):

            ax.plot(x_grid,self.effective_angle(i,x_grid))


        ax.set_xlabel('angle (deg)')
        ax.set_ylabel('effective area')



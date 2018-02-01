import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list


import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable

import astropy.time as astro_time
import astropy.coordinates as coord
import math
import numpy as np
from scipy import interpolate



from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.plotting.step_plots import step_plot, slice_disjoint, disjoint_patch_plot

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

        self._counts_combined = np.sum(self._counts, axis=1)

        self._counts_combined_rate = self._counts_combined / self.time_bin_length

        self._n_time_bins, self._n_channels  = self._counts.shape

        self._setup_geometery()

        self._compute_saa_regions()

        self._calculate_ang_eff()

        self._calculate_earth_occ()

        self._calculate_earth_occ_eff()


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
    def counts_combined(self):
        return self._counts_combined

    @property
    def counts_combined_rate(self):
        return self._counts_combined_rate

    @property
    def exposure(self):
        return self._exposure

    @property
    def time_bin_start(self):
        return self._bin_start

    @property
    def time_bin_stop(self):
        return self._bin_stop

    @property
    def time_bins(self):
        return np.vstack((self._bin_start, self._bin_stop)).T

    @property
    def time_bin_length(self):
        return self._bin_stop - self._bin_start

    @property
    def mean_time(self):
        return np.mean(self.time_bins, axis=1)


    def get_quaternion(self, met):

        return self._position_interpolator.quaternion(met)


    def effective_angle(self, angle, channel):
        """
        Interpolation function for the Solar Continuum
        :param channel: 
        :param angle: 
        :return: 
        """

        assert isinstance(channel,int), 'channel must be an integer'
        assert channel in range(self._n_channels), 'Invalid channel'

        return  np.array(interpolate.splev(angle, self._tck[channel], der=0))


    def effective_area(self, angle, channel):
        """
        Interpolation function for the earth albedo continuum
        :param channel:
        :param angle:
        :return:
        """

        assert isinstance(channel, int), 'channel must be an integer'
        assert channel in range(self._n_channels), 'Invalid channel'

        return np.array(interpolate.splev(angle, self._tck_occ[channel], der=0))

    def cgb_background(self, time_bins):

        return np.ones_like(time_bins)


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
        del y_all, x, y0, y1, y2, y3, y4, y5, y6, y7

    def _setup_geometery(self):

        n_bins_to_calculate = 800.

        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # ok we need to get the sun angle

        n_skip = int(np.ceil(self._n_time_bins/n_bins_to_calculate))

        sun_angle = []
        sun_time = []
        earth_angle = []
        earth_position = []
        pointing = []

        # go through a subset of times and calculate the sun angle with GBM geometry
        """
        ###SINGLECORE CALC###
        with progress_bar(n_bins_to_calculate, title='Calculating sun position') as p:

            for mean_time in self.mean_time[::n_skip]:
                det = gbm_detector_list[self._det](quaternion=self._position_interpolator.quaternion(mean_time),
                                                   sc_pos=self._position_interpolator.sc_pos(mean_time),
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_time.append(mean_time)
                earth_angle.append(det.earth_angle.value)
                earth_position.append(det.earth_position)
                pointing.append(det.center.icrs)

                p.increase()

        """
        ###MULTICORE CALC###
        time_array = self.mean_time[::n_skip]
        sun_angle_mul = []
        sun_time_mul = []
        earth_angle_mul = []
        earth_position_mul = []
        pointing_mul = []

        def calc_geo(i):

            time_array_slice = time_array[i * 100:i * 100 + 100]

            #with progress_bar(len(time_array_slice), title='Calculating sun position') as p:   #porgress bar seems to make break in multiprocess calculation
            for mean_time in time_array_slice:
                det = gbm_detector_list[self._det](quaternion=self._position_interpolator.quaternion(mean_time),
                                                   sc_pos=self._position_interpolator.sc_pos(mean_time),
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle_mul.append(det.sun_angle.value)
                sun_time_mul.append(mean_time)
                earth_angle_mul.append(det.earth_angle.value)
                earth_position_mul.append(det.earth_position)
                pointing_mul.append(det.center.icrs)

            #   p.increase()
            geo_dic = {}
            geo_dic['sun_angle'] = {}
            geo_dic['sun_time'] = {}
            geo_dic['earth_angle'] = {}
            geo_dic['earth_position'] = {}
            geo_dic['pointing'] = {}

            geo_dic['sun_angle'][i] = sun_angle_mul
            geo_dic['sun_time'][i] = sun_time_mul
            geo_dic['earth_angle'][i] = earth_angle_mul
            geo_dic['earth_position'][i] = earth_position_mul
            geo_dic['pointing'][i] = pointing_mul
            return geo_dic

        from pathos.multiprocessing import ProcessPool

        # Initialize Process pool with 8 threads
        pool = ProcessPool(8)

        results = pool.map(calc_geo, range(8))

        sun_angle_dic = {}
        sun_time_dic = {}
        earth_angle_dic = {}
        earth_position_dic = {}
        pointing_dic = {}

        for result in results:
            sun_angle_dic.update(result['sun_angle'])
            sun_time_dic.update(result['sun_time'])
            earth_angle_dic.update(result['earth_angle'])
            earth_position_dic.update(result['earth_position'])
            pointing_dic.update(result['pointing'])

        for i in range(8):
            sun_angle.extend(sun_angle_dic[i])
            sun_time.extend(sun_time_dic[i])
            earth_angle.extend(earth_angle_dic[i])
            earth_position.extend(earth_position_dic[i])
            pointing.extend(pointing_dic[i])

        del sun_angle_dic, sun_time_dic, earth_angle_dic, earth_position_dic, pointing_dic
        ##############

        # get the last data point

        mean_time = self.mean_time[-2]

        det = gbm_detector_list[self._det](quaternion=self._position_interpolator.quaternion(mean_time),
                                           sc_pos=self._position_interpolator.sc_pos(mean_time),
                                           time=astro_time.Time(self._position_interpolator.utc(mean_time)))

        sun_angle.append(det.sun_angle.value)
        sun_time.append(mean_time)
        earth_angle.append(det.earth_angle.value)
        earth_position.append(det.earth_position)
        pointing.append(det.center.icrs)


        self._pointing = np.array(pointing)#coord.concatenate(pointing)



        self._sun_angle = sun_angle
        self._sun_time = sun_time
        self._earth_angle = earth_angle
        self._earth_position = earth_position


        # interpolate it

        self._sun_angle_interpolator = interpolate.interp1d(self._sun_time, self._sun_angle)
        self._earth_angle_interpolator = interpolate.interp1d(self._sun_time, self._earth_angle)

        del sun_angle, sun_time, earth_angle, earth_position, pointing

    def _compute_saa_regions(self):

        # find where the counts are zero

        self._zero_idx = self._counts_combined == 0.

        idx = (self._zero_idx).nonzero()[0]

        slice_idx = np.array(slice_disjoint(idx))

        slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > 8)]

        # now find the times of the exits

        if slice_idx[-1 , 1] == self._n_time_bins - 1:

            # the last exit is just the end of the array
            self._saa_exit_idx = slice_idx[:-1, 1]

        else:

            self._saa_exit_idx = slice_idx[:, 1]

        self._saa_exit_mean_times = self.mean_time[self._saa_exit_idx]
        self._saa_exit_bin_start = self._bin_start[self._saa_exit_idx]
        self._saa_exit_bin_stop = self._bin_stop[self._saa_exit_idx]

        self._saa_slices = slice_idx

    def _calculate_earth_occ(self):
        """This function calculates the overlapping area fraction for a certain earth-angle and stores the data in arrays of the form: opening_ang, earth_occ\n
        Input:\n
        calc_earth_occ ( angle )\n
        Output:\n
        0 = angle of the detector-cone\n
        1 = area fraction of the earth-occulted area to the entire area of the detector-cone"""

        angles = np.arange(0, 180.5, .5)

        angle_d = []
        area_frac = []
        free_ar = []
        occ_area = []

        for i, angle in enumerate(angles):

            # get the distance from the satellite to the center of the earth
            sat_dist = 6912000.
            # get the earth_radius at the satellite's position
            earth_radius = 6371000.8
            atmosphere = 12000.
            r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere

            # define the opening angles of the overlapping cones (earth and detector).
            theta = math.asin(r / sat_dist)  # earth-cone
            opening_ang = np.arange(math.pi / 36000., math.pi / 2. + math.pi / 36000., math.pi / 36000.)

            # get the angle between the detector direction and the earth direction
            earth_ang = angle * 2. * math.pi / 360.

            # geometric considerations for the two overlapping spherical cap problem
            phi = math.pi / 2 - earth_ang
            f = (np.cos(theta) - np.cos(opening_ang) * np.sin(phi)) / (np.cos(phi))
            beta = np.arctan2(f, (np.cos(opening_ang)))

            # same considerations for the earth-component
            f_e = (np.cos(opening_ang) - np.cos(theta) * np.sin(phi)) / (np.cos(phi))
            beta_e = np.arctan2(f_e, (np.cos(theta)))

            # calculate one part of the overlapping area of the spherical caps. This area belongs to the detector-cone
            A_d_an2 = 2 * (np.arctan2(
                (np.sqrt(-(np.tan(beta)) ** 2 / ((np.sin(opening_ang)) ** 2) + (np.tan(beta)) ** 2 + 1) * np.sin(
                    opening_ang)),
                np.tan(beta)) - np.cos(opening_ang) * np.arccos(np.tan(beta) / np.tan(opening_ang)) - (np.arctan2(
                (np.sqrt(-(np.tan(beta)) ** 2 / ((np.sin(beta)) ** 2) + (np.tan(beta)) ** 2 + 1) * np.sin(beta)),
                np.tan(beta)) - np.cos(beta) * np.arccos(np.tan(beta) / np.tan(beta))))

            # calculate the other part of the overlapping area. This area belongs to the earth-cone
            A_e_an2 = 2 * (
                np.arctan2((np.sqrt(
                    -(np.tan(beta_e)) ** 2 / ((np.sin(theta)) ** 2) + (np.tan(beta_e)) ** 2 + 1) * np.sin(theta)),
                           np.tan(beta_e)) - np.cos(theta) * np.arccos(np.tan(beta_e) / np.tan(theta)) - (
                    np.arctan2((np.sqrt(
                        -(np.tan(beta_e)) ** 2 / ((np.sin(beta_e)) ** 2) + (np.tan(beta_e)) ** 2 + 1) * np.sin(beta_e)),
                               np.tan(beta_e)) - np.cos(beta_e) * np.arccos(np.tan(beta_e) / np.tan(beta_e))))

            # take the limitations of trignometric functions into account. -> Get rid of 2*pi jumps
            A_e_an2[np.where(earth_ang < beta)] = A_e_an2[np.where(earth_ang < beta)] - 2 * math.pi
            A_d_an2[np.where(f < 0)] = A_d_an2[np.where(f < 0)] - 2 * math.pi

            # combine the two area segments to get the total area
            A_an2 = A_d_an2 + A_e_an2

            # calculate the unocculted area of the detector cone
            free_area = 2 * math.pi * (1 - np.cos(opening_ang))

            # add values to the overlapping area, where either the detector-cone is completely embedded within the earth-cone or the other way around. Within this function both could be the case, because we are changing the angle of the detector-cone!
            A_an2[np.where(opening_ang <= theta - earth_ang)] = free_area[np.where(opening_ang <= theta - earth_ang)]
            A_an2[np.where(opening_ang >= theta + earth_ang)] = 2 * math.pi * (1 - np.cos(theta))
            A_an2[np.where(opening_ang <= earth_ang - theta)] = 0.

            # if the earth will never be within the detector-cone, the overlapping area will always be 0
            # if earth_ang > opening_ang[-1] + theta:
            #    A_an2 = np.zeros(len(opening_ang))

            # Apparently the numeric calculation of the analytic solution doesn't always return a value (probably because of runtime error). As a result there are several 'nan' entries in the A_an2 array. To get rid of those we interpolate over all the calculated solutions. We have chosen enough steps for the opening_ang to eliminate any errors due to this interpolation, because we get enough good results from the calculation.
            tck = interpolate.splrep(opening_ang[np.logical_not(np.isnan(A_an2))],
                                     A_an2[np.logical_not(np.isnan(A_an2))], s=0)
            A_an2 = interpolate.splev(opening_ang, tck, der=0)

            # calculate the fraction of the occulated area
            earth_occ = A_an2 / free_area

            angle_d.append(opening_ang * 180. / math.pi)
            area_frac.append(earth_occ)
            free_ar.append(free_area)
            occ_area.append(A_an2)

        self._earth_angs = angles
        self._angle_d = angle_d
        self._area_frac = area_frac
        self._free_area = free_area
        self._occ_area = occ_area

        del angles, angle_d, area_frac, free_area, occ_area


    def _calculate_earth_occ_eff(self):
        """This function converts the earth angle into an effective earth occultation considering the angular dependence of the effective area and stores the data in an array of the form: earth_occ_eff\n
        Input:\n
        calculate.earth_occ_eff ( earth_ang (in degrees), echan (integer in the range of 0-7 or 0-127), datatype='ctime' (or 'cspec'), detectortype='NaI' (or 'BGO') )\n
        Output:\n
        0 = effective unocculted detector area"""

        #earth_ang_0 = self._earth_angs  #np.arange(0, 180.5, .5)
        #angle_d = self._angle_d[0]
        #area_frac = self._area_frac
        #free_area = self._free_area#[0]
        #occ_area = self._occ_area

        self._tck_occ = collections.OrderedDict()

        for echan in range(self._n_channels):

            ang_fac = interpolate.splev(self._angle_d[0], self._tck[echan], der=0)

            free_circ_eff = [self._free_area[0] * ang_fac[0]]

            for i in range(1, len(self._free_area)):
                circ_area = self._free_area[i] - self._free_area[i - 1]
                circ_area_eff = circ_area * ang_fac[i]
                free_circ_eff.append(circ_area_eff)

            free_circ_eff = np.array(free_circ_eff)

            occ_circ_eff = []

            for j in range(0, len(self._earth_angs)):
                occ_circ_eff_0 = [self._occ_area[j][0] * ang_fac[0]]
                for i in range(1, len(self._occ_area[j])):
                    circ_area = self._occ_area[j][i] - self._occ_area[j][i - 1]
                    circ_area_eff = circ_area * ang_fac[i]
                    occ_circ_eff_0.append(circ_area_eff)

                occ_circ_eff.append(occ_circ_eff_0)

            occ_circ_eff = np.array(occ_circ_eff)
            # eff_area_frac = np.sum(occ_circ_eff)/np.sum(free_circ_eff)
            eff_area_frac_0 = np.sum(occ_circ_eff, axis=1) / np.sum(free_circ_eff)

            self._tck_occ[echan] = interpolate.splrep(self._earth_angs, eff_area_frac_0, s=0)

        del eff_area_frac_0, occ_circ_eff, occ_circ_eff_0, free_circ_eff, ang_fac

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

    @property
    def earth_position(self):

        return self._earth_position

    @property
    def saa_mask(self):

        return ~ self._zero_idx


    @property
    def saa_mean_times(self):

        return self._saa_exit_mean_times


    def saa_initial_values(self, echan):

        start_value_array = []

        for i, exit_idx in enumerate(self._saa_exit_idx):
            start_value_array.append(
                np.mean(self._counts[exit_idx:exit_idx+10, echan] / self.time_bin_length[exit_idx:exit_idx+10]))

        return np.array(start_value_array)

    def plot_light_curve(self,channel=0, ax=None):

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()



        step_plot(self.time_bins, self.rates[:,0], ax, fill=False, fill_min=0,color='green')


        # disjoint_patch_plot(ax, self._bin_start, self._bin_stop, ~self._zero_idx, )


        ax.set_ylabel('rate (cnts/s)')
        ax.set_xlabel('MET (s)')

        return fig


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


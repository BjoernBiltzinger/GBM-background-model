import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list

import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file, download_lat_spacecraft

import astropy.time as astro_time
import astropy.units as u
import astropy.coordinates as coord
import math
import numpy as np
from scipy import interpolate
import os
from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file, get_path_of_external_data_dir
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.plotting.step_plots import step_plot, slice_disjoint, disjoint_patch_plot
from gbmgeometry import GBMTime
from gbmbkgpy.utils.binner import Rebinner

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


class DataCleaner(object):

    def __init__(self, date, detector, data_type, rate_gen=None, min_bin_width=None, training=False, trigger_intervals=[]):
        self._data_type = data_type
        self._det = detector
        self._day = date
        self._rate_generator_DRM = rate_gen

        # compute all the quantities that are needed for making date calculations
        self._year = '20%s' % date[:2]
        self._month = date[2:-2]
        self._dd = date[-2:]

        day_at = astro_time.Time("%s-%s-%s" % (self._year, self._month, self._dd))

        self._min_met = GBMTime(day_at).met
        self._max_met = GBMTime(day_at + u.Quantity(1, u.day)).met

        # Calculate the MET time for the day
        day = self._day
        year = '20%s' % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        self._day_met = GBMTime(day_at).met

        day = astro_time.Time("%s-%s-%s" % (self._year, self._month, self._dd))
        gbm_time = GBMTime(day)
        self.mission_week = np.floor(gbm_time.mission_week.value)

        self.min_bin_width = min_bin_width
        self._training = training
        self._trigger_intervals = trigger_intervals

        # assert 'ctime' in self._data_type, 'currently only working for CTIME data'
        assert 'n' in self._det, 'currently only working NAI detectors'

        ### Download data-file and poshist file if not existing:
        datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(self._data_type, self._det, self._day)
        datafile_path = os.path.join(get_path_of_external_data_dir(), self._data_type, self._day, datafile_name)

        poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', self._day)
        poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)

        if not file_existing_and_readable(datafile_path):
            download_data_file(self._day, self._data_type, self._det)

        if not file_existing_and_readable(poshistfile_path):
            download_data_file(self._day, 'poshist')

        ###

        self._pos_hist = poshistfile_path

        with fits.open(datafile_path) as f:
            self._counts = f['SPECTRUM'].data['COUNTS']
            self._bin_start = f['SPECTRUM'].data['TIME']
            self._bin_stop = f['SPECTRUM'].data['ENDTIME']

            self._exposure = f['SPECTRUM'].data['EXPOSURE']
        # Delete entries if in data file there are time bins with same start and end time
        i = 0
        while i < len(self._bin_start):
            if self._bin_start[i] == self._bin_stop[i]:
                self._bin_start = np.delete(self._bin_start, [i])
                self._bin_stop = np.delete(self._bin_stop, [i])
                self._counts = np.delete(self._counts, [i], axis=0)
                self._exposure = np.delete(self._exposure, [i])
                print('Deleted empty time bin', i)
            else:
                i += 1
        # Delete time bins that are outside the interval covered by the poshist file
        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f['GLAST POS HIST'].data['SCLK_UTC']
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]
        # check all time bins if they are outside of this interval
        i = 0
        counter = 0
        while i < len(self._bin_start):
            if self._bin_start[i] < min_time_pos or self._bin_stop[i] > max_time_pos:
                self._bin_start = np.delete(self._bin_start, i)
                self._bin_stop = np.delete(self._bin_stop, i)
                self._counts = np.delete(self._counts, i, 0)
                self._exposure = np.delete(self._exposure, i)
                counter += 1
            else:
                i += 1
        if counter > 0:
            print(str(counter) + ' time bins had to been deleted because they were outside of the time interval covered'
                                 'by the poshist file...')
        self._n_entries = len(self._bin_start)
        self._counts_combined = np.sum(self._counts, axis=1)
        self._counts_combined_mean = np.mean(self._counts_combined)
        self._counts_combined_rate = self._counts_combined / self.time_bin_length
        self._n_time_bins, self._n_channels = self._counts.shape

        self._grb_mask = np.ones(len(self.time_bins), dtype=bool)
        self.n_bins_to_calculate = 800.

        if self._training:
            self.n_bins_to_calculate = 800.

        # Start precomputation of arrays:
        self._setup_geometery()
        self._build_lat_spacecraft()
        self._earth_rate_array()
        self._cgb_rate_array()

        if self._training:
            self._set_grb_mask()
            self._create_rebiner()
            self._rebinned_observed_counts()
            self._prepare_data()
        else:
            self._compute_saa_regions()

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

    @property
    def features(self):
        return self.features

    @property
    def saa_mask(self):

        return self._saa_mask

    def get_quaternion(self, met):

        return self._position_interpolator.quaternion(met)

    def cgb_background(self, time_bins):

        return np.ones_like(time_bins)

    def _compute_saa_regions(self):
        # find where the counts are zero

        min_saa_bin_width = 8
        bins_to_add = 8

        self._zero_idx = self._counts_combined == 0.
        idx = self._zero_idx.nonzero()[0]
        slice_idx = np.array(slice_disjoint(idx))

        # Only the slices which are longer than 8 time bins are used as saa (only for ctime data)
        if self._data_type == 'cspec':
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > 0)]
        else:
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > min_saa_bin_width)]

        # Add bins_to_add to bin_mask to exclude the bins with corrupt data:
        # Check first that the start and stop stop of the mask is not the beginning or end of the day
        slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] = \
            slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] - bins_to_add

        slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] = \
            slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] + bins_to_add

        # now find the times of the exits

        if slice_idx[-1, 1] == self._n_time_bins - 1:
            # the last exit is just the end of the array
            self._saa_exit_idx = slice_idx[:-1, 1]
        else:
            self._saa_exit_idx = slice_idx[:, 1]

        self._saa_exit_mean_times = self.mean_time[self._saa_exit_idx]
        self._saa_exit_bin_start = self._bin_start[self._saa_exit_idx]
        self._saa_exit_bin_stop = self._bin_stop[self._saa_exit_idx]

        # make a saa mask from the slices:
        self._saa_mask = np.ones_like(self._counts_combined, bool)

        for i in range(len(slice_idx)):
            self._saa_mask[slice_idx[i, 0]:slice_idx[i, 1] + 1] = False
            self._zero_idx[slice_idx[i, 0]:slice_idx[i, 1] + 1] = True

        self._saa_slices = slice_idx

    def _setup_geometery(self):
        n_bins_to_calculate = self.n_bins_to_calculate

        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # ok we need to get the sun angle

        n_skip = int(np.ceil(self._n_time_bins / n_bins_to_calculate))

        sun_angle = []
        sun_time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = []  # earth pos in icrs frame (skycoord)

        # additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos = []

        # ps testing
        det_ra = []  # det ra in icrs frame
        det_dec = []  # det dec in icrs frame

        if using_mpi:
            # if using mpi split the times at which the geometry is calculated to all ranks
            list_times_to_calculate = self.mean_time[::n_skip]
            self._times_per_rank = float(len(list_times_to_calculate)) / float(size)
            self._times_lower_bound_index = int(np.floor(rank * self._times_per_rank))
            self._times_upper_bound_index = int(np.floor((rank + 1) * self._times_per_rank))
            with progress_bar(len(list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]),
                              title='Calculating sun and earth position') as p:
                for mean_time in list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]:
                    quaternion_step = self._position_interpolator.quaternion(mean_time)
                    sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                    sun_angle.append(det.sun_angle.value)
                    sun_time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)

                    ra, dec = det.det_ra_dec_icrs
                    det_ra.append(ra)
                    det_dec.append(dec)

                    p.increase()
                # get the last data point with the last rank
            if rank == size - 1:
                mean_time = self.mean_time[-2]
                quaternion_step = self._position_interpolator.quaternion(mean_time)
                sc_pos_step = self._position_interpolator.sc_pos(mean_time)

                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)

                ra, dec = det.det_ra_dec_icrs
                det_ra.append(ra)
                det_dec.append(dec)

            # make the list numpy arrays
            sun_angle = np.array(sun_angle)
            sun_time = np.array(sun_time)
            earth_az = np.array(earth_az)
            earth_zen = np.array(earth_zen)
            earth_position = np.array(earth_position)

            quaternion = np.array(quaternion)
            sc_pos = np.array(sc_pos)
            det_ra = np.array(det_ra)
            det_dec = np.array(det_dec)

            # gather all results in rank=0
            sun_angle_gather = comm.gather(sun_angle, root=0)
            sun_time_gather = comm.gather(sun_time, root=0)
            earth_az_gather = comm.gather(earth_az, root=0)
            earth_zen_gather = comm.gather(earth_zen, root=0)
            earth_position_gather = comm.gather(earth_position, root=0)

            quaternion_gather = comm.gather(quaternion, root=0)
            sc_pos_gather = comm.gather(sc_pos, root=0)
            det_ra_gather = comm.gather(det_ra, root=0)
            det_dec_gather = comm.gather(det_dec, root=0)
            # make one list out of this
            if rank == 0:
                sun_angle_gather = np.concatenate(sun_angle_gather)
                sun_time_gather = np.concatenate(sun_time_gather)
                earth_az_gather = np.concatenate(earth_az_gather)
                earth_zen_gather = np.concatenate(earth_zen_gather)
                earth_position_gather = np.concatenate(earth_position_gather)

                quaternion_gather = np.concatenate(quaternion_gather)
                sc_pos_gather = np.concatenate(sc_pos_gather)
                det_ra_gather = np.concatenate(det_ra_gather)
                det_dec_gather = np.concatenate(det_dec_gather)

            # broadcast the final arrays again to all ranks
            sun_angle = comm.bcast(sun_angle_gather, root=0)
            sun_time = comm.bcast(sun_time_gather, root=0)
            earth_az = comm.bcast(earth_az_gather, root=0)
            earth_zen = comm.bcast(earth_zen_gather, root=0)
            earth_position = comm.bcast(earth_position_gather, root=0)

            quaternion = comm.bcast(quaternion_gather, root=0)
            sc_pos = comm.bcast(sc_pos_gather, root=0)
            det_ra = comm.bcast(det_ra_gather, root=0)
            det_dec = comm.bcast(det_dec_gather, root=0)

        else:
            # go through a subset of times and calculate the sun angle with GBM geometry

            ###SINGLECORE CALC###
            with progress_bar(n_bins_to_calculate, title='Calculating sun and earth position') as p:

                for mean_time in self.mean_time[::n_skip]:
                    quaternion_step = self._position_interpolator.quaternion(mean_time)
                    sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                    sun_angle.append(det.sun_angle.value)
                    sun_time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)

                    # test
                    ra, dec = det.det_ra_dec_icrs
                    det_ra.append(ra)
                    det_dec.append(dec)

                    p.increase()

            # get the last data point

            mean_time = self.mean_time[-2]

            quaternion_step = self._position_interpolator.quaternion(mean_time)
            sc_pos_step = self._position_interpolator.sc_pos(mean_time)
            det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                               sc_pos=sc_pos_step,
                                               time=astro_time.Time(self._position_interpolator.utc(mean_time)))

            sun_angle.append(det.sun_angle.value)
            sun_time.append(mean_time)
            az, zen = det.earth_az_zen_sat
            earth_az.append(az)
            earth_zen.append(zen)
            earth_position.append(det.earth_position)

            quaternion.append(quaternion_step)
            sc_pos.append(sc_pos_step)

            # test
            ra, dec = det.det_ra_dec_icrs
            det_ra.append(ra)
            det_dec.append(dec)

        self._sun_angle = sun_angle
        self._sun_time = sun_time
        self._earth_az = earth_az
        self._earth_zen = earth_zen
        self._earth_position = earth_position

        self._quaternion = quaternion
        self._sc_pos = sc_pos

        self._det_ra = np.array(det_ra)
        self._det_dec = np.array(det_dec)

        quaternion_array = np.array(quaternion)
        sc_array = np.array(sc_pos)
        sc_height_array = np.sqrt((sc_array ** 2).sum(axis=1))

        # interpolate it
        self._q0_interpolator = interpolate.interp1d(self._sun_time, quaternion_array[:, 0])
        self._q1_interpolator = interpolate.interp1d(self._sun_time, quaternion_array[:, 1])
        self._q2_interpolator = interpolate.interp1d(self._sun_time, quaternion_array[:, 2])
        self._q3_interpolator = interpolate.interp1d(self._sun_time, quaternion_array[:, 3])
        self._sc0_interpolator = interpolate.interp1d(self._sun_time, sc_array[:, 0])
        self._sc1_interpolator = interpolate.interp1d(self._sun_time, sc_array[:, 1])
        self._sc2_interpolator = interpolate.interp1d(self._sun_time, sc_array[:, 2])
        self._sc_height_interpolator = interpolate.interp1d(self._sun_time, sc_height_array)

        self._earth_az_interpolator = interpolate.interp1d(self._sun_time, self._earth_az)
        self._earth_zen_interpolator = interpolate.interp1d(self._sun_time, self._earth_zen)
        self._sun_angle_interpolator = interpolate.interp1d(self._sun_time, self._sun_angle)

        self._det_ra_interpolator = interpolate.interp1d(self._sun_time, self._det_ra)
        self._det_dec_interpolator = interpolate.interp1d(self._sun_time, self._det_dec)

        del sun_angle, sun_time, earth_az, earth_zen, quaternion_array, sc_array

    def _build_lat_spacecraft(self):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file
        mission_week = self.mission_week

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

                if not file_existing_and_readable(after_filepath):
                    download_lat_spacecraft(mission_week + 1)

            # first lets get the primary file

            lat_time = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
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

        # remove the self-variables for memory saving
        self._mc_b_interpolator = interpolate.interp1d(lat_time, mc_b)
        self._mc_l_interpolator = interpolate.interp1d(lat_time, mc_l)

        del mc_l, mc_b, lat_time

    def _earth_rate_array(self):
        """
        Calculate the earth_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the earth_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the earth
        albedo.
        :return:
        """
        print("Calculate Earth rate array")
        points = self._rate_generator_DRM.points
        earth_rates = self._rate_generator_DRM.earth_rate
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
            else:
                upper_index = self._times_upper_bound_index

            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth < opening_angle_earth:
                        earth_rate += earth_rates[i]
                array_earth_rate.append(earth_rate)

            array_earth_rate = np.array(array_earth_rate)

            array_earth_rate_g = comm.gather(array_earth_rate, root=0)
            if rank == 0:
                array_earth_rate_g = np.concatenate(array_earth_rate_g)
            array_earth_rate = comm.bcast(array_earth_rate_g, root=0)
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth < opening_angle_earth:
                        earth_rate += earth_rates[i]
                array_earth_rate.append(earth_rate)
        self._array_earth_rate = np.array(array_earth_rate).T
        self._earth_rate_interpolator = interpolate.interp1d(self._sun_time, self._array_earth_rate)

    def _cgb_rate_array(self):
        """
        Calculate the cgb_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the cgb_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are not occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the cgb
        spectrum.
        :return:
        """
        print("Calculate CGB rate array")
        points = self._rate_generator_DRM.points
        cgb_rates = self._rate_generator_DRM.cgb_rate
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
            else:
                upper_index = self._times_upper_bound_index

            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_rate = []
            for pos in self._earth_pos_inter_times:
                cgb_rate = np.zeros_like(cgb_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_rate += cgb_rates[i]
                array_cgb_rate.append(cgb_rate)
            array_cgb_rate = np.array(array_cgb_rate)

            array_cgb_rate_g = comm.gather(array_cgb_rate, root=0)
            if rank == 0:
                array_cgb_rate_g = np.concatenate(array_cgb_rate_g)
            array_cgb_rate = comm.bcast(array_cgb_rate_g, root=0)
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_rate = []
            for pos in self._earth_pos_inter_times:
                cgb_rate = np.zeros_like(cgb_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_rate += cgb_rates[i]
                array_cgb_rate.append(cgb_rate)
        self._array_cgb_rate = np.array(array_cgb_rate).T
        self._cgb_rate_interpolator = interpolate.interp1d(self._sun_time, self._array_cgb_rate)

    def calc_features(self, mean_times, echan):
        return np.stack((
            self._det_ra_interpolator(mean_times),
            self._det_dec_interpolator(mean_times),

            self._sc0_interpolator(mean_times),
            self._sc1_interpolator(mean_times),
            self._sc2_interpolator(mean_times),
            self._sc_height_interpolator(mean_times),
            self._q0_interpolator(mean_times),
            self._q1_interpolator(mean_times),
            self._q2_interpolator(mean_times),
            self._q3_interpolator(mean_times),

            self._earth_az_interpolator(mean_times),
            self._earth_zen_interpolator(mean_times),
            self._sun_angle_interpolator(mean_times),
            self._earth_rate_interpolator(mean_times)[echan],
            self._cgb_rate_interpolator(mean_times)[echan],
            self._mc_l_interpolator(mean_times),
            self.mission_week * np.ones(len(self.rebinned_mean_times))
        ), axis=1)

    def _set_grb_mask(self):
        """
        Mask known GRB intervals provided as an array
        """
        for trigger in self._trigger_intervals:
            bin_exclude = np.logical_and(self.time_bins[:, 0] > trigger[0],
                                         self.time_bins[:, 1] < trigger[1])
            self._grb_mask[np.where(bin_exclude)] = False

    def _create_rebiner(self):
        self._rebinner = Rebinner(self.time_bins[np.where(self._grb_mask)], self.min_bin_width)

    def _rebinned_observed_counts(self):
        masked_counts = self.counts[np.where(self._grb_mask)]

        self.rebinned_time_bins = self._rebinner.time_rebinned[2:-2]
        self.rebinned_time_bin_widths = np.diff(self.rebinned_time_bins, axis=1)[:, 0]
        self.rebinned_mean_times = np.mean(self.rebinned_time_bins, axis=1)

        self.rebinned_counts_0, = self._rebinner.rebin(masked_counts[:, 0])
        self.rebinned_counts_1, = self._rebinner.rebin(masked_counts[:, 1])
        self.rebinned_counts_2, = self._rebinner.rebin(masked_counts[:, 2])
        self.rebinned_counts_3, = self._rebinner.rebin(masked_counts[:, 3])
        self.rebinned_counts_4, = self._rebinner.rebin(masked_counts[:, 4])
        self.rebinned_counts_5, = self._rebinner.rebin(masked_counts[:, 5])
        self.rebinned_counts_6, = self._rebinner.rebin(masked_counts[:, 6])
        self.rebinned_counts_7, = self._rebinner.rebin(masked_counts[:, 7])

        self.rebinned_counts_0 = self.rebinned_counts_0[2:-2]
        self.rebinned_counts_1 = self.rebinned_counts_1[2:-2]
        self.rebinned_counts_2 = self.rebinned_counts_2[2:-2]
        self.rebinned_counts_3 = self.rebinned_counts_3[2:-2]
        self.rebinned_counts_4 = self.rebinned_counts_4[2:-2]
        self.rebinned_counts_5 = self.rebinned_counts_5[2:-2]
        self.rebinned_counts_6 = self.rebinned_counts_6[2:-2]
        self.rebinned_counts_7 = self.rebinned_counts_7[2:-2]

        self.rebinned_count_rates_0 = self.rebinned_counts_0 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_1 = self.rebinned_counts_1 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_2 = self.rebinned_counts_2 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_3 = self.rebinned_counts_3 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_4 = self.rebinned_counts_4 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_5 = self.rebinned_counts_5 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_6 = self.rebinned_counts_6 / self.rebinned_time_bin_widths
        self.rebinned_count_rates_7 = self.rebinned_counts_7 / self.rebinned_time_bin_widths

    def _prepare_data(self):
        self.rebinned_features = np.stack((
            self._det_ra_interpolator(self.rebinned_mean_times),
            self._det_dec_interpolator(self.rebinned_mean_times),

            self._sc0_interpolator(self.rebinned_mean_times),
            self._sc1_interpolator(self.rebinned_mean_times),
            self._sc2_interpolator(self.rebinned_mean_times),
            self._sc_height_interpolator(self.rebinned_mean_times),
            self._q0_interpolator(self.rebinned_mean_times),
            self._q1_interpolator(self.rebinned_mean_times),
            self._q2_interpolator(self.rebinned_mean_times),
            self._q3_interpolator(self.rebinned_mean_times),

            self._earth_az_interpolator(self.rebinned_mean_times),
            self._earth_zen_interpolator(self.rebinned_mean_times),
            self._sun_angle_interpolator(self.rebinned_mean_times),

            self._earth_rate_interpolator(self.rebinned_mean_times)[0],
            self._earth_rate_interpolator(self.rebinned_mean_times)[1],
            self._earth_rate_interpolator(self.rebinned_mean_times)[2],
            self._earth_rate_interpolator(self.rebinned_mean_times)[3],
            self._earth_rate_interpolator(self.rebinned_mean_times)[4],
            self._earth_rate_interpolator(self.rebinned_mean_times)[5],
            self._earth_rate_interpolator(self.rebinned_mean_times)[6],
            self._earth_rate_interpolator(self.rebinned_mean_times)[7],

            self._cgb_rate_interpolator(self.rebinned_mean_times)[0],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[1],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[2],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[3],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[4],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[5],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[6],
            self._cgb_rate_interpolator(self.rebinned_mean_times)[7],

            self._mc_l_interpolator(self.rebinned_mean_times),
            self.mission_week * np.ones_like(self.rebinned_mean_times),
        ), axis=1)

        self.rebinned_counts = np.stack((
            self.rebinned_counts_0,
            self.rebinned_counts_1,
            self.rebinned_counts_2,
            self.rebinned_counts_3,
            self.rebinned_counts_4,
            self.rebinned_counts_5,
            self.rebinned_counts_6,
            self.rebinned_counts_7,
        ), axis=1)

        self.rebinned_count_rates = np.stack((
            self.rebinned_count_rates_0,
            self.rebinned_count_rates_1,
            self.rebinned_count_rates_2,
            self.rebinned_count_rates_3,
            self.rebinned_count_rates_4,
            self.rebinned_count_rates_5,
            self.rebinned_count_rates_6,
            self.rebinned_count_rates_7,
        ), axis=1)

    def save_data(self, filename=None):
        if filename is None:
            filename = "cleaned_data_{}_{}.npz".format(self._day, self._det)

        if os.path.isfile(filename):
            raise Exception("Error: output file already exists")

        np.savez_compressed(filename,
                            counts=self.rebinned_counts,
                            count_rates=self.rebinned_count_rates,
                            features=self.rebinned_features,
                            feature_labels=[
                                'det_ra', 'det_dec', 'sc0', 'sc1', 'sc2', 'sc_height', 'q0', 'q1', 'q2', 'q3',
                                'earth_az', 'earth_zen', 'sun_angle', 'mc_l', 'mission_week',
                                'earth_rate_0', 'earth_rate_1', 'earth_rate_2', 'earth_rate_3', 'earth_rate_4', 'earth_rate_5', 'earth_rate_6', 'earth_rate_7',
                                'cgb_rate_0', 'cgb_rate_1', 'cgb_rate_2', 'cgb_rate_3', 'cgb_rate_4', 'cgb_rate_5', 'cgb_rate_6', 'cgb_rate_7',
                            ])

    @property
    def feature_labels(self):
        return [
            'det_ra', 'det_dec', 'sc0', 'sc1', 'sc2', 'sc_height', 'q0', 'q1', 'q2', 'q3',
            'earth_az', 'earth_zen', 'sun_angle', 'mc_l', 'mission_week',
            'earth_rate_0', 'earth_rate_1', 'earth_rate_2', 'earth_rate_3', 'earth_rate_4', 'earth_rate_5', 'earth_rate_6', 'earth_rate_7',
            'cgb_rate_0', 'cgb_rate_1', 'cgb_rate_2', 'cgb_rate_3', 'cgb_rate_4', 'cgb_rate_5', 'cgb_rate_6', 'cgb_rate_7',
        ]

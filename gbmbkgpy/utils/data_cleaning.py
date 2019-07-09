import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list

import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

import astropy.time as astro_time
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

    def __init__(self, date, detector, data_type, min_bin_width=None, training=False, trigger_intervals=[]):
        self._data_type = data_type
        self._det = detector
        self._day = date

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

        # Start precomputation of arrays:
        self._setup_geometery()

        if self._training:
            self._set_grb_mask()
            self._create_rebiner()
            self._rebinned_observed_counts()
            self._prepare_data()
        else:
            self._compute_saa_regions()

        # Calculate the MET time for the day
        day = self._day
        year = '20%s' % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        self._day_met = GBMTime(day_at).met

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
        n_bins_to_calculate = 800.

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

    def calc_features(self, mean_times):
        return np.stack((
            self._det_ra_interpolator(mean_times),
            self._det_dec_interpolator(mean_times),
            self._sc0_interpolator(mean_times),
            self._sc1_interpolator(mean_times),
            self._sc2_interpolator(mean_times),
            self._sc_height_interpolator(mean_times),
            self._earth_az_interpolator(mean_times),
            self._earth_zen_interpolator(mean_times),
            self._sun_angle_interpolator(mean_times),
            self._q0_interpolator(mean_times),
            self._q1_interpolator(mean_times),
            self._q2_interpolator(mean_times),
            self._q3_interpolator(mean_times)
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
            self._earth_az_interpolator(self.rebinned_mean_times),
            self._earth_zen_interpolator(self.rebinned_mean_times),
            self._sun_angle_interpolator(self.rebinned_mean_times),

            self._q0_interpolator(self.rebinned_mean_times),
            self._q1_interpolator(self.rebinned_mean_times),
            self._q2_interpolator(self.rebinned_mean_times),
            self._q3_interpolator(self.rebinned_mean_times)
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

    def save_data(self):

        filename = "cleaned_data_{}_{}.npz".format(self._day, self._det)

        if os.path.isfile(filename):
            raise Exception("Error: output file already exists")

        np.savez_compressed(filename,
                            counts=self.rebinned_counts,
                            count_rates=self.rebinned_count_rates,
                            features=self.rebinned_features)
        return

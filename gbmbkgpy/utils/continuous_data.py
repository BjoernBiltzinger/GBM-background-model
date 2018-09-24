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

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


class ContinuousData(object):

    def __init__(self, date, detector, data_type, rate_generator_DRM, use_SAA=True):
        self._data_type = data_type
        self._det = detector
        self._day = date
        self._use_SAA = use_SAA
        self._rate_generator_DRM = rate_generator_DRM
        #assert 'ctime' in self._data_type, 'currently only working for CTIME data'
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
                self._exposure=np.delete(self._exposure, [i])
                print('Deleted empty time bin', i)
            else:
                i+=1
        # Delete time bins that are outside the interval covered by the poshist file
        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f['GLAST POS HIST'].data['SCLK_UTC']
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]
        # check all time bins if they are outside of this interval
        i=0
        counter=0
        while i<len(self._bin_start):
            if self._bin_start[i]<min_time_pos or self._bin_stop[i]>max_time_pos:
                self._bin_start = np.delete(self._bin_start, i)
                self._bin_stop = np.delete(self._bin_stop, i)
                self._counts = np.delete(self._counts, i, 0)
                self._exposure = np.delete(self._exposure, i)
                counter+=1
            else:
                i+=1
        if counter>0:
            print(str(counter) + ' time bins had to been deleted because they were outside of the time interval covered'
                                 'by the poshist file...')
        self._n_entries = len(self._bin_start)
        self._counts_combined = np.sum(self._counts, axis=1)
        self._counts_combined_rate = self._counts_combined / self.time_bin_length
        self._n_time_bins, self._n_channels = self._counts.shape
        # Start precomputation of arrays:
        self._setup_geometery()
        self._compute_saa_regions()
        self._earth_rate_array()
        self._cgb_rate_array()

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


    def get_quaternion(self, met):

        return self._position_interpolator.quaternion(met)

    def cgb_background(self, time_bins):

        return np.ones_like(time_bins)

    def _setup_geometery(self):
        n_bins_to_calculate = 80.

        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # ok we need to get the sun angle

        n_skip = int(np.ceil(self._n_time_bins / n_bins_to_calculate))

        sun_angle = []
        sun_time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = [] #earth pos in icrs frame (skycoord)

        #additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos =[]

        #ps testing
        det_ra = [] #det ra in icrs frame
        det_dec = [] #det dec in icrs frame

        if using_mpi:
            #if using mpi split the times at which the geometry is calculated to all ranks
            list_times_to_calculate = self.mean_time[::n_skip]
            self._times_per_rank = float(len(list_times_to_calculate))/float(size)
            self._times_lower_bound_index = int(np.floor(rank*self._times_per_rank))
            self._times_upper_bound_index = int(np.floor((rank+1)*self._times_per_rank))
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

                    p.increase()
                #get the last data point with the last rank
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
            #make the list numpy arrays
            sun_angle = np.array(sun_angle)
            sun_time = np.array(sun_time)
            earth_az = np.array(earth_az)
            earth_zen = np.array(earth_zen)
            earth_position = np.array(earth_position)

            quaternion = np.array(quaternion)
            sc_pos = np.array(sc_pos)
            #gather all results in rank=0
            sun_angle_gather = comm.gather(sun_angle, root=0)
            sun_time_gather = comm.gather(sun_time, root=0)
            earth_az_gather = comm.gather(earth_az, root=0)
            earth_zen_gather = comm.gather(earth_zen, root=0)
            earth_position_gather = comm.gather(earth_position, root=0)


            quaternion_gather = comm.gather(quaternion, root=0)
            sc_pos_gather = comm.gather(sc_pos, root=0)
            #make one list out of this
            if rank == 0:
                sun_angle_gather = np.concatenate(sun_angle_gather)
                sun_time_gather = np.concatenate(sun_time_gather)
                earth_az_gather = np.concatenate(earth_az_gather)
                earth_zen_gather = np.concatenate(earth_zen_gather)
                earth_position_gather = np.concatenate(earth_position_gather)

                quaternion_gather=np.concatenate(quaternion_gather)
                sc_pos_gather = np.concatenate(sc_pos_gather)
            #broadcast the final arrays again to all ranks
            sun_angle = comm.bcast(sun_angle_gather, root=0)
            sun_time = comm.bcast(sun_time_gather, root=0)
            earth_az = comm.bcast(earth_az_gather, root=0)
            earth_zen = comm.bcast(earth_zen_gather, root=0)
            earth_position = comm.bcast(earth_position_gather, root=0)

            quaternion = comm.bcast(quaternion_gather, root=0)
            sc_pos = comm.bcast(sc_pos_gather, root=0)

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

                    #test
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

        #test
        self._det_ra = np.array(det_ra)
        self._det_dec = np.array(det_dec)
        # interpolate it

        self._sun_angle_interpolator = interpolate.interp1d(self._sun_time, self._sun_angle)

        del sun_angle, sun_time, earth_az, earth_zen

    def _compute_saa_regions(self):

        # find where the counts are zero

        min_saa_bin_width = 8
        bins_to_add = 8

        self._zero_idx = self._counts_combined == 0.
        idx = (self._zero_idx).nonzero()[0]
        slice_idx = np.array(slice_disjoint(idx))

        # Only the slices which are longer than 8 time bins are used as saa (only for ctime data)
        if self._data_type=='cspec':
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > 0)]
        else:
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > min_saa_bin_width)]


        # Add bins_to_add to bin_mask to exclude the bins with corrupt data:
        # Check first that the start and stop stop of the mask is not the beginning or end of the day
        slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] =\
            slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] - bins_to_add

        slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] =\
            slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] + bins_to_add



        # now find the times of the exits

        if slice_idx[-1 , 1] == self._n_time_bins - 1:

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


        # deleting 5000s after every saa exit => ignore saa's
        if not self._use_SAA:
            if self._bin_stop[slice_idx[0, 0]] - 5000 > self._bin_start[0]:
                self._saa_mask[0:slice_idx[0, 0] + 1] = False
                self._zero_idx[0:slice_idx[0, 0] + 1] = True
            else:
                j = 0
                while 5000 > self._bin_start[j] - self._bin_start[0]:
                    j += 1
                self._saa_mask[0:j] = False
                self._zero_idx[0:j] = True

            for i in range(len(slice_idx) - 1):
                if self._bin_stop[slice_idx[i + 1, 0]] - self._bin_start[slice_idx[i, 1]] < 5000:
                    self._saa_mask[slice_idx[i, 1]:slice_idx[i + 1, 0]] = False
                    self._zero_idx[slice_idx[i, 1]:slice_idx[i + 1, 0]] = True
                else:
                    j = 0
                    while self._bin_start[slice_idx[i, 1]] + 5000 > self._bin_start[slice_idx[i, 1] + j]:
                        j += 1
                    self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
                    self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True

            if self._bin_stop[slice_idx[-1, 1]] + 5000 > self._bin_stop[-1]:
                self._saa_mask[slice_idx[-1, 1]:len(self._counts_combined) + 1] = False
                self._zero_idx[slice_idx[-1, 1]:len(self._counts_combined) + 1] = True
            else:
                j = 0
                while self._bin_start[slice_idx[-1, 1]] + 5000 > self._bin_start[slice_idx[-1, 1] + j]:
                    j += 1
                self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
                self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True


        self._saa_slices = slice_idx


    @property
    def quaternion(self):

        return self._quaternion

    @property
    def sc_pos(self):

        return self._sc_pos

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

        return self._saa_mask


    @property
    def saa_mean_times(self):

        return self._saa_exit_mean_times


    def saa_initial_values(self, echan):

        start_value_array = []

        # Add mean of first 10 time bins for leftover decay from day before
        start_value_array.append(np.mean(self._counts[0:11, echan] / self.time_bin_length[0:11]))

        for i, exit_idx in enumerate(self._saa_exit_idx):
            start_value_array.append(
                np.mean(self._counts[exit_idx:exit_idx+10, echan] / self.time_bin_length[exit_idx:exit_idx+10]))

        return np.array(start_value_array)


    def saa_initial_decay_constants(self, echan):

        amplitudes_t0 = self.saa_initial_values(echan)

        amplitudes_t1 = []

        # Add mean of first 10 time bins for leftover decay from day before
        amplitudes_t1.append(np.mean(self._counts[100:121, echan] / self.time_bin_length[100:121]))

        for i, exit_idx in enumerate(self._saa_exit_idx):
            amplitudes_t1.append(
                np.mean(self._counts[exit_idx + 100: exit_idx + 120, echan] /
                        self.time_bin_length[exit_idx + 100: exit_idx + 120]))

        initial_decay_constants = (np.log(amplitudes_t1) - np.log(amplitudes_t0)) /\
                                  (self.mean_time[exit_idx + 100] - self.mean_time[exit_idx])

        # Replace positive values
        initial_decay_constants[np.where(initial_decay_constants > 0)] = \
            initial_decay_constants[np.where(initial_decay_constants > 0)] * -1.

        return np.array(initial_decay_constants)


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

    @property
    def use_SAA(self):

        return self._use_SAA

    def _earth_rate_array(self):
        """
        Calculate the earth_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the earth_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the earth
        albedo.
        :return:
        """
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
                upper_index = self._times_upper_bound_index + 1

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
                upper_index = self._times_upper_bound_index + 1

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


    def cgb_rate_array(self, met):
        """
        Interpolation function for the CGB continuum rate in a certain Ebin
        :param met: times at which to interpolate
        :return: array with the CGB rates expected over whole day in a certain Ebin
        """

        return self._cgb_rate_interpolator(met)

    def earth_rate_array(self, met):
        """
        Interpolation function for the Earth continuum rate in a certain Ebin
        :param met: times at which to interpolate
        :return: array with the Earth rates expected over whole day in a certain Ebin
        """

        return self._earth_rate_interpolator(met)

    @property
    def cgb_rate_interpolation_time(self):
        return self._array_cgb_rate

    @property
    def earth_rate_interpolation_time(self):
        return self._array_earth_rate

    @property
    def earth_az_interpolation_time(self):
        return self._earth_az

    @property
    def earth_zen_interpolation_time(self):
        return self._earth_zen

    @property
    def earth_pos_interpolation_time(self):
        return self._earth_pos_inter_times

    @property
    def saa_slices(self):
        return self._saa_slices

    @property
    def rate_generator_DRM(self):
        return self._rate_generator_DRM

    #test
    @property
    def det_ra_icrs(self):
        return self._det_ra

    @property
    def det_dec_icrs(self):
        return self._det_dec

    @property
    def times_lower_bound_index(self):
        """
        :return: the lower bound index of the part of the interpolation list covered by this rank
        """
        return self._times_lower_bound_index

    @property
    def times_upper_bound_index(self):
        """
        :return: the upper bound index of the part of the interpolation list covered by this rank
        """
        return self._times_upper_bound_index

import astropy.io.fits as fits

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

import astropy.time as astro_time

import numpy as np

import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
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

valid_det_names = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']


class Data(object):

    def __init__(self, date, detector, data_type, echan_list, test=False):
        """
        Initalize the ContinousData Class, which contains the information about the time bins 
        and counts of the data.
        """

        assert data_type == 'ctime' or data_type == 'cspec', 'Please use a valid data type: ctime or cspec'
        assert detector in valid_det_names, 'Please use a valid det name. One of these: {}'.format(valid_det_names)
        assert type(date) == list and len(date[0]) == 6, 'Date variable has to be a list and every entry must have ' \
                                                         'the format YYMMDD'
        if data_type == 'ctime':
            assert type(echan_list) and max(echan_list) <= 7 and min(echan_list) >= 0 \
                   and all(isinstance(x, int) for x in echan_list), 'Echan_list variable must be a list and can only ' \
                                                                    'have integer entries between 0 and 7'

        if data_type == 'cspec':
            assert type(echan_list) and max(echan_list) <= 127 and min(echan_list) >= 0 \
                   and all(isinstance(x, int) for x in echan_list), 'Echan_list variable must be a list and can only ' \
                                                                    'have integer entries between 0 and 127'

        self._data_type = data_type
        self._det = detector
        self._day_list = sorted(date)
        self._echan_list = echan_list
        self._test = test
        if self._data_type == 'ctime':
            self._echan_mask = np.zeros(8, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        elif self._data_type == 'cspec':
            self._echan_mask = np.zeros(128, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True

        self._build_arrays()

        self._rebinned = False
        self._data_rebinner = None
        self._rebinned_counts = None
        self._rebinned_time_bins = None
        self._rebinned_saa_mask = None

    def rebinn_data(self, min_bin_width, saa_mask):
        """
        Rebins the time bins to a min bin width
        :param min_bin_width:
        :return:
        """
        self._rebinned = True

        self._data_rebinner = Rebinner(self._time_bins, min_bin_width, mask=saa_mask)

        for i, echan in enumerate(self._echan_list):
            if i == 0:
                count_array_tmp = self._data_rebinner.rebin(self._counts[:, i])[0]
                count_array = np.zeros((len(count_array_tmp), len(self._echan_list)))
                count_array[:, i] = count_array_tmp
            else:
                count_array[:, i] = self._data_rebinner.rebin(self._counts[:, i])[0]

        self._rebinned_counts = count_array.astype(np.uint16)

        self._rebinned_time_bins = self._data_rebinner.time_rebinned

        self._rebinned_saa_mask = self._data_rebinner.rebinned_saa_mask

    @property
    def counts(self):
        """
        Returns the count information of all time bins
        :return:
        """
        if self._rebinned:
            return self._rebinned_counts
        else:
            return self._counts

    @property
    def time_bins(self):
        """
        Returns the time bin information of all time bins
        :return:
        """
        if self._rebinned:
            return self._rebinned_time_bins
        else:
            return self._time_bins

    @property
    def rebinned_saa_mask(self):
        if self._rebinned:
            return self._rebinned_saa_mask
        else:
            raise Exception('Data is unbinned, the saa mask has to be obtained from the SAA_calc object')

    @property
    def det(self):
        """
        Returns the used detector
        :return:
        """
        return self._det
            
    @property
    def day_start_times(self):
        """
        Returns the start time of the day
        :return:
        """
        return self._day_start_times

    @property
    def day_stop_times(self):
        """
        Returns the stop time of the day
        :return:
        """
        return self._day_stop_times

    @property
    def day_met(self):
        """
        Returns array with the day_met of all used days
        :return:
        """
        return self._day_met_array

    @property
    def following_day(self):
        """
        Returns array which gives which of the days are following days (len(self._following_day)=num_days-1)
        :return:
        """
        return self._following_day

    @property
    def day(self):
        """
        Returns day_list
        :return:
        """
        return self._day_list

    @property
    def data_type(self):
        """
        Returns used data_type
        :return:
        """
        return self._data_type

    @property
    def detector_id(self):
        """
        Returns detector number
        :return:
        """
        return self._det[-1]

    @property
    def time_bin_width(self):
        """
        Returns width of the time bins
        :return:
        """
        if self._rebinned:
            return np.diff(self._rebinned_time_bins, axis=1)[:, 0]
        else:
            return np.diff(self._time_bins, axis=1)[:, 0]

    @property
    def mean_time(self):
        """
        Returns mean time of the time bins
        :return:
        """
        if self._rebinned:
            return np.mean(self._rebinned_time_bins, axis=1)
        else:
            return np.mean(self._time_bins, axis=1)

    def _build_arrays(self):
        """
        Iterates over all wanted days and adds the count and time bin information in one big array
        :return:
        """
        following_day = np.array([])
        for i, day in enumerate(self._day_list):
            counts, time_bins, day_met = self._one_day_data(day)
            if i == 0:
                count_array = counts
                time_bins_array = time_bins
                day_met_array = np.array([day_met])
                day_start_times = np.array([time_bins[0, 0]])
                day_stop_times = np.array([time_bins[-1, 1]])
            else:
                j = 0
                for j in range(len(counts)):
                    if time_bins[j, 0] > time_bins_array[-1, 1]:
                        if time_bins_array[j, 0] < time_bins[-1, 1] + 1000:
                            following_day = np.append(following_day, True)
                        else:
                            following_day = np.append(following_day, False)
                        break

                count_array = np.append(count_array, counts[j:], axis=0)
                time_bins_array = np.append(time_bins_array, time_bins[j:], axis=0)
                day_start_times = np.append(day_start_times, time_bins[j, 0])
                day_stop_times = np.append(day_stop_times, time_bins[-1, 1])
                day_met_array = np.append(day_met_array, day_met)
        self._counts = count_array
        self._time_bins = time_bins_array
        self._day_start_times = day_start_times
        self._day_stop_times = day_stop_times
        self._day_met_array = day_met_array
        self._following_day = following_day

    def _one_day_data(self, day):
        """
        Prepares the data for one day
        :param day:
        :return:
        """
        version = 'v00' if not self._test else 'test'

        # Download data-file and poshist file if not existing:
        datafile_name = 'glg_{0}_{1}_{2}_{3}.pha'.format(self._data_type, self._det, day, version)
        datafile_path = os.path.join(get_path_of_external_data_dir(), self._data_type, day, datafile_name)

        poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', day)
        poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)

        # If MPI is used only one rank should download the data; the others wait
        if using_mpi:
            if rank == 0:
                if not file_existing_and_readable(datafile_path):
                    download_data_file(day, self._data_type, self._det)

                if not file_existing_and_readable(poshistfile_path):
                    download_data_file(day, 'poshist')
            comm.Barrier()
        else:
            if not file_existing_and_readable(datafile_path):
                download_data_file(day, self._data_type, self._det)

            if not file_existing_and_readable(poshistfile_path):
                download_data_file(day, 'poshist')

        # Save poshistfile_path for later usage
        self._pos_hist = poshistfile_path

        # Open the datafile of the CTIME/CSPEC data and read in all needed quantities
        with fits.open(datafile_path) as f:
            counts = f['SPECTRUM'].data['COUNTS']
            bin_start = f['SPECTRUM'].data['TIME']
            bin_stop = f['SPECTRUM'].data['ENDTIME']

        # Sometimes there are corrupt time bins where the time bin start = time bin stop
        # So we have to delete these times bins
        idx_zero_bins = np.where(bin_start==bin_stop)[0]

        bin_start = np.delete(bin_start, idx_zero_bins)
        bin_stop = np.delete(bin_stop, idx_zero_bins)
        counts = np.delete(counts, idx_zero_bins, axis=0)

        # Sometimes the poshist file does not cover the whole time covered by the CTIME/CSPEC file.
        # So we have to delete these time bins 

        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f['GLAST POS HIST'].data['SCLK_UTC']
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]

        # check for all time bins if they are outside of this interval
        idx_below_min_time = np.where(bin_start < min_time_pos)
        idx_above_max_time = np.where(bin_start > max_time_pos)
        idx_out_of_bounds =  np.unique(np.hstack((idx_below_min_time, idx_above_max_time)))

        bin_start = np.delete(bin_start, idx_out_of_bounds)
        bin_stop = np.delete(bin_stop, idx_out_of_bounds)
        counts = np.delete(counts, idx_out_of_bounds, axis=0)

        # Calculate the MET time for the day
        day = day
        year = '20%s' % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        day_met = GBMTime(day_at).met
        # Get time bins
        time_bins = np.vstack((bin_start, bin_stop)).T

        # Only keep the count informations we need for the echan's we want to fit
        counts = counts.T[self._echan_mask].T

        return counts, time_bins, day_met

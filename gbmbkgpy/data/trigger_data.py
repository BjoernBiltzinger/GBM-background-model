import astropy.io.fits as fits

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_trigdata_file

import numpy as np

import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
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
        rank = 0
except:
    using_mpi = False
    rank = 0

valid_det_names = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']


class TrigData(object):

    def __init__(self, trigger, detector, data_type, echan_list):
        """
        Initalize the TrigData Class, which contains the information about the time bins
        and counts of the data.
        """

        assert data_type == 'trigdat' 'Please use a valid data type: trigdat'
        assert detector in valid_det_names, 'Please use a valid det name. One of these: {}'.format(valid_det_names)

        assert len(trigger) == 9, 'Please provide a valid trigger in the format YYMMDDxxx'

        if data_type == 'trigdat':
            assert type(echan_list) and max(echan_list) <= 7 and min(echan_list) >= 0 \
                   and all(isinstance(x, int) for x in echan_list), 'Echan_list variable must be a list and can only ' \
                                                                    'have integer entries between 0 and 7'

        self._data_type = data_type
        self._det = detector
        self._trigger = trigger
        self._echan_list = echan_list

        if self._data_type == 'trigdat':
            self._echan_mask = np.zeros(8, dtype=bool)
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
    def data_type(self):
        """
        Returns used data_type
        :return:
        """
        return self._data_type

    @property
    def trigger(self):
        """
        Returns trigger
        :return:
        """
        return self._trigger

    @property
    def trigdata_path(self):
        """
        Returns trigger
        :return:
        """
        return self._trigdata_path

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

        # Download data-file and poshist file if not existing:
        datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(self._data_type, 'all', self._trigger)
        datafile_path = os.path.join(get_path_of_external_data_dir(), self._data_type, self._trigger, datafile_name)

        # If MPI is used only one rank should download the data; the others wait
        if rank == 0:
            if not file_existing_and_readable(datafile_path):
                download_trigdata_file(self._trigger, self._data_type)

        if using_mpi:
            comm.Barrier()

        self._trigdata_path = datafile_path

        # Open the datafile of the CTIME/CSPEC data and read in all needed quantities
        with fits.open(datafile_path) as f:
            counts = f['SPECTRUM'].data['COUNTS']
            bin_start = f['SPECTRUM'].data['TIME']
            bin_stop = f['SPECTRUM'].data['ENDTIME']

        # Sometimes there are corrupt time bins where the time bin start = time bin stop
        # So we have to delete these times bins
        i = 0
        while i < len(bin_start):
            if bin_start[i] == bin_stop[i]:
                bin_start = np.delete(bin_start, [i])
                bin_stop = np.delete(bin_stop, [i])
                counts = np.delete(counts, [i], axis=0)
            else:
                i += 1

        # Get time bins
        time_bins = np.vstack((bin_start, bin_stop)).T

        # Only keep the count informations we need for the echan's we want to fit
        counts = counts.T[self._echan_mask].T

        self._counts = counts
        self._time_bins = time_bins
        self._day_start_times = [time_bins[0, 0]]
        self._day_stop_times = [time_bins[-1, 1]]



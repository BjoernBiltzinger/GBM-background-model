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

valid_det_names = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]


class TrigData(object):
    def __init__(self, trigger, detector, data_type, echan_list, test=False):
        """
        Initalize the TrigData Class, which contains the information about the time bins
        and counts of the data.
        """
        assert data_type == "trigdat", "Please use a valid data type: trigdat"
        assert (
            detector in valid_det_names
        ), "Please use a valid det name. One of these: {}".format(valid_det_names)

        assert (
            len(trigger) == 11
        ), "Please provide a valid trigger in the format bnYYMMDDxxx"

        if data_type == "trigdat":
            assert (
                type(echan_list)
                and max(echan_list) <= 7
                and min(echan_list) >= 0
                and all(isinstance(x, int) for x in echan_list)
            ), (
                "Echan_list variable must be a list and can only "
                "have integer entries between 0 and 7"
            )

        self._data_type = data_type
        self._det = detector
        self._det_idx = valid_det_names.index(detector)
        self._trigger = trigger
        self._echan_list = echan_list

        if self._data_type == "trigdat":
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

    @property
    def day_start_times(self):
        """
        Returns the start time of the trigdata file to keep it conform with daily data
        :return:
        """
        return np.array([self._time_bins[0, 0]])

    @property
    def day_stop_times(self):
        """
        Returns the stop time of the trigdata file to keep it conform with daily data
        :return:
        """
        return np.array([self._time_bins[-1, 1]])

    @property
    def day_met(self):
        """
        Returns array with the day_met of all used days
        :return:
        """
        return np.array([self._time_bins[0, 0]])

    @property
    def day(self):
        """
        Returns the trigger name to keep it conform with daily data, this is used to create the output directory
        :return:
        """
        return [self._trigger]

    @property
    def rebinned_saa_mask(self):
        if self._rebinned:
            return self._rebinned_saa_mask
        else:
            raise Exception(
                "Data is unbinned, the saa mask has to be obtained from the SAA_calc object"
            )

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
    def trigtime(self):
        return self._trigtime

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

    def _build_arrays(self):
        year = "20%s" % self._trigger[2:4]

        # Download data-file and poshist file if not existing:
        datafile_name = "glg_{0}_{1}_{2}_v00.fit".format(
            self._data_type, "all", self._trigger
        )
        datafile_path = os.path.join(
            get_path_of_external_data_dir(), self._data_type, year, datafile_name
        )

        # If MPI is used only one rank should download the data; the others wait
        if rank == 0:
            if not file_existing_and_readable(datafile_path):
                download_trigdata_file(self._trigger, self._data_type)

        if using_mpi:
            comm.Barrier()

        self._trigdata_path = datafile_path

        evntrate = "EVNTRATE"

        # Open the datafile of the CTIME/CSPEC data and read in all needed quantities
        with fits.open(datafile_path) as trigdat:
            self._trigtime = trigdat[evntrate].header["TRIGTIME"]
            bin_start = trigdat[evntrate].data["TIME"]
            bin_stop = trigdat[evntrate].data["ENDTIME"]

            rates = trigdat[evntrate].data["RATE"]

            num_times = len(bin_start)
            rates = rates.reshape(num_times, 14, 8)

        # Sort out the high res times because they are dispersed with the normal
        # times.

        # The delta time in the file.
        # This routine is modeled off the procedure in RMFIT.
        myDelta = bin_stop - bin_start
        bin_start[myDelta < 0.1] = np.round(bin_start[myDelta < 0.1], 4)
        bin_stop[myDelta < 0.1] = np.round(bin_stop[myDelta < 0.1], 4)

        bin_start[~(myDelta < 0.1)] = np.round(bin_start[~(myDelta < 0.1)], 3)
        bin_stop[~(myDelta < 0.1)] = np.round(bin_stop[~(myDelta < 0.1)], 3)

        fine = True

        if fine:

            # Create a starting list of array indices.
            # We will dumb then ones that are not needed

            all_index = range(len(bin_start))

            # masks for all the different delta times and
            # the mid points for the different binnings
            temp1 = myDelta < 0.1
            temp2 = np.logical_and(myDelta > 0.1, myDelta < 1.0)
            temp3 = np.logical_and(myDelta > 1.0, myDelta < 2.0)
            temp4 = myDelta > 2.0
            midT1 = (bin_start[temp1] + bin_stop[temp1]) / 2.0
            midT2 = (bin_start[temp2] + bin_stop[temp2]) / 2.0
            midT3 = (bin_start[temp3] + bin_stop[temp3]) / 2.0

            # Dump any index that occurs in a lower resolution
            # binning when a finer resolution covers the interval
            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if bin_start[indx] < x < bin_stop[indx]:
                        try:

                            all_index.remove(indx)
                        except:
                            pass

            for indx in np.where(temp3)[0]:
                for x in midT2:
                    if bin_start[indx] < x < bin_stop[indx]:
                        try:

                            all_index.remove(indx)

                        except:
                            pass
            for indx in np.where(temp4)[0]:
                for x in midT3:
                    if bin_start[indx] < x < bin_stop[indx]:
                        try:

                            all_index.remove(indx)
                        except:
                            pass

            all_index = np.array(all_index)
        else:

            # Just deal with the first level of fine data
            all_index = np.where(myDelta > 1.0)[0].tolist()

            temp1 = np.logical_and(myDelta > 1.0, myDelta < 2.0)
            temp2 = myDelta > 2.0
            midT1 = (bin_start[temp1] + bin_stop[temp1]) / 2.0

            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if bin_start[indx] < x < bin_stop[indx]:

                        try:

                            all_index.remove(indx)

                        except:
                            pass

            all_index = np.array(all_index)

        # Now dump the indices we do not need
        bin_start = bin_start[all_index]
        bin_stop = bin_stop[all_index]

        rates = rates[all_index, :, :]

        # Now we need to sort because GBM may not have done this!

        sort_mask = np.argsort(bin_start)
        bin_start = bin_start[sort_mask]
        bin_stop = bin_stop[sort_mask]
        rates = rates[sort_mask, :, :]

        from threeML.utils.time_interval import TimeIntervalSet

        self._time_intervals = TimeIntervalSet.from_starts_and_stops(
            bin_start, bin_stop
        )

        counts = rates[:, self._det_idx, :] * self._time_intervals.widths.reshape(
            (len(self._time_intervals), 1)
        )

        time_bins = self._time_intervals.bin_stack

        self._counts = counts
        self._time_bins = time_bins

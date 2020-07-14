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


class Data(object):
    def __init__(self, dates, detectors, data_type, echans, simulation=False):
        """
        Initalize the ContinousData Class, which contains the information about the time bins 
        and counts of the data.
        """

        assert (
            data_type == "ctime" or data_type == "cspec"
        ), "Please use a valid data type: ctime or cspec"
        self._data_type = data_type
        for det in detectors:
            assert (
                det in valid_det_names
            ), "Please use a valid det name. One of these: {}".format(valid_det_names)

        assert type(dates) == list and len(dates[0]) == 6, (
            "Date variable has to be a list and every entry must have "
            "the format YYMMDD"
        )

        self._echan_mask_construction(echans)

        self._detectors = sorted(detectors)
        self._dates = sorted(dates)
        self._echans = echans
        self._simulation = simulation

        self._build_arrays()

        self._rebinned = False
        self._data_rebinner = None
        self._rebinned_counts = None
        self._rebinned_time_bins = None
        self._rebinned_saa_mask = None

    def _echan_mask_construction(self, echans):
        """
        Construct the echan masks for the reconstructed energy ranges
        :param echans: list with echans
        """
        if self._data_type == "ctime":
            echans_mask = []
            for e in echans:
                bounds = e.split("-")
                mask = np.zeros(8, dtype=bool)
                if len(bounds) == 1:
                    # Only one echan given
                    index = int(bounds[0])
                    assert (index <= 7 and index >= 0), (
                        "Only Echan numbers between 0 and 7 are allowed"
                    )
                    mask[index] = True
                else:
                    # Echan start and stop given
                    index_start = int(bounds[0])
                    index_stop = int(bounds[1])
                    assert (index_start <= 7 and index_start >= 0), (
                        "Only Echan numbers between 0 and 7 are allowed"
                    )
                    assert (index_stop <= 7 and index_stop >= 0), (
                        "Only Echan numbers between 0 and 7 are allowed"
                    )
                    mask[index_start:index_stop+1] = np.ones(1+index_stop-
                                                           index_start,
                                                           dtype=bool)
                echans_mask.append(mask)

        if self._data_type == "cspec":
            echans_mask = []
            for e in echans:
                bounds = e.split("-")
                mask = np.zeros(128, dtype=bool)
                if len(bounds) == 1:
                    # Only one echan given
                    index = int(bounds[0])
                    assert (index <= 127 and index >= 0), (
                        "Only Echan numbers between 0 and 127 are allowed"
                    )
                    mask[index] = True
                else:
                    # Echan start and stop given
                    index_start = int(bounds[0])
                    index_stop = int(bounds[1])
                    assert (index_start <= 127 and index_start >= 0), (
                        "Only Echan numbers between 0 and 127 are allowed"
                    )
                    assert (index_stop <= 127 and index_stop >= 0), (
                        "Only Echan numbers between 0 and 127 are allowed"
                    )
                    mask[index_start:index_stop+1] = np.ones(1+index_stop-
                                                           index_start,
                                                           dtype=bool)
                echans_mask.append(mask)
        self._echans_mask = echans_mask
        
    def rebinn_data(self, min_bin_width, saa_mask, save_memory=False):
        """
        Rebins the time bins to a min bin width
        :param min_bin_width:
        :return:
        """
        self._rebinned = True

        self._data_rebinner = Rebinner(self._time_bins, min_bin_width, mask=saa_mask)

        self._rebinned_time_bins = self._data_rebinner.time_rebinned

        self._rebinned_saa_mask = self._data_rebinner.rebinned_saa_mask

        self._rebinned_counts = self._data_rebinner.rebin(self._counts)[0].astype(
            np.int64
        )

        # Initialize the valid bin mask to all True
        self._valid_rebinned_time_mask = np.ones(
            len(self._rebinned_time_bins), dtype=bool
        )

        if save_memory:
            self._time_bins = None
            self._counts = None

    @property
    def counts(self):
        """
        Returns the count information of all time bins
        :return:
        """
        if self._rebinned:
            return self._rebinned_counts[self._valid_rebinned_time_mask]
        else:
            return self._counts[self._valid_time_mask]

    @property
    def time_bins(self):
        """
        Returns the time bin information of all time bins
        :return:
        """
        if self._rebinned:
            return self._rebinned_time_bins[self._valid_rebinned_time_mask]
        else:
            return self._time_bins[self._valid_time_mask]

    @property
    def rebinned_saa_mask(self):
        if self._rebinned:
            return self._rebinned_saa_mask
        else:
            raise Exception(
                "Data is unbinned, the saa mask has to be obtained from the SAA_calc object"
            )

    @property
    def valid_time_mask(self):
        return self._valid_time_mask

    @property
    def valid_rebinned_time_mask(self):
        return self._valid_rebinned_time_mask

    @property
    def detectors(self):
        """
        Returns the used detector
        :return:
        """
        return self._detectors

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
        Returns array which gives which of the days are following days
        (len(self._following_day)=num_days-1)
        :return:
        """
        return self._following_day

    @property
    def dates(self):
        """
        Returns day_list
        :return:
        """
        return self._dates

    @property
    def echans(self):
        """
        Returns day_list
        :return:
        """
        return self._echans

    @property
    def data_type(self):
        """
        Returns used data_type
        :return:
        """
        return self._data_type

    @property
    def time_bin_width(self):
        """
        Returns width of the time bins
        :return:
        """
        if self._rebinned:
            return np.diff(
                self._rebinned_time_bins[self._valid_rebinned_time_mask], axis=1
            )[:, 0]
        else:
            return np.diff(self._time_bins[self._valid_time_mask], axis=1)[:, 0]

    @property
    def mean_time(self):
        """
        Returns mean time of the time bins
        :return:
        """
        if self._rebinned:
            return np.mean(
                self._rebinned_time_bins[self._valid_rebinned_time_mask], axis=1
            )
        else:
            return np.mean(self._time_bins[self._valid_time_mask], axis=1)

    def _build_arrays(self):
        """
        Iterates over all wanted days and adds the count and time bin information in one big array
        :return:
        """
        following_day = np.array([])

        for det_idx, det in enumerate(self._detectors):

            for day_idx, day in enumerate(self._dates):
                counts, time_bins, day_met = self._one_day_one_det_data(day, det)

                if day_idx == 0:
                    count_array = counts
                    time_bins_array = time_bins
                    day_met_array = np.array([day_met])
                    day_start_times = np.array([time_bins[0, 0]])
                    day_stop_times = np.array([time_bins[-1, 1]])

                else:
                    start_index = np.searchsorted(
                        time_bins[:, 0], time_bins_array[-1, 1], side="right"
                    )

                    if time_bins_array[start_index, 0] < time_bins[-1, 1] + 1000:
                        following_day = np.append(following_day, True)
                    else:
                        following_day = np.append(following_day, False)

                    count_array = np.append(count_array, counts[start_index:], axis=0)
                    time_bins_array = np.append(
                        time_bins_array, time_bins[start_index:], axis=0
                    )
                    day_start_times = np.append(
                        day_start_times, time_bins[start_index, 0]
                    )
                    day_stop_times = np.append(day_stop_times, time_bins[-1, 1])
                    day_met_array = np.append(day_met_array, day_met)

            if det_idx == 0:
                count_total = np.zeros(
                    (len(time_bins_array), len(self._detectors), len(self._echans))
                )
                count_total[:, det_idx, :] = count_array
                time_bins_total = time_bins_array
                day_start_times_total = day_start_times
                day_stop_times_total = day_stop_times
                day_met_total = day_met_array
            else:
                assert np.array_equal(time_bins_total, time_bins_array)
                assert np.array_equal(day_start_times_total, day_start_times)
                assert np.array_equal(day_stop_times_total, day_stop_times)
                assert np.array_equal(day_met_total, day_met_array)
                count_total[:, det_idx, :] = count_array

        self._counts = count_total.astype(np.int64)
        self._time_bins = time_bins_total
        self._day_start_times = day_start_times_total
        self._day_stop_times = day_stop_times_total
        self._day_met_array = day_met_total
        self._following_day = following_day

        # Initialize the valid_mask to all True
        self._valid_time_mask = np.ones(len(self._time_bins), dtype=bool)

    def _one_day_one_det_data(self, day, det):
        """
        Prepares the data for one day
        :param day:
        :return:
        """
        version = "v00"

        # Download data-file and poshist file if not existing:
        datafile_name = "glg_{0}_{1}_{2}_{3}.pha".format(
            self._data_type, det, day, version
        )

        if self._simulation:
            datafile_path = os.path.join(
                get_path_of_external_data_dir(),
                "simulation",
                self._data_type,
                day,
                datafile_name,
            )
        else:
            datafile_path = os.path.join(
                get_path_of_external_data_dir(), self._data_type, day, datafile_name
            )

        poshistfile_name = "glg_{0}_all_{1}_v00.fit".format("poshist", day)
        poshistfile_path = os.path.join(
            get_path_of_external_data_dir(), "poshist", poshistfile_name
        )

        # If MPI is used only one rank should download the data; the others wait
        if using_mpi:
            if rank == 0:
                if not file_existing_and_readable(datafile_path):
                    download_data_file(day, self._data_type, det)

                if not file_existing_and_readable(poshistfile_path):
                    download_data_file(day, "poshist")
            comm.Barrier()
        else:
            if not file_existing_and_readable(datafile_path):
                download_data_file(day, self._data_type, det)

            if not file_existing_and_readable(poshistfile_path):
                download_data_file(day, "poshist")

        # Save poshistfile_path for later usage
        self._pos_hist = poshistfile_path

        # Open the datafile of the CTIME/CSPEC data and read in all needed quantities
        with fits.open(datafile_path) as f:
            counts = f["SPECTRUM"].data["COUNTS"]
            bin_start = f["SPECTRUM"].data["TIME"]
            bin_stop = f["SPECTRUM"].data["ENDTIME"]

        # Sometimes there are corrupt time bins where the time bin start = time bin stop
        # So we have to delete these times bins
        idx_zero_bins = np.where(bin_start == bin_stop)[0]

        bin_start = np.delete(bin_start, idx_zero_bins)
        bin_stop = np.delete(bin_stop, idx_zero_bins)
        counts = np.delete(counts, idx_zero_bins, axis=0)

        # Sometimes the poshist file does not cover the whole time covered by the CTIME/CSPEC file.
        # So we have to delete these time bins

        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f["GLAST POS HIST"].data["SCLK_UTC"]
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]

        # check for all time bins if they are outside of this interval
        idx_below_min_time = np.where(bin_start < min_time_pos)
        idx_above_max_time = np.where(bin_start > max_time_pos)
        idx_out_of_bounds = np.unique(
            np.hstack((idx_below_min_time, idx_above_max_time))
        )

        bin_start = np.delete(bin_start, idx_out_of_bounds)
        bin_stop = np.delete(bin_stop, idx_out_of_bounds)
        counts = np.delete(counts, idx_out_of_bounds, axis=0)

        # Calculate the MET time for the day
        day = day
        year = "20%s" % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        day_met = GBMTime(day_at).met
        # Get time bins
        time_bins = np.vstack((bin_start, bin_stop)).T

        # Convert to numpy int64
        counts = counts.astype(np.int64)

        # Only keep the count informations we need for the echan's we want to fit
        counts = self._add_counts_echan(counts)

        return counts, time_bins, day_met

    def _add_counts_echan(self, counts):
        """
        Add the counts together according to the echan masks
        :param counts: Counts in all time bins and all echans
        :return: summed counts in the definied echans and combined echans
        """
        sum_counts = np.zeros((len(counts), len(self._echans_mask)))
        for i, echan_mask in enumerate(self._echans_mask):
            for j, entry in enumerate(echan_mask):
                if entry:
                    sum_counts[:, i] += counts[:, j]

        return sum_counts

    def mask_invalid_bins(self, geometry_times):
        """
       This function mask the bins that are out of range for the interpolations
       """

        self._valid_time_mask = np.logical_and(
            (self._time_bins[:, 0] >= geometry_times[0]),
            (self._time_bins[:, 1]) <= geometry_times[-1],
        )

        if self._rebinned:
            self._valid_rebinned_time_mask = np.logical_and(
                (self._rebinned_time_bins[:, 0] >= geometry_times[0]),
                (self._rebinned_time_bins[:, 1]) <= geometry_times[-1],
            )
        else:
            self._valid_rebinned_time_mask = None

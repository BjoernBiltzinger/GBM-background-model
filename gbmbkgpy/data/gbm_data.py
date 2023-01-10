import numpy as np
import astropy.io.fits as fits

from gbmbkgpy.data.data import Data
from gbmbkgpy.io.downloading import download_gbm_file
from gbmbkgpy.io.package_data import get_path_of_external_data_dir


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

class GBMData(Data):

    def __init__(self, name, date, data_type, detector,
                 echans, min_time=None, max_time=None):

        self._date = date
        self._data_type = data_type
        self._detector = detector
        self._echans = echans
        self._min_time = min_time
        self._max_time = max_time

        assert detector in valid_det_names,\
            f"{detector} is not a valid detector name"

        self._download_data()

        self._echan_mask_construction()

        counts, time_bins = self._read_in_data()

        super().__init__(name, time_bins, counts)

        self._mask_saa()

    def _mask_saa(self):
        # find large gaps in time bins and mask these bins
        self._valid_time_mask = (self._time_bins[:, 1] -
                                 self._time_bins[:, 0]) < 50

    def _read_in_data(self):
        """
        Read in all the data
        """

        dir_path = get_path_of_external_data_dir()
        data_file_dir_path = dir_path / self._data_type / self._date

        poshist_file_path = (dir_path / "poshist" / self._date /
                        f"glg_poshist_all_{self._date}_v00.fit")


        data_file_path = (data_file_dir_path /
                          f"glg_{self._data_type}_{self._detector}"
                          f"_{self._date}_v00.pha")

        with fits.open(data_file_path) as f:
            counts = f["SPECTRUM"].data["COUNTS"]
            bin_start = f["SPECTRUM"].data["TIME"]
            bin_stop = f["SPECTRUM"].data["ENDTIME"]

            edge_start = f["EBOUNDS"].data["E_MIN"]
            edge_stop = f["EBOUNDS"].data["E_MAX"]

        self._Ebin_out_edge = np.append(edge_start, edge_stop[-1])
        # some clean ups:

        # Sometimes there are corrupt time bins where the
        # time bin start = time bin stop
        idx_zero_bins = bin_start == bin_stop
        bin_start = bin_start[~idx_zero_bins]
        bin_stop = bin_stop[~idx_zero_bins]
        counts = counts[~idx_zero_bins]

        # Sometimes the poshist file does not cover the whole time
        # covered by the CTIME/CSPEC file.
        #
        # Get boundary for time interval covered by the poshist file
        with fits.open(poshist_file_path) as f:
            pos_times = f["GLAST POS HIST"].data["SCLK_UTC"]
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]

        # check for all time bins if they are outside of this interval
        idx_outside_poshist = np.logical_or(bin_start < min_time_pos,
                                            bin_start > max_time_pos)

        bin_start = bin_start[~idx_outside_poshist]
        bin_stop = bin_stop[~idx_outside_poshist]
        counts = counts[~idx_outside_poshist]

        # remove time bins outside of the time between
        # min_time and max_time if they are given
        if self._min_time is not None:
            start_idx = np.argwhere(bin_stop > self._min_time)[0, 0]
        else:
            start_idx = 0
        if self._max_time is not None:
            stop_idx = np.argwhere(bin_start < self._max_time)[-1, 0]
        else:
            stop_idx = -1

        bin_start = bin_start[start_idx:stop_idx]
        bin_stop = bin_stop[start_idx:stop_idx]
        counts = counts[start_idx:stop_idx]

        # Get time bins
        time_bins = np.vstack((bin_start, bin_stop)).T

        # Convert to numpy int64
        counts = counts.astype(np.int64)

        # bin the counts with the echan mask
        counts = self._add_counts_echan(counts)

        return counts, time_bins

    def _download_data(self):
        # download the poshist files
        download_gbm_file(self._date, "poshist")

        # download data files
        download_gbm_file(self._date, self._data_type, self._detector)

    def _echan_mask_construction(self):
        """
        Construct the echan masks for the reconstructed energy ranges
        :param echans: list with echans
        """
        if self._data_type == "ctime":
            max_echan = 7
        elif self._data_type == "cspec":
            max_echan = 127

        echans_mask = []
        for e in self._echans:
            bounds = e.split("-")
            mask = np.zeros(max_echan+1, dtype=bool)
            if len(bounds) == 1:
                # Only one echan given
                index = int(bounds[0])
                assert (
                    0 <= index <= max_echan
                ), f"Only Echan numbers between 0 and {max_echan} are allowed"
                mask[index] = True
            else:
                # Echan start and stop given
                index_start = int(bounds[0])
                index_stop = int(bounds[1])
                assert (
                    0 <= index_start <= max_echan
                ), f"Only Echan numbers between 0 and {max_echan} are allowed"
                assert (
                    0 <= index_stop <= max_echan
                ), f"Only Echan numbers between 0 and {max_echan} are allowed"
                mask[index_start: index_stop + 1] = np.ones(
                    1 + index_stop - index_start, dtype=bool
                )
            echans_mask.append(mask)
        self._echans_mask = np.array(echans_mask)

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

    def cut_out_saa(self, t):
        """
        Cut out a certain time t after every SAA exit
        """

        # times of saa
        saa_times = self.saa_times

        for t_0 in saa_times:
            self.mask_data(t_0, t)

    @property
    def saa_times(self):
        # find the time bins with a gap to the previous time bin => SAA
        saa_idx = np.argwhere(self._time_bins[1:, 0] - self._time_bins[:-1, 1]
                              > 10)
        return self._time_bins[saa_idx+1, 0]

    @property
    def ebin_out_edges(self):
        return self._Ebin_out_edge

    @property
    def echans_mask(self):
        return self._echans_mask

    @property
    def det(self):
        return self._detector

import numpy as np

from gbmbkgpy.utils.binner import Rebinner


class Data:

    def __init__(self, name, time_bins, counts=None):
        """
        Init Data object
        :param time_bins: Time bins of data
        :param counts: Counts for all time bins, echans
        """
        self._name = name
        self._time_bins = time_bins
        self._counts = counts
        self._rebinned = False
        # Initialize the valid_mask to all True
        self._valid_time_mask = np.ones(len(self._time_bins), dtype=bool)

    def rebin_data(self, min_bin_width, save_memory=False):
        """
        Rebins the time bins to a min bin width
        :param min_bin_width: min time of the new bins
        :param mask: additional mask to mask the original bins
        :param save_memory:
        :return:
        """
        self._rebinned = True

        self._data_rebinner = Rebinner(self._time_bins, min_bin_width,
                                       mask=self._valid_time_mask)

        self._rebinned_time_bins = self._data_rebinner.time_rebinned

        self._rebinned_mask = self._data_rebinner.rebinned_mask

        self._rebinned_counts = self._data_rebinner.rebin(self._counts)[0].astype(
            np.int64
        )

        # Initialize the valid bin mask to mask saa bins all True
        self._valid_rebinned_time_mask = (self._rebinned_time_bins[:, 1] -
                                          self._rebinned_time_bins[:, 0]) < 100


        if save_memory:
            self._time_bins = None
            self._counts = None

    @property
    def time_bin_width(self):
        """
        Returns width of the time bins
        :return: width of time bins
        """
        if self._rebinned:
            return np.diff(
                self._rebinned_time_bins[self.valid_rebinned_time_mask], axis=1
            )[:, 0]

        return np.diff(self._time_bins[self.valid_time_mask], axis=1)[:, 0]

    @property
    def mean_time(self):
        """
        Returns mean time of the time bins
        :return: mean time of time bins
        """
        if self._rebinned:
            return np.mean(
                self._rebinned_time_bins[self.valid_rebinned_time_mask], axis=1
            )

        return np.mean(self._time_bins[self.valid_time_mask], axis=1)

    @property
    def counts(self):
        """
        Returns the count information of all time bins
        :return: counts
        """
        if self._rebinned:
            return self._rebinned_counts[self.valid_rebinned_time_mask]

        return self._counts[self.valid_time_mask]

    @property
    def time_bins(self):
        """
        Returns the time bin information of all time bins
        :return: time_bins
        """
        if self._rebinned:
            return self._rebinned_time_bins[self.valid_rebinned_time_mask]

        return self._time_bins[self.valid_time_mask]

    @property
    def rebinned_mask(self):
        if self._rebinned:
            return self._rebinned_mask

        raise Exception(
            "Data is unbinned"
        )

    @property
    def valid_time_mask(self):
        return self._valid_time_mask

    @property
    def valid_rebinned_time_mask(self):
        return self._valid_rebinned_time_mask

    @property
    def name(self):
        return self._name

    @property
    def num_echan(self):
        return self._counts.shape[1]


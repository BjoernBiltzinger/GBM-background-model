import numpy as np

from gbmbkgpy.utils.binner import Rebinner


class Data:

    def __init__(self, name, time_bins, counts=None, valid_time_mask=None):
        """
        Init Data object
        :param time_bins: Time bins of data
        :param counts: Counts for all time bins, echans
        """
        self._name = name
        self._time_bins = time_bins
        self._counts = counts
        #self._rebinned = False

        self._min_bin_width = 0
        self._rebinned_time_bins = time_bins
        self._rebinned_counts = counts

        self._fit_rebinned_time_bins = time_bins
        self._fit_rebinned_counts = counts

        # Initialize the valid_mask to all True
        if valid_time_mask is not None:
            self._valid_time_mask = valid_time_mask
        else:
            self._valid_time_mask = np.ones(len(self._time_bins), dtype=bool)

        self._valid_rebinned_time_mask = self._valid_time_mask

        self._fit_time_mask = self._valid_time_mask

        self._fit_rebinned_time_mask = self._valid_time_mask

    def rebin_data(self, min_bin_width):
        """
        Rebins the time bins to a min bin width
        :param min_bin_width: min time of the new bins
        :param mask: additional mask to mask the original bins
        :param save_memory:
        :return:
        """
        self._min_bin_width = min_bin_width

        #self._rebinned = True

        valid_data_rebinner = Rebinner(self._time_bins, min_bin_width,
                                       mask=self._valid_time_mask)

        self._rebinned_time_bins = valid_data_rebinner.time_rebinned

        self._valid_rebinned_time_mask = valid_data_rebinner.rebinned_mask

        self._rebinned_counts = valid_data_rebinner.rebin(self._counts)[0].astype(
            np.int64
        )

        fit_data_rebinner = Rebinner(self._time_bins, min_bin_width,
                                     mask=self._fit_time_mask)

        self._fit_rebinned_time_bins = fit_data_rebinner.time_rebinned

        self._fit_rebinned_time_mask = fit_data_rebinner.rebinned_mask

        self._fit_rebinned_counts = fit_data_rebinner.rebin(self._counts)[0].astype(
            np.int64
        )


    def mask_start_of_data(self, t):
        """
        Mask start of data
        """
        self.mask_data(self._time_bins[0, 0], t)

    def mask_data(self, t_0, t, unvalid=True):
        """
        Mask all the time bins starting between t0 and t0+t
        """
        mask = np.logical_and(self._time_bins[:, 0]-t_0 <= t,
                              self._time_bins[:, 0] >= t_0)

        self._fit_time_mask[mask] = False

        if unvalid:
            self._valid_time_mask[mask] = False

        #if self._rebinned:
        mask = np.logical_and(
            self._rebinned_time_bins[:, 0]-t_0 <= t,
            self._rebinned_time_bins[:, 0] >= t_0
        )

        self._fit_rebinned_time_mask[mask] = False

        if unvalid:
            self._valid_rebinned_time_mask[mask] = False

    @property
    def fit_counts(self):
        """
        Returns the count information of all time bins
        :return: counts
        """
        return self._fit_rebinned_counts[self._fit_rebinned_time_mask]

    @property
    def fit_time_bins(self):
        """
        Returns the time bin information of all time bins
        :return: time_bins
        """
        return self._fit_rebinned_time_bins[self._fit_rebinned_time_mask]

    @property
    def time_bin_width(self):
        """
        Returns width of the time bins
        :return: width of time bins
        """
        #if self._rebinned:
        return np.diff(
            self._rebinned_time_bins[self.valid_rebinned_time_mask], axis=1
        )[:, 0]

        #return np.diff(self._time_bins[self.valid_time_mask], axis=1)[:, 0]

    @property
    def mean_time(self):
        """
        Returns mean time of the time bins
        :return: mean time of time bins
        """

        return np.mean(
            self._rebinned_time_bins[self.valid_rebinned_time_mask], axis=1
        )

    @property
    def counts(self):
        """
        Returns the count information of all time bins
        :return: counts
        """
        return self._rebinned_counts[self.valid_rebinned_time_mask]

    @property
    def time_bins(self):
        """
        Returns the time bin information of all time bins
        :return: time_bins
        """
        return self._rebinned_time_bins[self.valid_rebinned_time_mask]

    @property
    def fit_time_mask(self):
        return self._fit_time_mask

    @property
    def fit_rebinned_time_mask(self):
        return self._fit_rebinned_time_mask

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

    @property
    def min_bin_width(self):
        return self._min_bin_width

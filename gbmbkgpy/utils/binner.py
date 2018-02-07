import numpy as np


class Rebinner(object):
    """
    A class to rebin vectors keeping a minimum bin_width. It supports array with a mask, so that elements excluded
    through the mask will not be considered for the rebinning
    """

    def __init__(self, vector_to_rebin_on, min_bin_width, mask=None):

        if mask is not None:

            mask = np.array(mask, bool)

            assert mask.shape[0] == len(vector_to_rebin_on), "The provided mask must have the same number of " \
                                                             "elements as the vector to rebin on"

        else:
            mask = np.ones_like(vector_to_rebin_on[:, 0], dtype=bool)

        self._mask = mask

        self._stops = []
        self._starts = []
        self._grouping = []

        sum_bin_width = 0.
        bin_open = False

        for index, bin in enumerate(vector_to_rebin_on):

            if not mask[index]:
                # This element is excluded by the mask

                if not bin_open:

                    continue

                else:

                    # The bin needs to be closed here!

                    self._stops.append(index)

                    sum_bin_width = 0.
                    bin_open = False

            else:

                # This element is included by the mask

                this_bin_width = bin[1] - bin[0]

                if not bin_open:
                    # Open a new bin
                    bin_open = True

                    self._starts.append(index)
                    sum_bin_width = 0.

                # Add the current bin width to the sum_bin_with
                sum_bin_width += this_bin_width

                if sum_bin_width >= min_bin_width:

                    if index == (len(vector_to_rebin_on) - 1):
                        stop_index = index
                        print"index is {}".format(index)
                    else:
                        stop_index = index + 1

                    self._stops.append(stop_index)

                    bin_open = False

        # At the end of the loop, see if we left a bin open, if we did, close it
        if bin_open:
            self._stops.append(len(vector_to_rebin_on) - 1)

        assert len(self._starts) == len(self._stops), "This is a bug: the starts and stops of the bins are not in " \
                                                      "equal number"

        self._min_bin_width = min_bin_width

        self._rebinned_vector_idx = np.array(zip(self._starts, self._stops))

        self._time_rebinned = np.array(zip(vector_to_rebin_on[self._starts, 0], vector_to_rebin_on[self._stops, 0]))
        #Set stop time of last bin to correct value
        self._time_rebinned[-1][1] = vector_to_rebin_on[self._stops[-1]][1]

    @property
    def n_bins(self):
        """
        Returns the number of bins defined.
        :return:
        """

        return len(self._starts)

    @property
    def time_rebinned(self):

        return self._time_rebinned

    def rebin(self, *vectors):
        """
        Rebin the given vectores and return them as a list
        :param vectors:
        :return:
        """

        rebinned_vectors = []

        for vector in vectors:

            assert len(vector) == len(
                self._mask), "The vector to rebin must have the same number of elements of the" \
                             "original (not-rebinned) vector"

            # Transform in array because we need to use the mask
            vector_a = np.array(vector)

            rebinned_vector = []

            for low_bound, hi_bound in zip(self._starts, self._stops):
                rebinned_vector.append(np.sum(vector_a[low_bound:hi_bound]))

            # Vector might not contain counts, so we use a relative comparison to check that we didn't miss
            # anything.
            # NOTE: we add 1e-100 because if both rebinned_vector and vector_a contains only 0, the check would
            # fail when it shouldn't

            assert abs((np.sum(rebinned_vector) + 1e-100) / (np.sum(vector_a[self._mask]) + 1e-100) - 1) < 1e-4

            rebinned_vectors.append(np.array(rebinned_vector))

        return rebinned_vectors

    def rebin_errors(self, *vectors):
        """
        Rebin errors by summing the squares
        Args:
            *vectors:
        Returns:
            array of rebinned errors
        """

        rebinned_vectors = []

        for vector in vectors:  # type: np.ndarray[np.ndarray]

            assert len(vector) == len(self._mask), "The vector to rebin must have the same number of elements of the" \
                                                   "original (not-rebinned) vector"

            rebinned_vector = []

            for low_bound, hi_bound in zip(self._starts, self._stops):

                rebinned_vector.append(np.sqrt(np.sum(vector[low_bound:hi_bound] ** 2)))

            rebinned_vectors.append(np.array(rebinned_vector))

        return rebinned_vectors
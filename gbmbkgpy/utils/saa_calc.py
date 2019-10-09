import numpy as np

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


class SAA_calc(object):
    def __init__(self, data_object, bins_to_add=8, time_after_SAA=5000, short_time_intervals=False):
        """
        Initalize the SAA calculation that calculates the start and stop times of the SAAs and builds
        masks. 
        :params time_bins: the time bins of the day
        :params bins_to_add: number of bins to add before and after the SAA always
        :params time_after_SAA: time after the SAA to ignore if the SAA is not included in the model
        :params short_time_intervals: Should short time intervals (<1000 sec.) be used in the analysis?
        """

        assert type(bins_to_add) == int, 'bins_to_add gives the number of time_bins to add before and after the SAA. It must therefore be an int but it is {}.'.format(type(bins_to_add))
        assert type(data_object.time_bins) == np.ndarray, 'Invalid type for time_bins. Must be an array but is {}.'.format(type(data_object.time_bins))
        assert type(time_after_SAA) == int, 'time_after_SAA must be a int but is a {}'.format(type(time_after_SAA))
        assert type(short_time_intervals) == bool, 'short_time_intervals must be a bool but is {}'.format(short_time_intervals)

        self._time_bins = data_object.time_bins
        self._build_masks(bins_to_add, time_after_SAA, short_time_intervals)

    @property
    def saa_mask(self):
        """
        Returns SAA mask
        """
        return self._saa_mask

    @property
    def times_bins(self):
        """
        Returns times bins
        """
        return self._time_bins

    @property
    def num_saa(self):
        """
        Returns number of SAA's
        :return:
        """
        return self._num_saa

    @property
    def saa_exit_times(self):
        """
        Returns the times of the SAA exit times
        :return:
        """
        return self._saa_exit_time_bins[:, 1]

    def _build_masks(self, bins_to_add, time_after_SAA, short_time_intervals):
        """
        Calculates masks that cover the SAAs and some time bins before and after the SAAs
        :params bins_to_add: number of bins to add to mask before and after time bin
        """

        # Calculate the time jump between two successive time bins. During the SAAs no data is recorded.
        # This leads to a time jump between two successive time bins before and after the SAA.
        jump = self._time_bins[1:, 0] - self._time_bins[:-1, 1]

        # Get mask for which the jump is > 1 second
        jump_large = jump > 1

        # Get the indices of the time bins for which the jump to the previous time bin is >1 second
        # +1 is needed because we started with second time bin (we can not calculate the time jump
        # between the first time bin and the time bin before that one)
        idx = jump_large.nonzero()[0] + 1

        # Build slices, that have as first entry start of SAA and as second end of SAA
        slice_idx = np.array(self.slice_disjoint(idx))
        self._num_saa = len(slice_idx)

        # Add bins_to_add before and after SAAs
        slice_idx[:, 0][np.where(slice_idx[:, 0] >= bins_to_add)] = \
            slice_idx[:, 0][np.where(slice_idx[:, 0] >= bins_to_add)] - bins_to_add

        slice_idx[:, 1][np.where(slice_idx[:, 1] <= len(self._time_bins) - 1 - bins_to_add)] = \
            slice_idx[:, 1][np.where(slice_idx[:, 1] <= len(self._time_bins) - 1 - bins_to_add)] + bins_to_add

        # Find the times of the exits
        if slice_idx[-1, 1] == len(self._time_bins) - 1:
            # the last exit is just the end of the array
            saa_exit_idx = slice_idx[:-1, 1]
        else:
            saa_exit_idx = slice_idx[:, 1]

        self._saa_exit_time_bins = self._time_bins[saa_exit_idx]

        # make a saa mask from the slices:
        self._saa_mask = np.ones(len(self._time_bins), bool)
        for sl in slice_idx:
            self._saa_mask[sl[0]:sl[1] + 1] = False
        # If time_after_SAA is not send to None we add time_after_SAA seconds after every SAA exit to the mask
        # Useful to ignore the times with a large influence by the SAA
        if time_after_SAA is not None:
            # Set first time_after_SAA seconds False
            j = 0
            while time_after_SAA > self._time_bins[j, 1] - self._time_bins[0, 0]:
                j += 1
            self._saa_mask[0:j] = False

            # Do the same for every SAA exit. We have to be careful to not cause an error when the time
            # after a SAA is less than time_after_SAA seconds
            for i in range(len(slice_idx)):

                if self._time_bins[:, 0][slice_idx[i, 1]] + time_after_SAA > self._time_bins[-1, 0]:

                    self._saa_mask[slice_idx[i, 1]:len(self._time_bins)] = False

                else:

                    j = 0
                    while time_after_SAA > self._time_bins[:, 1][slice_idx[i, 1] + j] - self._time_bins[:, 0][slice_idx[i, 1]]:
                        j += 1

                    self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False

        # If wanted delete separeted time intervals shorter than 1000 seconds (makes plots nicer)
        if not short_time_intervals:
            # get index intervals of SAA mask
            index_start = [0]
            index_stop = []

            for i in range(len(self._saa_mask) - 1):
                if self._saa_mask[i] == False and self._saa_mask[i + 1] == True:
                    index_stop.append(i - 1)
                if self._saa_mask[i] == True and self._saa_mask[i + 1] == False:
                    index_start.append(i)

            if len(index_start) > len(index_stop):
                index_stop.append(-1)

            assert len(index_start) == len(index_stop), 'Something is wrong, index_start and index_stop must have same length. But index_start as length {} and index_stop has length {}.'.format(len(index_start), len(index_stop))

            # set saa_mask=False between end and next start if time is <min_duration
            for i in range(len(index_stop) - 1):
                if self._time_bins[:, 1][index_start[i + 1]] - self._time_bins[:, 0][index_stop[i]] < 1000:
                    self._saa_mask[index_stop[i] - 5:index_start[i + 1] + 5] = np.ones_like(self._saa_mask[index_stop[i] - 5:index_start[i + 1] + 5]) == 2

    def slice_disjoint(self, arr):
        """
        Returns an array of disjoint indices from a bool array
        :param arr: and array of bools
        """

        slices = []
        start_slice = arr[0]
        counter = 0
        for i in range(len(arr) - 1):
            if arr[i + 1] > arr[i] + 1:
                end_slice = arr[i]
                slices.append([start_slice, end_slice])
                start_slice = arr[i + 1]
                counter += 1
        if counter == 0:
            return [[arr[0], arr[-1]]]
        if end_slice != arr[-1]:
            slices.append([start_slice, arr[-1]])
        return slices

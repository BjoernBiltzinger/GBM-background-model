import astropy.io.fits as fits
import numpy as np


class ContinuousData(object):
    def __init__(self, file_name):
        with fits.open(file_name) as f:
            self._counts = f['SPECTRUM'].data['COUNTS']
            self._bin_start = f['SPECTRUM'].data['TIME']
            self._bin_stop = f['SPECTRUM'].data['ENDTIME']

            self._n_entries = len(self._bin_start)

            self._exposure = f['SPECTRUM'].data['EXPOSURE']
            self._bin_start = f['SPECTRUM'].data['TIME']

    @property
    def rates(self):
        return self._counts / self._exposure.reshape((self._n_entries, 1))

    @property
    def counts(self):
        return self._counts

    @property
    def exposure(self):
        return self._exposure

    @property
    def time_bins(self):
        return np.vstack((self._bin_start, self._bin_stop)).T

    @property
    def mean_time(self):
        return np.mean(self.time_bins, axis=1)

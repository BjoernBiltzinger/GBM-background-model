FLARE_SOURCE, CONTINUUM_SOURCE, POINT_SOURCE, SAA_SOURCE, GLOBAL_SOURCE = 'flare_source', 'continuum_source', 'point_source', 'saa_source', 'global_source'
from scipy import integrate
import numpy as np

class Source(object):
    def __init__(self, name, source_type, shape, echan):
        self._name = name
        self._source_type = source_type
        self._shape = shape
        self._echan = echan

        assert source_type in [POINT_SOURCE, CONTINUUM_SOURCE, FLARE_SOURCE, SAA_SOURCE, GLOBAL_SOURCE], 'improper source'

    def __call__(self):

        return self._shape()

    def get_counts_old(self, a, b):
        return (b-a)/6*(self._shape(a)+4*self._shape((a+b)/2)+self._shape(b))

    def get_counts_quad(self, a, b):
        return integrate.quad(self._shape, a, b)

    def get_counts(self, time_bins, echan, bin_mask=None):
        if bin_mask is None:
            bin_mask = np.ones(len(time_bins), dtype=bool)  # np.full(len(time_bins), True)

        return integrate.cumtrapz(self._shape(echan)[bin_mask], time_bins)

    @property
    def name(self):
        return self._name

    @property
    def source_type(self):
        return self._source_type

    @property
    def echan(self):
        return self._echan

    @property
    def parameters(self):

        return self._shape.parameters


class ContinuumSource(Source):
    def __init__(self, name, continuum_shape, echan):
        super(ContinuumSource, self).__init__(name, CONTINUUM_SOURCE, continuum_shape, echan)


class FlareSource(Source):
    def __init__(self, name, flare_shape, echan):
        super(FlareSource, self).__init__(name, FLARE_SOURCE, flare_shape, echan)


class PointSource(Source):
    def __init__(self, name, point_shape, echan):
        super(PointSource, self).__init__(name, POINT_SOURCE, point_shape, echan)


class SAASource(Source):
    def __init__(self, name, saa_shape, echan):
        super(SAASource, self).__init__(name, SAA_SOURCE, saa_shape, echan)

class GlobalSource(Source):
    def __init__(self, name, continuum_shape):
        super(GlobalSource, self).__init__(name, GLOBAL_SOURCE, continuum_shape, 1) #dummy value for echan




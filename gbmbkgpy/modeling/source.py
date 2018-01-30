FLARE_SOURCE, CONTINUUM_SOURCE, POINT_SOURCE, SAA_SOURCE = 'flare_source', 'continuum_source', 'point_source', 'saa_source'
from scipy import integrate
import numpy as np

class Source(object):
    def __init__(self, name, source_type, shape):
        self._name = name
        self._source_type = source_type
        self._shape = shape

        assert source_type in [POINT_SOURCE, CONTINUUM_SOURCE, FLARE_SOURCE, SAA_SOURCE], 'improper source'

    def __call__(self):

        return self._shape()

    def get_flux_old(self, a, b):
        return (b-a)/6*(self._shape(a)+4*self._shape((a+b)/2)+self._shape(b)) #integrate.quad(self._shape, a, b)

    def get_flux_quad(self, a, b):
        return integrate.quad(self._shape, a, b)

    def get_flux(self, time_bins, bin_mask=None):

        if bin_mask is None:
            bin_mask = np.full(len(time_bins), True)

        return integrate.cumtrapz(self._shape()[bin_mask], time_bins)

    @property
    def name(self):
        return self._name

    @property
    def source_type(self):
        return self._source_type

    @property
    def parameters(self):

        return self._shape.parameters


class ContinuumSource(Source):
    def __init__(self, name, continuum_shape):
        super(ContinuumSource, self).__init__(name, CONTINUUM_SOURCE, continuum_shape)


class FlareSource(Source):
    def __init__(self, name, flare_shape):
        super(FlareSource, self).__init__(name, FLARE_SOURCE, flare_shape)


class PointSource(Source):
    def __init__(self, name, point_shape):
        super(PointSource, self).__init__(name, POINT_SOURCE, point_shape)


class SAASource(Source):
    def __init__(self, name, saa_shape):
        super(SAASource, self).__init__(name, SAA_SOURCE, saa_shape)




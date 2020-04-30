(
    FLARE_SOURCE,
    CONTINUUM_SOURCE,
    POINT_SOURCE,
    SAA_SOURCE,
    GLOBAL_SOURCE,
    FIT_SPECTRUM_SOURCE,
    TRANSIENT_SOURCE,
) = (
    "flare_source",
    "continuum_source",
    "point_source",
    "saa_source",
    "global_source",
    "fit_spectrum_source",
    "transient_source",
)
from scipy import integrate
import numpy as np


class Source(object):
    def __init__(self, name, source_type, shape, index):
        self._name = name
        self._source_type = source_type
        self._shape = shape
        self._index = index

        assert source_type in [
            POINT_SOURCE,
            CONTINUUM_SOURCE,
            FLARE_SOURCE,
            SAA_SOURCE,
            GLOBAL_SOURCE,
            FIT_SPECTRUM_SOURCE,
            TRANSIENT_SOURCE,
        ], "improper source"

    def __call__(self):
        return self._shape()

    def recalculate_counts(self):
        self._shape.recalculate_counts()

    def get_counts(self, time_bins, bin_mask=None):
        """
        Calls the evaluation of the source to get the counts per bin. Uses a bin_mask to exclude some bins if needed.
        No need of integration here anymore! This is done in the function class of the sources!
        :param time_bins:
        :param echan:
        :param bin_mask:
        :return:
        """
        if bin_mask is None:
            bin_mask = np.ones(
                len(time_bins), dtype=bool
            )  # np.full(len(time_bins), True)
        return self._shape()[bin_mask]

    @property
    def name(self):
        return self._name

    @property
    def source_type(self):
        return self._source_type

    @property
    def echan(self):
        """
        Returns the index of the echan (the position of the echan in the echan_list)
        :return:
        """
        return self._index

    @property
    def parameters(self):
        return self._shape.parameters


class ContinuumSource(Source):
    def __init__(self, name, continuum_shape, index):
        super(ContinuumSource, self).__init__(
            name, CONTINUUM_SOURCE, continuum_shape, index
        )


class FlareSource(Source):
    def __init__(self, name, flare_shape, index):
        super(FlareSource, self).__init__(name, FLARE_SOURCE, flare_shape, index)


class PointSource(Source):
    def __init__(self, name, point_shape, index):
        super(PointSource, self).__init__(name, POINT_SOURCE, point_shape, index)


class SAASource(Source):
    def __init__(self, name, saa_shape, index):
        super(SAASource, self).__init__(name, SAA_SOURCE, saa_shape, index)


class TransientSource(Source):
    def __init__(self, name, transient_shape, index):
        super(TransientSource, self).__init__(
            name, TRANSIENT_SOURCE, transient_shape, index
        )


class GlobalSource(Source):
    def __init__(self, name, continuum_shape):
        super(GlobalSource, self).__init__(
            name, GLOBAL_SOURCE, continuum_shape, -1
        )  # dummy value for echan index


class FitSpectrumSource(Source):
    def __init__(self, name, continuum_shape):
        super(FitSpectrumSource, self).__init__(
            name, FIT_SPECTRUM_SOURCE, continuum_shape, -1
        )  # dummy value for echan index

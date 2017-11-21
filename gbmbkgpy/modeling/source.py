FLARE_SOURCE, CONTINUUM_SOURCE, POINT_SOURCE = 'flare_source', 'continuum_source', 'point_source'


class Source(object):
    def __init__(self, name, source_type, shape):
        self._name = name
        self._source_type = source_type
        self._shape = shape

        assert source_type in [POINT_SOURCE, CONTINUUM_SOURCE, FLARE_SOURCE], 'improper source'

    def __call__(self, x):

        return self._shape(x)

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



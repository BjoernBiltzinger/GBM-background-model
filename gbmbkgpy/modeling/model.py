import collections
from source import CONTINUUM_SOURCE, POINT_SOURCE, FLARE_SOURCE

class Model(object):
    def __init__(self, *sources):

        self._continuum_sources = collections.OrderedDict()

        self._flare_sources = collections.OrderedDict()

        self._point_sources = collections.OrderedDict()

        for source in sources:
            self._add_source(source)

        self._update_parameters()

    @property
    def free_parameters(self):
        """
        Get a dictionary with all the free parameters in this model
        :return: dictionary of free parameters
        """

        # Refresh the list

        self._update_parameters()

        # Filter selecting only free parameters

        free_parameters_dictionary = collections.OrderedDict()

        for parameter_name, parameter in self._parameters.iteritems():

            if parameter.free:
                free_parameters_dictionary[parameter_name] = parameter

        return free_parameters_dictionary

    @property
    def parameters(self):
        """
        Return a dictionary with all parameters
        :return: dictionary of parameters
        """
        self._update_parameters()

        return self._parameters

    def _update_parameters(self):

        parameters = collections.OrderedDict()

        for source in [self._continuum_sources, self._flare_sources, self._point_sources]:

            for parameter_name, parameter in source.parameters.iteritems():
                parameters[parameter_name] = parameter

        self._parameters = parameters

    def _add_source(self, source):

        if source.source_type == POINT_SOURCE:

            self._point_sources[source.name] = source

        if source.source_type == FLARE_SOURCE:

            self._flare_sources[source.name] = source

        if source.source_type == CONTINUUM_SOURCE:

            self._continuum_sources[source.name] = source

    @property
    def point_sources(self):

        return self._point_sources

    @property
    def flare_sources(self):

        return self._flare_sources

    @property
    def continuum_sources(self):

        return self._continuum_sources

    def __call__(self, x):
        return self._a * x + self._b

    def set_fit_parameters(self, a, b):
        self._a = a
        self._b = b

    def set_data(self):
        # get data from external_prop
        # get continuous data
        pass

    def _build_model(self, x):
        self._a = "calculating parameter"
        self._b = "calculating parameter"

    def plot_model(self, model):
        pass


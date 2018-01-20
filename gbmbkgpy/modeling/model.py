import collections
from source import CONTINUUM_SOURCE, POINT_SOURCE, FLARE_SOURCE, SAA_SOURCE
import numpy as np

class Model(object):
    def __init__(self, *sources):

        self._continuum_sources = collections.OrderedDict()

        self._flare_sources = collections.OrderedDict()

        self._point_sources = collections.OrderedDict()

        self._saa_sources = collections.OrderedDict()

        for source in sources:
            self._add_source(source)

        self._update_parameters()

        self._saa_regions = []

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

        for sources in [self._continuum_sources, self._flare_sources, self._point_sources, self._saa_sources]:

            for source in sources.itervalues():

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

        if source.source_type == SAA_SOURCE:

            self._saa_sources[source.name] = source

    @property
    def point_sources(self):

        return self._point_sources

    @property
    def flare_sources(self):

        return self._flare_sources

    @property
    def continuum_sources(self):

        return self._continuum_sources

    @property
    def saa_sources(self):

        return self._saa_sources

    @property
    def n_point_sources(self):

        return len(self._point_sources)

    @property
    def n_flare_sources(self):

        return len(self._flare_sources)

    @property
    def n_continuum_sources(self):

        return len(self._continuum_sources)

    @property
    def n_saa_sources(self):

        return len(self._saa_sources)

    def get_continuum_flux(self, id, time_bins):
        """
        
        :param id: 
        :param time_bins:
        :return: 
        """

        return self._continuum_sources.values()[id].get_flux(time_bins)

    def get_flare_flux(self, id, t_start, t_stop):
        """
        
        :param id: 
        :param t: 
        :return: 
        """

        return self._flare_sources.values()[id].get_flux(t_start, t_stop)

    def get_point_source_flux(self, id, t_start, t_stop):
        """
        
        :param id: 
        :param t: 
        :return: 
        """

        return self._point_sources.values()[id].get_flux(t_start, t_stop)

    def get_saa_source_flux(self, id, t_start, t_stop):
        """

        :param id:
        :param t:
        :return:
        """
        return self._saa_sources.values()[id].get_flux(t_start, t_stop)

    def add_SAA_regions(self, *regions):
        """
        Add SAA temporal regions which cause the model to be set to zero
        
        :param regions: 
        :return: 
        """

    def get_flux(self, time_bins):

        total_flux = np.zeros(len(time_bins))

        for continuum_source in self._continuum_sources.values():
            total_flux += np.ndarray.flatten(continuum_source.get_flux(time_bins))

        for flare_source in self._flare_sources.values():
            total_flux += flare_source.get_flux(time_bins)

        for point_source in self._point_sources.values():
            total_flux += point_source.get_flux(time_bins)

        for saa_source in self._saa_sources.values():
            total_flux += saa_source.get_flux(time_bins)

        return total_flux

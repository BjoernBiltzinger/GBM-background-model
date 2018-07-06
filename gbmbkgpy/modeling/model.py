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

    def set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters:
        :return:
        """
        for i, parameter in enumerate(self.free_parameters.itervalues()):

            parameter.value = new_parameters[i]

    def set_parameter_bounds(self, new_bounds):
        """
        Set the parameter bounds
        :param new_bounds:
        :return:
        """
        for i, parameter in enumerate(self._parameters.itervalues()):
            parameter.bounds = new_bounds[i]

    @property
    def normalization_parameters(self):
        """
        Get a dictionary with all the normalization parameters in this model
        :return: dictionary of normalization parameters
        """

        # Refresh the list

        self._update_parameters()

        # Filter selecting only normalization parameters

        normalization_parameters_dictionary = collections.OrderedDict()

        for parameter_name, parameter in self._parameters.iteritems():

            if parameter.normalization:
                normalization_parameters_dictionary[parameter_name] = parameter

        return normalization_parameters_dictionary

    @property
    def not_normalization_parameters(self):
        """
        Get a dictionary with all the parameters that are not normalization in this model
        :return: dictionary of not normalization parameters
        """

        # Refresh the list

        self._update_parameters()

        # Filter selecting only normalization parameters

        normalization_parameters_dictionary = collections.OrderedDict()

        for parameter_name, parameter in self._parameters.iteritems():

            if parameter.normalization == False:
                normalization_parameters_dictionary[parameter_name] = parameter

        return normalization_parameters_dictionary

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

    def set_initial_SAA_amplitudes(self, norm_array):
        """
        Sets the initial normalization of the saa_sources
        :param norm_array:
        :return:
        """
        for i, saa_source in enumerate(self._saa_sources.itervalues()):

                saa_source.parameters['A-%s' % i].value = norm_array[i]


    def set_initial_continuum_amplitudes(self, norm_array):
        """
        Sets the initial normalization of the continuum sources
        :param norm_array:
        :return:
        """
        for i, continuum_source in enumerate(self._continuum_sources.itervalues()):

            for j, parameter in enumerate(continuum_source.parameters.itervalues()):

                parameter.value = norm_array[i]

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

    def get_continuum_counts(self, id, time_bins, saa_mask):
        """
        
        :param id: 
        :param time_bins:
        :return: 
        """
        source_counts = self._continuum_sources.values()[id].get_counts(time_bins)[:, 0]

        # The SAA sections will be set to zero if a saa_mask is provided
        if saa_mask is not None:
            assert len(time_bins) == len(saa_mask), "The time_bins and saa_mask should be of equal length"
            source_counts[np.where(~saa_mask)] = 0.

        return source_counts

    def get_flare_counts(self, id, time_bins, saa_mask):
        """
        
        :param time_bins:
        :param id:
        :param t: 
        :return: 
        """
        source_counts = self._flare_sources.values()[id].get_counts(time_bins)[:, 0]

        # The SAA sections will be set to zero if a saa_mask is provided
        if saa_mask is not None:
            assert len(time_bins) == len(saa_mask), "The time_bins and saa_mask should be of equal length"
            source_counts[np.where(~saa_mask)] = 0.

        return source_counts

    def get_point_source_counts(self, time_bins, saa_mask):
        """
        
        :param time_bins:
        :param id:
        :param t: 
        :return: 
        """
        source_counts = np.zeros(len(time_bins))

        for i, point_source in enumerate(self._point_sources):
            source_counts += self._point_sources.values()[i].get_counts(time_bins)[:, 0]

        # The SAA sections will be set to zero if a saa_mask is provided
        if saa_mask is not None:
            assert len(time_bins) == len(saa_mask), "The time_bins and saa_mask should be of equal length"
            source_counts[np.where(~saa_mask)] = 0.

        return source_counts

    def get_saa_counts(self, time_bins, saa_mask):
        """

        :param time_bins:
        :param id:
        :param t:
        :return:
        """
        source_counts = np.zeros(len(time_bins))

        for i, saa in enumerate(self._saa_sources):
            source_counts += self._saa_sources.values()[i].get_counts(time_bins)[:, 0]

        # The SAA sections will be set to zero if a saa_mask is provided
        if saa_mask is not None:
            assert len(time_bins) == len(saa_mask), "The time_bins and saa_mask should be of equal length"
            source_counts[np.where(~saa_mask)] = 0.

        return source_counts

    def add_SAA_regions(self, *regions):
        """
        Add SAA temporal regions which cause the model to be set to zero
        
        :param regions: 
        :return: 
        """

    def get_counts(self, time_bins, bin_mask=None, saa_mask=None):
        """
        Calculates the counts for all sources in the model and returns the summed up array.
        Only one of the following usecases can be used!
        1) The bin_mask serves for masking the saa sections for faster fitting
        2) The saa_mask sets the SAA sections to zero when the counts for all time bins are returned

        :param time_bins:
        :param bin_mask:
        :return:
        """
        if bin_mask is not None:
            assert saa_mask is None, "There should only be a bin mask or a saa_mask provided"

        if bin_mask is None:
            bin_mask = np.ones(len(time_bins), dtype=bool)  # np.full(len(time_bins), True)

        total_counts = np.zeros(len(time_bins))

        for continuum_source in self._continuum_sources.values():
            total_counts += continuum_source.get_counts(time_bins, bin_mask)[:, 0]

        for flare_source in self._flare_sources.values():
            total_counts += flare_source.get_counts(time_bins, bin_mask)[:, 0]

        for point_source in self._point_sources.values():
            total_counts += point_source.get_counts(time_bins, bin_mask)[:, 0]

        for saa_source in self._saa_sources.values():
            total_counts += saa_source.get_counts(time_bins, bin_mask)[:, 0]

        # The SAA sections will be set to zero if a saa_mask is provided
        if saa_mask is not None:
            assert len(time_bins) == len(saa_mask), "The time_bins and saa_mask should be of equal length"
            total_counts[np.where(~saa_mask)] = 0.

        return total_counts

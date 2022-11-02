import numpy as np

from gbmbkgpy.utils.likelihood import cstat_numba


def check_valid_source_name(source, source_list):
    """
    check if the source is already in the list
    """
    for s in source_list:
        if s.name == source.name:
            raise AssertionError("Two sources with the same names")


class ModelDet:

    def __init__(self, data):
        """

        """
        self._data = data
        self._sources = []

    def add_source(self, source):
        """
        Add a photon source - shared between all dets and echans
        """
        check_valid_source_name(source, self._sources)

        # set time bins for source
        source.set_time_bins(self._data.time_bins)

        # add to list
        self._sources.append(source)

        # update current parameters
        self.update_current_parameters()

    def get_model_counts_given_source(self, source_name_list: list, bin_mask=None):
        counts = np.zeros_like(self._data.counts, dtype=float)
        for name in source_name_list:
            found = False
            for source in self._sources:
                if name == source.name:
                    counts += source.get_counts(bin_mask)
                    found = True
                    break
            if not found:
                source_names = self.get_source_names()
                raise AssertionError(f"No source with the name {name}"
                                     "Sources with the following names exist:"
                                     f"{source_names}")
        return counts

    def get_source_names(self):
        names = []
        for source in self._sources:
            names.append(source.name)
        return names

    def get_model_counts(self, bin_mask=None):
        counts = np.zeros_like(self._data.counts, dtype=float)
        for source in self._sources:
            counts += source.get_counts(bin_mask)
        return counts

    def log_like(self):
        return cstat_numba(self.get_model_counts(), self._data.counts)

    def update_current_parameters(self):
        # return all parameters
        parameters = {}
        if len(self._sources) > 0:
            for source in self._sources:
                for name, param in source.parameters.items():
                    parameters[f"{source.name}_{name}"] = param
        self._current_parameters = parameters

    def generate_counts(self):
        """
        Generate counts in time bins as poisson draw of the
        precicted model rates
        """
        model_counts = self.get_model_counts()
        return np.random.poisson(model_counts)

    @property
    def current_parameters(self):
        return self._current_parameters


class ModelAll:

    def __init__(self, *model_dets):
        self._model_dets: ModelDet = model_dets

    def log_like(self):
        log_like = 0
        for model in self._model_dets:
            log_like += model.log_like
        return log_like

    @property
    def parameter(self):
        parameters = {}
        for model in self._model_dets:
            for name, param in model.parameters.items():
                parameters[name] = param
        return parameters

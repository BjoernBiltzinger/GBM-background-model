import collections
import random
import numpy as np
from datetime import datetime

import pymultinest

from gbmbkgpy.utils.mpi import check_mpi
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.likelihood import cstat_numba

using_mpi, rank, size, comm = check_mpi()

def check_valid_source_name(source, source_list):
    """
    check if the source is already in the list
    """
    for s in source_list:
        if s.name == source.name:
            raise AssertionError("Two sources with the same names")

def create_output_dir():

    base = get_path_of_external_data_dir()
    output_dir = (base / "fits" / "mn_out" /
                  datetime.now().strftime("%m-%d_%H-%M"))


    # If the output path is to long (MultiNest only supports 100 chars)
    # we will create a symbolic link with a random directory name
    # and remove it when MultiNest finished.

    if len(str(output_dir)) > 72:

        tmp_output_dir = (base / "fits" / "mn_out" /
                          str(random.getrandbits(16)))
    else:

        tmp_output_dir = output_dir

    # Create output dir for multinest if not existing

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        if tmp_output_dir != output_dir:

            tmp_output_dir.symlink_to(output_dir)

    return output_dir, tmp_output_dir

def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l, r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return min([np.where(a == left)[0][0], np.where(a == right)[0][0]])

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

    def log_like(self):
        return cstat_numba(self.get_model_counts(), self._data.counts)

    def log_prior(self, trial_values) -> float:
        """Compute the sum of log-priors, used in the parallel tempering sampling"""

        # Compute the sum of the log-priors

        log_prior = 0

        for i, (parameter_name, parameter) in enumerate(self.parameter.items()):

            prior_value = parameter.prior(trial_values[i])

            if prior_value == 0:
                # Outside allowed region of parameter space

                return -np.inf

            else:

                parameter.value = trial_values[i]

                log_prior += np.log(prior_value)

        return log_prior

    def minimize_multinest(
        self,
        n_live_points=400,
        const_efficiency_mode=False
    ):
        """ Multinest Fit """
        #assert (
        #    has_pymultinest
        #), "You need to have pymultinest installed to use this function"
        # We need to wrap the function, because multinest maximizes instead of minimizing
        def func_wrapper(values, ndim, nparams):
            # values is a wrapped C class. Extract from it the values in a python list
            values_list = [values[i] for i in range(ndim)]
            self.set_parameters(values_list)
            return self.log_like() * (-1)

        # priors
        prior = self._get_multinest_prior()

        # output dir
        output_dir, tmp_output_dir = create_output_dir()

        # Run PyMultiNest
        sampler = pymultinest.run(
            func_wrapper,
            prior,
            len(self.parameter),
            len(self.parameter),
            n_live_points=n_live_points,
            outputfiles_basename=str((tmp_output_dir / "fit_").absolute()),
            multimodal=True,  # True was default
            resume=True,
            verbose=True,  # False was default
            importance_nested_sampling=False,
            const_efficiency_mode=const_efficiency_mode,
        )

        # Store the sample for further use (if needed)
        self._sampler = sampler

        if rank == 0:

            # If we used a temporary output dir then remove the symbolic link
            if tmp_output_dir != output_dir:
                tmp_output_dir.unlink()


            # analyse : taken from 3ML
            multinest_analyzer = pymultinest.analyse.Analyzer(
                n_params=len(self.parameter),
                outputfiles_basename=str((tmp_output_dir / "fit_").absolute())
            )

            self._raw_samples =\
                multinest_analyzer.get_equal_weighted_posterior()[
                    :, :-1
                ]

            self._samples = collections.OrderedDict()

            for i, parameter_name in enumerate(self.parameter.keys()):
                # Add the samples for this parameter for this source

                self._samples[parameter_name] = self._raw_samples[:, i]


            # Get the log. likelihood values from the chain
            log_like_values = multinest_analyzer.get_equal_weighted_posterior()[
                :, -1
            ]

            # now get the log probability
            self._log_probability_values = log_like_values + np.array(
                [self.log_prior(samples) for samples in self._raw_samples]
            )

    def load_fit(self, output_dir):
        """
        Only works if the fitted model was created exactly like the model
        here. Same sources & same order!
        """
        # analyse : taken from 3ML
        if not isinstance(output_dir, str):
            output_dir = str(output_dir.absolute())

        multinest_analyzer = pymultinest.analyse.Analyzer(
            n_params=len(self.parameter),
            outputfiles_basename=output_dir
        )

        self._raw_samples =\
            multinest_analyzer.get_equal_weighted_posterior()[
                :, :-1
            ]

        self._samples = collections.OrderedDict()

        for i, parameter_name in enumerate(self.parameter.keys()):
            # Add the samples for this parameter for this source

            self._samples[parameter_name] = self._raw_samples[:, i]

        # Get the log. likelihood values from the chain
        log_like_values = multinest_analyzer.get_equal_weighted_posterior()[
            :, -1
        ]

        # now get the log probability
        self._log_probability_values = log_like_values + np.array(
            [self.log_prior(samples) for samples in self._raw_samples]
        )

    def get_model_counts_given_source(self, source_name_list: list,
                                      bin_mask=None):
        counts = np.zeros_like(self._data.counts, dtype=float)
        for name in source_name_list:
            found = False
            for source in self._sources:
                if name == source.name:
                    counts += source.get_counts(bin_mask)
                    found = True
                    break
            if not found:
                raise AssertionError(f"No source with the name {name}"
                                     "Sources with the following names exist:"
                                     f"{self.source_names}")
        return counts

    def _get_multinest_prior(self):
        """
        Here, we construct the prior.
        """

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self.parameter.items()):

                try:

                    params[i] = parameter.prior.from_unit_cube(
                        params[i]
                    )

                except AttributeError:

                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with multinest" % parameter_name
                    )

                    # Give a test run to the prior to check that it is working. If it crashes while multinest is going

        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self.parameter.items())

        _ = prior([0.5] * n_dim, n_dim, [])

        return prior

    def get_model_counts(self, bin_mask=None):
        counts = np.zeros_like(self._data.counts, dtype=float)
        for source in self._sources:
            counts += source.get_counts(bin_mask)
        return counts

    def set_parameters(self, values):
        """
        Set parameters to values in the array values
        """
        for v, param in zip(values, self.parameter.values()):
            param.value = v

    def update_current_parameters(self):
        # update the dict with the parameters from all sources saved
        parameters = {}
        if len(self._sources) > 0:
            for source in self._sources:
                for name, param in source.parameters.items():
                    parameters[f"{source.name}_{name}"] = param
        self._current_parameters = parameters

    def set_samples(self, samples):
        self._samples = samples

    def set_raw_samples(self, raw_samples):
        self._raw_samples = raw_samples

    def set_log_probability_values(self, log_probability_values):
        self._log_probability_values = log_probability_values

    def set_parameter_median(self):
        idx = arg_median(self._log_probability_values)
        self.set_parameters(self._raw_samples[idx, :])

    @property
    def source_names(self):
        names = []
        for source in self._sources:
            names.append(source.name)
        return names

    @property
    def parameter(self):
        return self._current_parameters


class ModelCombine(ModelDet):

    def __init__(self, *model_dets):
        self._model_dets: ModelDet = model_dets
        self._sampler = None

    def log_like(self):
        log_like = 0
        for model in self._model_dets:
            log_like += model.log_like()
        return log_like

    @property
    def parameter(self):
        parameters = {}
        for model in self._model_dets:
            for name, param in model.parameter.items():
                parameters[name] = param
        return parameters

    def minimize_multinest(
        self,
        n_live_points=400,
        const_efficiency_mode=False
    ):

        super().minimize_multinest(n_live_points=n_live_points,
                                   const_efficiency_mode=const_efficiency_mode)

        self.send_samples_to_submodels()

    def load_fit(self, output_dir):
        """
        Only works if the fitted model was created exactly like the model
        here. Same sources & same order!
        """
        super().load_fit(output_dir)

        self.send_samples_to_submodels()

    def send_samples_to_submodels(self):
        if rank == 0:
            # send subsets of samples to the indiv. models of the different
            # dets
            for model in self._model_dets:
                samples = collections.OrderedDict()
                # loop over model det parameters and add them
                for param_name in model.parameter.keys():
                    samples[param_name] = self._samples[param_name]
                    num_samples = len(self._samples[param_name])
                # summarize them in the raw_samples array
                raw_samples = np.zeros((num_samples, len(samples)))
                for i, s in enumerate(samples.values()):
                    raw_samples[:, i] = s

                model.set_samples(samples)
                model.set_raw_samples(raw_samples)

                model.set_log_probability_values(self._log_probability_values)

    @property
    def model_dets(self):
        return self._model_dets

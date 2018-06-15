from astromodels.functions.priors import Uniform_prior, Log_uniform_prior
import pymultinest
import numpy as np
import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
import json
import collections
import math


class MultiNestFit(object):
    def __init__(self, likelihood, parameters):

        self._likelihood = likelihood

        self._day = self._likelihood._data._day
        self._det = self._likelihood._data._det
        self._echan = self._likelihood._echan
        self.parameters = parameters

        self._n_dim = len(self._likelihood._free_parameters)

        # We need to wrap the function, because multinest maximizes instead of minimizing
        def func_wrapper(values, ndim, nparams):
            # values is a wrapped C class. Extract from it the values in a python list
            values_list = [values[i] for i in range(ndim)]
            return self._likelihood(values_list) * (-1)

        # First build a uniform prior for each parameters
        self._param_priors = collections.OrderedDict()

        for parameter_name in self.parameters:

            min_value, max_value = self.parameters[parameter_name].bounds

            assert min_value is not None, "Minimum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            assert max_value is not None, "Maximum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            # Compute the difference in order of magnitudes between minimum and maximum

            if min_value > 0:

                orders_of_magnitude_span = math.log10(max_value) - math.log10(min_value)

                if orders_of_magnitude_span > 2:

                    # Use a Log-uniform prior
                    self._param_priors[parameter_name] = Log_uniform_prior(lower_bound=min_value, upper_bound=max_value)

                else:

                    # Use a uniform prior
                    self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

            else:

                # Can only use a uniform prior
                self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self.parameters.items()):

                try:

                    params[i] = self._param_priors[parameter_name].from_unit_cube(params[i])

                except AttributeError:

                    raise RuntimeError("The prior you are trying to use for parameter %s is "
                                       "not compatible with multinest" % parameter_name)

        # Give a test run to the prior to check that it is working. If it crashes while multinest is going
        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self.parameters)

        _ = prior([0.5] * n_dim, n_dim, [])

        # declare local likelihood_wrapper object:
        self._loglike = func_wrapper
        self._prior = prior

    def fit(self):
        output_dir = os.path.join(get_path_of_external_data_dir(), 'fits', 'multinest_out/')

        # Run PyMultiNest
        sampler = pymultinest.run(self._loglike,
                                  self._prior,
                                  self._n_dim,
                                  self._n_dim,
                                  outputfiles_basename=output_dir,
                                  multimodal=True,
                                  resume=False)

        ## Use PyMULTINEST analyzer to gather parameter info
        multinest_analyzer = pymultinest.analyse.Analyzer(n_params=self._n_dim,
                                                          outputfiles_basename=output_dir)

        # Get the function value from the chain
        func_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

        # Store the sample for further use (if needed)

        self._sampler = sampler

        # Get the samples from the sampler

        _raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]

        # Find the minimum of the function (i.e. the maximum of func_wrapper)

        idx = func_values.argmax()

        best_fit_values = _raw_samples[idx]

        minimum = func_values[idx] * (-1)

        return best_fit_values, minimum

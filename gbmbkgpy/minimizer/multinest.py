import pymultinest
import numpy as np
import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
import json


class MultiNestFit(object):
    def __init__(self, likelihood):

        self._likelihood = likelihood

        self._day = self._likelihood._data._day
        self._det = self._likelihood._data._det
        self._echan = self._likelihood._echan

        self._nparams = len(self._likelihood._free_parameters)

        def func_wrapper(values, ndim, nparams):
            return self._likelihood(values)

        def prior(cube, ndim, nparams):
            for n in range(ndim):
                cube[n] = cube[n] * 10 ** 4  # delcare linear prior up to 1e+10

        # declare local likelihood_wrapper object:
        self._loglike_function, self._param_priors = func_wrapper, prior

    def fit(self):
        output_dir = os.path.join(get_path_of_external_data_dir(), 'fits', 'multinest_out/')

        # Run PyMultiNest
        sampler = pymultinest.run(self._loglike_function,
                                  self._param_priors,
                                  self._nparams,
                                  self._nparams,
                                  outputfiles_basename=output_dir,
                                  multimodal=True,
                                  resume=False)
        # Save parameter names
        param_index = []
        for i, parameter in enumerate(self._likelihood._parameters.values()):
            param_index.append(parameter.name)
        self._param_names = param_index

        json.dump(self._param_names, open(output_dir + 'params.json', 'w'))

        # read in results
        analyzer = pymultinest.Analyzer(self._nparams, outputfiles_basename=output_dir)
        best_fit_params = analyzer.get_best_fit()['parameters']
        self._sampler = sampler

        return best_fit_params
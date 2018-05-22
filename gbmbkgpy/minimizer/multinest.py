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

    def likelihood_wrapper(self, cube, ndim, nparams):
        return self._likelihood(cube)

    def prior(self, cube, ndim, nparams):

        for n in range(ndim):
            cube[n] = 10 ** (cube[n] * 16 - 8)  # log-uniform prior between 10^-8 and 10^8

    def fit(self):

        output_dir = os.path.join(get_path_of_external_data_dir(), 'fits', 'multinest_out/')

        # Run PyMultiNest
        pymultinest.run(self.likelihood_wrapper, self.prior, self._nparams, outputfiles_basename=output_dir, resume=False, verbose=True)

        # Save parameter names
        param_index = []
        for i, parameter in enumerate(self._likelihood._parameters.values()):
            param_index.append(parameter.name)
        self._param_names = param_index

        json.dump(self._param_names, open(output_dir + 'params.json', 'w'))

        # read in results
        analyzer = pymultinest.Analyzer(self._nparams, outputfiles_basename=output_dir)
        best_fit_params = analyzer.get_best_fit()['parameters']

        return best_fit_params
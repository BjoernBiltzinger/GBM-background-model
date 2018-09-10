import numpy as np
from scipy.optimize import minimize, basinhopping
import pandas as pd
from datetime import datetime
import json
import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir


class Minimizer(object):

    def __init__(self, likelihood):

        self._likelihood = likelihood
        self._result_steps = {}
        self._fitted_params_steps = {}
        self._fitted_params = {}

        self._day = self._likelihood._data._day
        self._det = self._likelihood._data._det


    def fit(self, n_interations = 6, method_1 = 'L-BFGS-B',  method_2 = 'Powell'):
        """
        Fits the model stepwise by calling the scipy.minimize in the following steps:
        1. Fit linear/normalization paramters
        2. Fit SAA parameters
        3. Fit all parameters with bounds
        4. Fit all parameters without bounds multiple times: Powell
        5. Fit all parameters without bounds and high precision: Powell
        :param n_interations:
        :return:
        """

        start = datetime.now()

        # method_1 = 'L-BFGS-B'
        # #method_1 = 'TNC'
        # method_2 = 'Powell'

        # First do the linear fit for normalizations and fix the other parameters
        self._likelihood.fix_parameters(self._likelihood.get_not_normalization_parameter_list)
        self._fit_with_bounds(method_1, type="linear", iter_nr=1)

        if self._likelihood.use_SAA:
            # Fix the normalizations and fit for the other parameters
            self._likelihood.fix_parameters(self._likelihood.get_normalization_parameter_list[0:4])
            self._likelihood.unfix_parameters(self._likelihood.get_not_normalization_parameter_list)
            self._fit_with_bounds(method_1, type="SAA", iter_nr=2)

        # Unfix all parameters and fit all with bounds
        self._likelihood.unfix_parameters(self._likelihood.get_normalization_parameter_list)
        self._fit_with_bounds(method_1, type="full constrained", iter_nr=3)

        # Fit all parameters without bounds in three runs to improve speed and accuracy
        if n_interations > 3:
            for i in range(4, n_interations):
                self._fit_without_bounds(method_2, iter_nr=i, options={})

            # Final run with improved accuracy
            self._fit_without_bounds(method_2, iter_nr=n_interations, options={'xtol': 0.000001, 'ftol': 0.000001})

        self.result = self._result_steps['%s' % n_interations]


        print ("The total Optimization took: {}".format(datetime.now() - start))

        print ("The Optimization ended with message:  {}".format(self.result.message))

        print ("Success = {}".format(self.result.success))

        # save the fit results and errors

        self._save_fits_file()

        # display the results

        print self.display()

        return self.result

    def _fit_with_bounds(self, method='L-BFGS-B', type='bounded', iter_nr=1, ftol=1e-9):

        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        bounds = self._likelihood.get_free_parameter_bounds
        self._result_steps[str(iter_nr)] = minimize(self._likelihood, start_params, method=method, bounds=bounds,
                                           options={'maxiter': 15000, 'gtol': 1e-10, 'ftol': ftol})

        self._build_fit_param_df('Fit-'+str(iter_nr))
        print ("{}. The {} optimization took: {}".format(str(iter_nr), type, datetime.now() - step))

    def _fit_without_bounds(self, method='Powell', iter_nr=1, options={}):
        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        self._result_steps[str(iter_nr)] = minimize(self._likelihood, start_params, method=method, options=options)
        self._build_fit_param_df('Fit-' + str(iter_nr))
        print ("{}. The {}st unconstrained optimization took: {}".format(iter_nr, iter_nr - 3, datetime.now() - step))

    def _fit_basinhopping(self, iter_nr):
        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        self._result_steps[str(iter_nr)] = basinhopping(self._likelihood, start_params)
        self._build_fit_param_df('Fit-' + str(iter_nr))
        print ("{}. The basinhopping optimization took: {}".format(iter_nr, iter_nr - 3, datetime.now() - step))

    def _save_fits_file(self):

        data = {}

        data['fitted-param-steps'] = {'param-names': self._param_index, 'param-values': self._fitted_params}

        data['fit-result'] = {'param-names': [], 'param-values': []}

        for i, parameter in enumerate(self._likelihood._parameters.itervalues()):
            data['fit-result']['param-names'].append(parameter.name)
            data['fit-result']['param-values'].append(parameter.value)

        folder_path = os.path.join(get_path_of_external_data_dir(), 'fits')

        # create directory if it doesn't exist
        if not os.access(folder_path, os.F_OK):
            print("Making New Directory")
            os.mkdir(folder_path)

        file_number = 0
        file_name = 'Fit_' + str(self._day) + '_' + str(self._det) + '_' + str(
            file_number) + '.json'

        # If file already exists increase file number
        while os.path.isfile(os.path.join(folder_path, file_name)):
            file_number += 1
            file_name = 'Fit_' + str(self._day) + '_' + str(self._det)  + '_' + str(
                file_number) + '.json'

        # Writing JSON data
        with open(os.path.join(folder_path, file_name), 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def display(self, label = "fitted_value"):
        """
        display the results using pandas series or dataframe
        
        
        :param self: 
        :return: 
        """

        self._fit_params = {}
        self._fit_params['parameter'] = []
        self._fit_params[label] = []

        for i, parameter in enumerate(self._likelihood._parameters.itervalues()):
            self._fit_params['parameter'].append(parameter.name)
            self._fit_params[label].append(parameter.value)

        self.fitted_params = pd.DataFrame(data=self._fit_params)

        return self.fitted_params

    def _build_fit_param_df(self, label):

        param_index = []
        self._fitted_params[label] = []

        for i, parameter in enumerate(self._likelihood._parameters.itervalues()):
            param_index.append(parameter.name)
            self._fitted_params[label].append(parameter.value)

        self._param_index = param_index

    @property
    def fitted_param_steps(self):
        return pd.DataFrame(data=self._fitted_params, index=self._param_index)

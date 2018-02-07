import numpy as np
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime


class Minimizer(object):

    def __init__(self, likelihood):

        self._likelihood = likelihood
        self._result_steps = {}

    def fit(self, n_interations = 6):
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

        method_1 = 'L-BFGS-B'
        #method_1 = 'TNC'
        method_2 = 'Powell'

        # First do the linear fit for normalizations and fix the other parameters
        self._likelihood.fix_parameters(self._likelihood.get_not_normalization_parameter_list)
        self._fit_with_bounds(method_1, type="linear", iter_nr=1)

        # Fix the normalizations and fit for the other parameters
        self._likelihood.fix_parameters(self._likelihood.get_normalization_parameter_list[0:4])
        self._likelihood.unfix_parameters(self._likelihood.get_not_normalization_parameter_list)
        self._fit_with_bounds(method_1, type="SAA", iter_nr=2)

        # Unfix all parameters and fit all with bounds
        self._likelihood.unfix_parameters(self._likelihood.get_normalization_parameter_list)
        self._fit_with_bounds(method_1, type="full constrained", iter_nr=3)

        # Fit all parameters without bounds in three runs to improve speed and accuracy
        if n_interations > 3:
            for i in range(4, n_interations-1):
                self._fit_without_bounds(method_2, iter_nr=i, options={})

            # Final run with improved accuracy
            self._fit_without_bounds(method_2, iter_nr=n_interations, options={'xtol': 0.000001, 'ftol': 0.000001})

        self.result = self._result_steps['%s' % n_interations]


        print "The total Optimization took: {}".format(datetime.now() - start)

        print "The Optimization ended with status:  {}".format(self.result.status)

        print "Success = {}".format(self.result.success)

        # save the fit results and errors

        self._save()

        # display the results

        self.display()

        return self.result

    def _fit_with_bounds(self, method, type, iter_nr):

        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        bounds = self._likelihood.get_free_parameter_bounds
        self._result_steps[str(iter_nr)] = minimize(self._likelihood, start_params, method=method, bounds=bounds,
                                           options={'maxiter': 10000, 'gtol': 1e-08, 'ftol': 1e-10})
        print "{}. The {} optimization took: {}".format(str(iter_nr), type, datetime.now() - step)

    def _fit_without_bounds(self, method, iter_nr, options):
        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        self._result_steps[str(iter_nr)] = minimize(self._likelihood, start_params, method=method, options=options)
        print "{}. The {}st unconstrained optimization took: {}".format(iter_nr, iter_nr - 3, datetime.now() - step)

    def _save(self):

        pass


    def display(self):
        """
        display the results using pandas series or dataframe
        
        
        :param self: 
        :return: 
        """

        data_dic = {}

        for i, parameter in enumerate(self._likelihood._parameters.itervalues()):
                    data_dic[parameter.name] = {}
                    data_dic[parameter.name]['fitted value'] = parameter.value
        fittet_params = pd.DataFrame.from_dict(data_dic, orient='index')



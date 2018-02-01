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
        4. Fit all parameters without bounds: Powell
        5. Fit all parameters without bounds: Powell
        6. Fit all parameters without bounds and high precision: Powell
        :param n_interations:
        :return:
        """

        start = datetime.now()

        # First do the linear fit for normalizations and fix the other parameters
        self._likelihood.fix_parameters(self._likelihood.get_not_normalization_parameter_list)

        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        bounds = self._likelihood.get_free_parameter_bounds
        self._result_steps['1'] = minimize(self._likelihood, start_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'gtol': 1e-08, 'ftol':1e-10})
        print "1. The linear optimization took: {}".format(datetime.now() - step)


        # Fix the normalizations and fit for the other parameters
        self._likelihood.fix_parameters(self._likelihood.get_normalization_parameter_list[0:4])
        self._likelihood.unfix_parameters(self._likelihood.get_not_normalization_parameter_list)

        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        bounds = self._likelihood.get_free_parameter_bounds
        self._result_steps['2'] = minimize(self._likelihood, start_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'gtol': 1e-08, 'ftol':1e-10})
        print "2. The SAA optimization took: {}".format(datetime.now() - step)


        # Unfix all parameters and fit all with bounds
        self._likelihood.unfix_parameters(self._likelihood.get_normalization_parameter_list)

        step = datetime.now()
        start_params = self._likelihood.get_free_parameter_values
        bounds = self._likelihood.get_free_parameter_bounds
        self._result_steps['3'] = minimize(self._likelihood, start_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'gtol': 1e-08, 'ftol':1e-10})
        print "3. The full constrained optimization took: {}".format(datetime.now() - step)

        if n_interations > 3:
            # Fit all parameters without bounds in three runs to improve speed and accuracy
            step = datetime.now()
            start_params = self._likelihood.get_free_parameter_values
            self._result_steps['4'] = minimize(self._likelihood, start_params, method='Powell', options={})
            print "4. The 1st unconstrained optimization took: {}".format(datetime.now() - step)

        if n_interations > 4:
            step = datetime.now()
            start_params = self._likelihood.get_free_parameter_values
            self._result_steps['5'] = minimize(self._likelihood, start_params, method='Powell', options={})
            print "5. The 2nd unconstrained optimization took: {}".format(datetime.now() - step)

        if n_interations > 5:
            #Final run with improved accuracy
            step = datetime.now()
            start_params = self._likelihood.get_free_parameter_values
            self._result_steps['6'] = minimize(self._likelihood, start_params, method='Powell', options={'xtol': 0.000001, 'ftol': 0.000001})
            print "6. The 3rd unconstrained optimization took: {}".format(datetime.now() - step)
            self.result = self._result_steps['6']
        print "The total Optimization took: {}".format(datetime.now() - start)

        if n_interations < 6:
            self.result = self._result_steps['%s' % n_interations]
        # save the fit results and errors

        self._save()

        # display the results

        self.display()

        return self.result

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



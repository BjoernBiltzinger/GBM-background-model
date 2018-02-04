import numpy as np
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.utils.statistics.stats_tools import Significance
import copy

class BackgroundLike(object):

    def __init__(self, data, model, echan):
        """
        
        :param data: 
        :param model: 
        """
        self._data = data #type: ContinuousData

        self._model = model #type: Model

        self._free_parameters = self._model.free_parameters

        self._parameters = self._model.parameters

        self._echan = echan

        self._total_scale_factor = 1.

        #TODO: the data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins[2:-2]
        self._saa_mask = self._data.saa_mask[2:-2]
        self._time_bins = self._data.time_bins[self._data.saa_mask][2:-2]

        #TODO: extract the counts from the data object. should be same size as time bins
        self._counts = self._data.counts[:, echan][self._data.saa_mask][2:-2]


    def _evaluate_model(self):
        """
        
        loops over time bins and extracts the model flux and returns this array
        
        
        :return: 
        """

        model_counts = self._model.get_counts(self._time_bins, self._saa_mask)

        """ OLD:
        model_flux = []
        
        for bin in self._time_bins:
            model_flux.append(self._model.get_flux(bin[0], bin[1]))
        """

        return model_counts

    def _set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        for i, parameter in enumerate(self._free_parameters.itervalues()):

            parameter.value = new_parameters[i]

    @property
    def get_normalization_parameter_list(self):
        """
        Gets a list of the parameter names in the model which are for normalization
        :return:
        """

        norm_param_list = []

        for parameter_name in self._model.normalization_parameters:

            norm_param_list.append(parameter_name)

        return norm_param_list

    @property
    def get_not_normalization_parameter_list(self):
        """
        Gets a list of the parameter names in the model which are NOT for normalization
        :return: not_norm_param_list
        """

        not_norm_param_list = []

        for parameter_name in self._model.not_normalization_parameters:

            not_norm_param_list.append(parameter_name)

        return not_norm_param_list

    def fix_parameters(self, parameter_names):
        """
        Fixe the parameters to their value
        :param parameter_names:
        :return:
        """

        for param_name in parameter_names:

            parameter_exits = False

            for parameter_name in self._parameters:

                if param_name == parameter_name:

                    self._parameters[param_name]._free = False

                    parameter_exits = True

                    #print ("Parameter {0} has been fixed".format(param_name))

            if parameter_exits == False:
                print ("Parameter does not exist in parameter list")

        # update the free parameter list
        self._free_parameters = self._model.free_parameters

    def unfix_parameters(self, parameter_names):
        """
        Unfix the parameters
        :param parameter_names:
        :return:
        """

        for param_name in parameter_names:

            parameter_exits = False

            for parameter_name in self._parameters:

                if param_name == parameter_name:
                    self._parameters[param_name]._free = True

                    parameter_exits = True

                    #print ("Parameter {0} has been unfixed".format(param_name))

            if parameter_exits == False:
                print ("Parameter does not exist in parameter list")

        # update the free parameter list
        self._free_parameters = self._model.free_parameters

    @property
    def get_free_parameter_values(self):
        """
        Returns a list with all free parameter values.
        :return:
        """
        param_value_list = []
        for i, parameter in enumerate(self._free_parameters.itervalues()):
            param_value_list.append(parameter.value)

        return param_value_list

    @property
    def get_free_parameter_bounds(self):
        """
        Returns a list with all free parameter bounds.
        :return:
        """
        param_bound_list = []
        for i, parameter in enumerate(self._free_parameters.itervalues()):
            param_bound_list.append(parameter.bounds)

        return param_bound_list

    def get_synthetic_data(self, synth_parameters, synth_model=None):
        """
        Creates a ContinousData object with synthetic data based on the total counts from the synth_model
        If no synth_model is passed it makes a deepcopy of the existing model
        :param synth_parameters:
        :return:
        """

        synth_data = copy.deepcopy(self._data)

        if synth_model == None:
            synth_model = copy.deepcopy(self._model)


        for i, parameter in enumerate(synth_model.free_parameters.itervalues()):
            parameter.value = synth_parameters[i]


        synth_data.counts[:, self._echan][2:-2] = np.random.poisson(synth_model.get_counts(synth_data.time_bins[2:-2]))

        self._synth_model = synth_model

        return synth_data


    def __call__(self, parameters):
        """
        
        :return: the poisson log likelihood
        """

        self._set_free_parameters(parameters)


        M = self._evaluate_model()
        M_fixed, tiny = self._fix_precision(M)

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero

        negative_mask = (M < 0)
        if (len(negative_mask.nonzero()[0]) > 0):
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0



        d_times_logM = self._counts * logM

        log_likelihood = np.sum(M_fixed - d_times_logM)

        return log_likelihood


    def _fix_precision(self, v):
      """
      Round extremely small number inside v to the smallest usable
      number of the type corresponding to v. This is to avoid warnings
      and errors like underflows or overflows in math operations.
    
    
      :param v: 
      :return: 
      """

      tiny = np.float64(np.finfo(v[0]).tiny)
      zero_mask = (np.abs(v) <= tiny)
      if (len(zero_mask.nonzero()[0]) > 0):
          v[zero_mask] = np.sign(v[zero_mask]) * tiny

      return v, tiny

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        non_tiny_mask = (M > 2.0 * tiny)

        tink_mask = np.logical_not(non_tiny_mask)

        if (len(tink_mask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tink_mask] = np.abs(M[tink_mask]) / tiny + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = np.log(M)

        return logM

    def _calc_significance(self):

        rebinned_observed_counts = self._counts
        rebinned_background_counts = np.zeros_like(self._counts)
        rebinned_model_counts = self._model.get_counts(self._time_bins, self._saa_mask)


        significance_calc = Significance(rebinned_observed_counts,rebinned_background_counts + rebinned_model_counts /
                                         self._total_scale_factor, self._total_scale_factor)

        self.residuals = significance_calc.known_background()
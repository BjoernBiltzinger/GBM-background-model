import re
import numpy as np
import numexpr as ne
import numba

from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.modeling.model import Model


class BackgroundLike(object):

    def __init__(self, data, model, saa_object, use_numba=False):
        """
        Init backgroundlike that compares the data with the model
        :param data:
        :param model:
        :param echans:
        """

        self._data = data  # type: Data
        self._model = model  # type: Model
        self._use_numba = use_numba
        # The MET start time of the day

        self._free_parameters = self._model.free_parameters
        self._parameters = self._model.parameters

        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        # Get the SAA and GRB mask:
        self._saa_mask = saa_object.saa_mask
        self._grb_mask = np.ones(len(self._total_time_bins), dtype=bool)  # np.full(len(self._total_time_bins), True)
        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins.
        self._total_counts = self._data.counts
        self._masked_counts = self._data.counts[self._total_mask]

        self._total_scale_factor = 1.
        self._grb_mask_calculated = False

        self._get_sources_fit_spectrum()
        self._build_log_like()


    def _set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        for i, parameter in enumerate(self._free_parameters.values()):
            parameter.value = new_parameters[i]

    def set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        self._model.set_free_parameters(new_parameters)

    def model_counts(self):
        """
        Returns the predicted counts from the model for all time bins,
        the saa_mask sets the SAA sections to zero.
        :return:
        """

        return self._model.get_counts(self._total_time_bins, saa_mask=self._saa_mask)

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
        Fix the parameters to their value
        :param parameter_names:
        :return:
        """

        for param_name in parameter_names:

            parameter_exits = False

            for parameter_name in self._parameters:

                if param_name == parameter_name:
                    self._parameters[param_name]._free = False

                    parameter_exits = True

                    # print ("Parameter {0} has been fixed".format(param_name))

            if not parameter_exits:
                print ("Parameter does not exist in parameter list")

        self.update_free_parameters()

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

                    # print ("Parameter {0} has been unfixed".format(param_name))

            if parameter_exits == False:
                print ("Parameter does not exist in parameter list")

        self.update_free_parameters()

    def update_free_parameters(self):
        """
        Update the free parameter list
        :return:
        """
        self._free_parameters = self._model.free_parameters

    @property
    def get_free_parameter_values(self):
        """
        Returns a list with all free parameter values.
        :return:
        """
        param_value_list = []
        for i, parameter in enumerate(self._free_parameters.values()):
            param_value_list.append(parameter.value)

        return param_value_list

    @property
    def get_free_parameter_bounds(self):
        """
        Returns a list with all free parameter bounds.
        :return:
        """
        param_bound_list = []
        for i, parameter in enumerate(self._free_parameters.values()):
            param_bound_list.append(parameter.bounds)

        return param_bound_list

    def _get_sources_fit_spectrum(self):

        self._sources_fit_spectrum = self._model.fit_spectrum_sources.values()

    def _build_cov_call(self):

        def cov_call(*parameters):
            return self.__call__(parameters)

        self.cov_call = cov_call

    def __call__(self, parameters):
        """
                :return: the poisson log likelihood
                """
        self._set_free_parameters(parameters)

        ######### Calculate rates for new spectral parameter
        for source in self._sources_fit_spectrum:
            source.recalculate_counts()
        ########

        return self._get_log_likelihood()


    def _build_log_like(self):

        if self._use_numba:

            print('Use numba likelihood')

            if self._data.data_type == 'trigdat':

                def log_like_numba():

                    M = self._evaluate_model()

                    counts = self._masked_counts

                    return _log_likelihood_numba_trigdat(M, counts)
            else:

                def log_like_numba():

                    M = selfelf._evaluate_model()

                    counts = self._masked_counts

                    return _log_likelihood_numba(M, counts)

            self._get_log_likelihood = log_like_numba

        else:

            print('Use vectorized likelihood')

            def log_like_vector():

                M = self._evaluate_model()
                # Poisson loglikelihood statistic (Cash) is:
                # L = Sum ( M_i - D_i * log(M_i))

                logM = self._evaluate_logM(M)
                # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
                # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
                # to zero, then overwrite the elements corresponding to D_i > 0

                counts = self._masked_counts
                d_times_logM = ne.evaluate("counts*logM")

                log_likelihood = ne.evaluate("sum(M - d_times_logM)")
                return log_likelihood

            self._get_log_likelihood = log_like_vector


    def _evaluate_model(self):
        """
        Loops over time bins and extracts the model counts and returns this array
        :return:
        """

        return self._model.get_counts(time_bins=self._time_bins, bin_mask=self._total_mask)

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0][0][0]).tiny)

        non_tiny_mask = (M > 2.0 * tiny)

        tink_mask = np.logical_not(non_tiny_mask)

        if (len(tink_mask.nonzero()[0]) > 0):
            logM = np.zeros_like(M)
            logM[tink_mask] = np.abs(M[tink_mask]) / tiny + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = ne.evaluate("log(M)")

        return logM

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

    def set_grb_mask(self, *intervals):
        """
        Sets the grb_mask for the provided intervals to False
        These are intervals specified as "-10 -- 5", "0-10", and so on
        :param intervals:
        :return:
        """

        list_of_intervals = []

        for interval in intervals:
            imin, imax = self._parse_interval(interval)

            list_of_intervals.append([imin, imax])

            bin_exclude = np.logical_and(self._total_time_bins[:, 0] > imin, self._total_time_bins[:, 1] < imax)

            self._grb_mask[np.where(bin_exclude)] = False

        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins.
        self._masked_counts = self._data.counts[self._total_mask]
        self._total_counts = self._data.counts

    def _parse_interval(self, time_interval):
        """
        The following regular expression matches any two numbers, positive or negative,
        like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        :param time_interval:
        :return:
        """
        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def reset_grb_mask(self):
        """

        :return:
        """
        self._grb_mask = np.ones(len(self._total_time_bins), dtype=bool)

        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins.
        self._masked_counts = self._data.counts[self._total_mask]
        self._total_counts = self._data.counts

    @property
    def data(self):
        return self._data


@numba.njit(numba.float64(numba.float64[:,:,:], numba.int64[:,:,:]), parallel=True)
def _log_likelihood_numba(M, counts):
    # Poisson loglikelihood statistic (Cash) is:
    # L = Sum ( M_i - D_i * log(M_i))
    val = 0.
    for i in numba.prange(M.shape[0]):
        for j in numba.prange(M.shape[1]):
            for k in numba.prange(M.shape[2]):
                val += M[i,j,k]-counts[i,j,k]*np.log(M[i,j,k])
    return val

@numba.njit(numba.float64(numba.float64[:,:,:], numba.float64[:,:,:]), parallel=True)
def _log_likelihood_numba_trigdat(M, counts):
    # Poisson loglikelihood statistic (Cash) is:
    # L = Sum ( M_i - D_i * log(M_i))
    val = 0.
    for i in numba.prange(M.shape[0]):
        for j in numba.prange(M.shape[1]):
            for k in numba.prange(M.shape[2]):
                val += M[i,j,k]-counts[i,j,k]*np.log(M[i,j,k])
    return val

import numpy as np
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
from gbmgeometry import GBMTime
import astropy.time as astro_time
import copy
import re
import os
import json
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
import numexpr as ne

import astropy.io.fits as fits

NO_REBIN = 1E-99

class BackgroundLike(object):

    def __init__(self, data, model, echan_list):
        """
        Init backgroundlike that compares the data with the model
        :param data:
        :param model:
        :param echan_list:
        """

        self._data = data       # type: ContinuousData
        self._model = model     # type: Model
        self._echan_list = echan_list #list of all echans which should be fitted

        self._name = "Count rate detector %s" % self._data._det
        # The MET start time of the day

        self._free_parameters = self._model.free_parameters
        self._parameters = self._model.parameters


        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins[2:-2]
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        # Get the SAA and GRB mask:
        self._saa_mask = self._data.saa_mask[2:-2]
        self._grb_mask = np.ones(len(self._total_time_bins), dtype=bool)  # np.full(len(self._total_time_bins), True)
        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins. For all echans together
        self._counts_all_echan = self._data.counts[2:-2][self._total_mask]
        self._total_counts_all_echan = self._data.counts[2:-2]

        self._total_scale_factor = 1.
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False

        self._get_sources_fit_spectrum()
    def _create_rebinner_before_fit(self, min_bin_width):
        """
        This method rebins the observed counts bevore the fitting process.
        The fitting will be done on the rebinned counts afterwards!
        :param min_bin_width:
        :return:
        """
        self._fit_rebinned = True

        self._fit_rebinner = Rebinner(self._total_time_bins, min_bin_width, self._saa_mask)


    def _rebinned_observed_counts_fitting(self):
        """
        :return:
        """
        # Rebinn the observec counts on time
        self._rebinned_observed_counts_fitting_all_echan=[]
        for echan in self._echan_list:
            self._rebinned_observed_counts_fitting_all_echan.append(self._fit_rebinner.rebin(self._total_counts_all_echan[:, echan]))
        self._rebinned_observed_counts_fitting_all_echan=np.array(self._rebinned_observed_counts_fitting_all_echan)

    def _rebinned_model_counts_fitting(self):
        """
        :return:
        """
        # the rebinned expected counts from the model
        self._rebinned_model_counts_fitting_all_echan = []
        for echan in self._echan_list:
            self._rebinned_model_counts_fitting_all_echan.append(self._fit_rebinner.rebin(self.model_counts(echan))[0])
        self._rebinned_model_counts_fitting_all_echan=np.array(self._rebinned_model_counts_fitting_all_echan)

    def _evaluate_model(self, echan):
        """
        Loops over time bins and extracts the model counts and returns this array
        :return: 
        """

        model_counts = self._model.get_counts(self._time_bins, echan, bin_mask=self._total_mask)

        if self._fit_rebinned == True:
            index = int(np.argwhere(self._echan_list == echan))
            return self._rebinned_model_counts_fitting_all_echan[index]

        else:
            return model_counts

    def _set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        for i, parameter in enumerate(self._free_parameters.itervalues()):

            parameter.value = new_parameters[i]


    def set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        self._model.set_free_parameters(new_parameters)

    def model_counts(self, echan):
        """
        Returns the predicted counts from the model for all time bins,
        the saa_mask sets the SAA sections to zero.
        :return:
        """

        return self._model.get_counts(self._total_time_bins, echan, saa_mask=self._saa_mask)

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

        for echan in self._echan_list:
            synth_data.counts[:, echan][2:-2] = np.random.poisson(synth_model.get_counts(synth_data.time_bins[2:-2],echan))

        self._synth_model = synth_model

        return synth_data
    
    def _get_sources_fit_spectrum(self):
        
        self._sources_fit_spectrum = self._model.fit_spectrum_sources.values()
    

    def __call__(self, parameters):
        """
        
        :return: the poisson log likelihood
        """
        self._set_free_parameters(parameters)
        if self._fit_rebinned:
            self._rebinned_observed_counts_fitting()
            self._rebinned_model_counts_fitting()
        log_likelihood_list=[]
        ######### Calculate rates for new spectral parameter
        for source in self._sources_fit_spectrum:
            source.recalculate_counts()
        ########
        for echan in self._echan_list:
            log_likelihood_list.append(self._get_log_likelihood_echan(echan))
        log_likelihood_list=np.array(log_likelihood_list)
        return np.sum(log_likelihood_list)

    def _get_log_likelihood_echan(self, echan):

        M = self._evaluate_model(echan)
        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)
        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0

        # Use rebinned counts if fir_rebinned is set to true:
        if self._fit_rebinned:
            index = int(np.argwhere(self._echan_list==echan))
            d_times_logM = self._rebinned_observed_counts_fitting_all_echan[index] * logM

        else:

            counts = self._counts_all_echan[:,echan]
            d_times_logM = ne.evaluate("counts*logM")

        log_likelihood = ne.evaluate("sum(M - d_times_logM)")
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

            logM = ne.evaluate("log(M)")

        return logM


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

            bin_exclude = np.logical_and(self._time_bins[:, 0] > imin, self._time_bins[:, 1] < imax)

            self._grb_mask[np.where(bin_exclude)] = False


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
        self._grb_mask = np.ones(len(self._time_bins), dtype=bool)  # np.full(len(self._time_bins), True)


    def _read_fits_file(self, date, detector, echan, file_number=0):

        file_name = 'Fit_' + str(date) + '_' + str(detector) + '_' + str(echan) + '_' + str(file_number) + '.json'
        file_path = os.path.join(get_path_of_external_data_dir(), 'fits', file_name)

        # Reading data back
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    def load_fits_file(self, date, detector, echan):

        data = self._read_fits_file(date, detector, echan)

        fit_result = np.array(data['fit-result']['param-values'])

        self.set_free_parameters(fit_result)

        print("Fits file was successfully loaded and the free parameters set")


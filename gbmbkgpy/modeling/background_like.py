import numpy as np
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
import copy
import re
import os
import json
from gbmbkgpy.io.package_data import get_path_of_external_data_dir


NO_REBIN = 1E-99

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

        self._name = "Count rate detector %s" %self._data._det

        self._grb_mask_calculated = False

        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins[2:-2]
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        self._saa_mask = self._data.saa_mask[2:-2]
        self._grb_mask = np.full(len(self._total_time_bins), True)
        # Total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins
        self._counts = self._data.counts[:, echan][2:-2][self._total_mask]
        self._total_counts = self._data.counts[:, echan][2:-2]

        self._rebinner = None



    def _evaluate_model(self):
        """
        
        loops over time bins and extracts the model flux and returns this array
        
        
        :return: 
        """

        model_counts = self._model.get_counts(self._time_bins, bin_mask=self._total_mask)

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


    def _set_grb_mask(self, *intervals):
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


    def _reset_grb_mask(self):
        """

        :return:
        """
        self._grb_mask = np.full(len(self._time_bins), True)

    def _calc_significance(self):

        rebinned_observed_counts = self._counts
        rebinned_background_counts = np.zeros_like(self._counts)
        rebinned_model_counts = self._model.get_counts(self._time_bins, self._saa_mask)


        significance_calc = Significance(rebinned_observed_counts,rebinned_background_counts + rebinned_model_counts /
                                         self._total_scale_factor, self._total_scale_factor)

        self._unbinned_residuals = significance_calc.known_background()


    def display_model(self, data_color='k', model_color='r', step=True, show_data=True, show_residuals=True,
                      show_legend=True, min_bin_width=1E-99, plot_sources=False,
                      **kwargs):

        """
        Plot the current model with or without the data and the residuals. Multiple models can be plotted by supplying
        a previous axis to 'model_subplot'.
        Example usage:
        fig = data.display_model()
        fig2 = data2.display_model(model_subplot=fig.axes)
        :param data_color: the color of the data
        :param model_color: the color of the model
        :param step: (bool) create a step count histogram or interpolate the model
        :param show_data: (bool) show_the data with the model
        :param show_residuals: (bool) shoe the residuals
        :param ratio_residuals: (bool) use model ratio instead of residuals
        :param show_legend: (bool) show legend
        :param min_rate: the minimum rate per bin
        :param model_label: (optional) the label to use for the model default is plugin name
        :param model_subplot: (optional) axis or list of axes to plot to
        :return:
        """


        model_label = "Geometric Background Model"

        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)


        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if (min_bin_width is not NO_REBIN) or (self._rebinner is None):


            this_rebinner = Rebinner(self._total_time_bins, min_bin_width, self._saa_mask)

        else:

            # Use the rebinner already in the data
            this_rebinner = self._rebinner


        # Residuals

        # we need to get the rebinned counts
        self._rebinned_observed_counts, = this_rebinner.rebin(self._total_counts)

        # the rebinned counts expected from the model
        self._rebinned_model_counts, = this_rebinner.rebin(self.model_counts)

        self._rebinned_background_counts = np.zeros_like(self._rebinned_observed_counts)

        self._rebinned_time_bins = this_rebinner.time_rebinned

        self._rebinned_time_bin_widths = np.diff(self._rebinned_time_bins, axis=1)[:, 0]

        significance_calc = Significance(self._rebinned_observed_counts,
                                         self._rebinned_background_counts + self._rebinned_model_counts / self._total_scale_factor,
                                         self._total_scale_factor)


        residual_errors = None
        self._residuals = significance_calc.known_background()


        residual_plot.add_data(np.mean( self._rebinned_time_bins, axis=1),
                                        self._rebinned_observed_counts / self._rebinned_time_bin_widths,
                                        self._residuals,
                                        residual_yerr=residual_errors,
                                        yerr=None,
                                        xerr=None,
                                        label=self._name,
                                        color=data_color,
                                        show_data=show_data)

        # if step:
        #
        #     residual_plot.add_model_step(new_energy_min,
        #                                  new_energy_max,
        #                                  new_chan_width,
        #                                  new_model_rate,
        #                                  label=model_label,
        #                                  color=model_color)
        # else:

        # We always plot the model un-rebinned here

        # Mask the array so we don't plot the model where data have been excluded
        # y = expected_model_rate / chan_width
        y = self.model_counts / self._total_time_bin_widths

        x = np.mean(self._total_time_bins, axis=1)

        residual_plot.add_model(x,
                                y,
                                label=model_label,
                                color=model_color)

        if plot_sources:

            source_list = self._get_list_of_sources(self._total_time_bin_widths)

            residual_plot.add_list_of_sources(x, source_list)

        return residual_plot.finalize(xlabel="Time\n(MET)",
                                      ylabel="Count Rate\n(counts s$^{-1}$",# keV$^{-1}$)",
                                      xscale='linear',
                                      yscale='linear',
                                      show_legend=show_legend)

    def _get_list_of_sources(self, time_bin_width=1.):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """

        source_list = []
        color_list = ['b', 'g', 'c', 'm', 'y', 'k', 'w']

        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, self._total_time_bins, self._saa_mask)
            source_list.append({"label": source_name, "data": data / time_bin_width, "color": color_list[i]})

        saa_data = self._model.get_saa_counts(self._total_time_bins, self._saa_mask)

        source_list.append({"label": "SAA_decays", "data": saa_data / time_bin_width, "color": color_list[i+1]})

        return source_list


    def _read_fits_file(self, date, detector, echan):

        file_name = 'Fit_' + str(date) + '_' + str(detector) + '_' + str(echan) + '.json'
        file_path = os.path.join(get_path_of_external_data_dir(), 'fits', file_name)

        # Reading data back
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    def load_fits_file(self, date, detector, echan):

        data = self._read_fits_file(date, detector, echan)

        fit_result = np.array(data['fit-result']['param-values'])

        self._set_free_parameters(fit_result)

        print("Fits file successfully loaded and parameters set")

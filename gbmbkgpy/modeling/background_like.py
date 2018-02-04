import numpy as np
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
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


    def display_model(self, data_color='k', model_color='r', step=True, show_data=True, show_residuals=True,
                      show_legend=True, min_rate=1E-99, model_label=None,
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

        if model_label is None:
            model_label = "%s Model" % self._name

        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        # energy_min, energy_max = self._rsp.ebounds[:-1], self._rsp.ebounds[1:]

        energy_min = np.array(self._observed_spectrum.edges[:-1])
        energy_max = np.array(self._observed_spectrum.edges[1:])

        chan_width = energy_max - energy_min

        expected_model_rate = self.expected_model_rate

        # figure out the type of data

        src_rate = self.source_rate
        src_rate_err = self.source_rate_error

        # rebin on the source rate

        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if (min_rate is not NO_REBIN) or (self._rebinner is None):


            this_rebinner = Rebinner(src_rate, min_rate, self._mask)

        else:

            # Use the rebinner already in the data
            this_rebinner = self._rebinner

        # get the rebinned counts
        new_rate, new_model_rate = this_rebinner.rebin(src_rate, expected_model_rate)
        new_err, = this_rebinner.rebin_errors(src_rate_err)

        # adjust channels
        new_energy_min, new_energy_max = this_rebinner.get_new_start_and_stop(energy_min, energy_max)
        new_chan_width = new_energy_max - new_energy_min

        # mean_energy = np.mean([new_energy_min, new_energy_max], axis=0)

        # For each bin find the weighted average of the channel center
        mean_energy = []
        delta_energy = [[], []]
        mean_energy_unrebinned = (energy_max + energy_min) / 2.0

        for e_min, e_max in zip(new_energy_min, new_energy_max):


            # Find all channels in this rebinned bin
            idx = (mean_energy_unrebinned >= e_min) & (mean_energy_unrebinned <= e_max)

            # Find the rates for these channels
            r = src_rate[idx]

            if r.max() == 0:

                # All empty, cannot weight
                this_mean_energy = (e_min + e_max) / 2.0


            else:


                # negative src rates cause the energy mean to
                # go outside of the bounds. So we fix negative rates to
                # zero when computing the mean

                idx_negative = r<0.

                r[idx_negative] =0.

                # Do the weighted average of the mean energies
                weights = r / np.sum(r)

                this_mean_energy = np.average(mean_energy_unrebinned[idx], weights=weights)


            # Compute "errors" for X (which aren't really errors, just to mark the size of the bin)

            delta_energy[0].append(this_mean_energy - e_min)
            delta_energy[1].append(e_max - this_mean_energy)
            mean_energy.append(this_mean_energy)

        # Residuals

        # we need to get the rebinned counts
        rebinned_observed_counts, = this_rebinner.rebin(self.observed_counts)

        rebinned_observed_count_errors, = this_rebinner.rebin_errors(self.observed_count_errors)

        # the rebinned counts expected from the model
        rebinned_model_counts = new_model_rate * self._observed_spectrum.exposure

        rebinned_background_counts = np.zeros_like(rebinned_observed_counts)


        significance_calc = Significance(rebinned_observed_counts,
                                         rebinned_background_counts + rebinned_model_counts / self._total_scale_factor,
                                         self._total_scale_factor)


        residual_errors = None
        residuals = significance_calc.known_background()


        residual_plot.add_data(mean_energy,
                               new_rate / new_chan_width,
                               residuals,
                               residual_yerr=residual_errors,
                               yerr=new_err / new_chan_width,
                               xerr=delta_energy,
                               label=self._name,
                               color=data_color,
                               show_data=show_data)

        if step:

            residual_plot.add_model_step(new_energy_min,
                                         new_energy_max,
                                         new_chan_width,
                                         new_model_rate,
                                         label=model_label,
                                         color=model_color)
        else:

            # We always plot the model un-rebinned here

            # Mask the array so we don't plot the model where data have been excluded
            # y = expected_model_rate / chan_width
            y = np.ma.masked_where(~self._mask, expected_model_rate / chan_width)

            x = np.mean([energy_min, energy_max], axis=0)

            residual_plot.add_model(x,
                                    y,
                                    label=model_label,
                                    color=model_color)

        return residual_plot.finalize(xlabel="Energy\n(keV)",
                                      ylabel="Net rate\n(counts s$^{-1}$ keV$^{-1}$)",
                                      xscale='log',
                                      yscale='log',
                                      show_legend=show_legend)

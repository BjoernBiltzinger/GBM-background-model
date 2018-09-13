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


NO_REBIN = 1E-99

class BackgroundLike(object):

    def __init__(self, data, model, echan_list):
        """
        
        :param data: 
        :param model: 
        """

        self._data = data       # type: ContinuousData
        self._model = model     # type: Model
        self._echan_list = echan_list #list of all echans which should be fitted
        self._use_SAA = data.use_SAA

        self._name = "Count rate detector %s" % self._data._det
        # The MET start time of the day
        self._day_met = self._data._day_met


        self._free_parameters = self._model.free_parameters
        self._parameters = self._model.parameters


        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins[2:-2]

        # Get the SAA and GRB mask:
        self._saa_mask = self._data.saa_mask[2:-2]
        self._grb_mask = np.full(len(self._total_time_bins), True)
        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~ np.logical_xor(self._saa_mask, self._grb_mask)

        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins. For all echans together
        self._counts_all_echan = self._data.counts[2:-2][self._total_mask]
        self._total_counts_all_echan = self._data.counts[2:-2]

        self._total_scale_factor = 1.
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False
        self._grb_triggers = {}
        self._occ_region = {}

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
        # Rebinn the observec counts on time
        self._rebinned_observed_counts_fitting_all_echan=[]
        for echan in self._echan_list:
            self._rebinned_observed_counts_fitting_all_echan.append(self._fit_rebinner.rebin(self._total_counts_all_echan[:, echan]))
        self._rebinned_observed_counts_fitting_all_echan=np.array(self._rebinned_observed_counts_fitting_all_echan)

    def _rebinned_model_counts_fitting(self):
        # the rebinned expected counts from the model
        self._rebinned_model_counts_fitting_all_echan = []
        for echan in self._echan_list:
            self._rebinned_model_counts_fitting_all_echan.append(self._fit_rebinner.rebin(self.model_counts(echan))[0])
        self._rebinned_model_counts_fitting_all_echan=np.array(self._rebinned_model_counts_fitting_all_echan)

    def _evaluate_model(self, echan):
        """
        
        loops over time bins and extracts the model counts and returns this array
        
        
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


        synth_data.counts[:, self._echan][2:-2] = np.random.poisson(synth_model.get_counts(synth_data.time_bins[2:-2]))

        self._synth_model = synth_model

        return synth_data


    def __call__(self, parameters):
        """
        
        :return: the poisson log likelihood
        """
        self._set_free_parameters(parameters)
        if self._fit_rebinned==True:
            self._rebinned_observed_counts_fitting()
            self._rebinned_model_counts_fitting()
        log_likelihood_list=[]
        for echan in self._echan_list:
            log_likelihood_list.append(self._get_log_likelihood_echan(echan))
        log_likelihood_list=np.array(log_likelihood_list)
        return np.sum(log_likelihood_list)

    def _get_log_likelihood_echan(self, echan):

        M = self._evaluate_model(echan)
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

        # Use rebinned counts if fir_rebinned is set to true:
        if self._fit_rebinned == True:
            index = int(np.argwhere(self._echan_list==echan))
            d_times_logM = self._rebinned_observed_counts_fitting_all_echan[index] * logM

        else:
            d_times_logM = self._counts_all_echan[:,echan] * logM

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
        self._grb_mask = np.full(len(self._time_bins), True)

    def display_model(self, echan, data_color='k', model_color='r', step=True, show_data=True, show_residuals=True,
                      show_legend=True, min_bin_width=1E-99, plot_sources=False, show_grb_trigger=False,
                      show_model=True, change_time=False, show_occ_region=False, **kwargs):

        """
        Plot the current model with or without the data and the residuals. Multiple models can be plotted by supplying
        a previous axis to 'model_subplot'.
        Example usage:
        fig = data.display_model()
        fig2 = data2.display_model(model_subplot=fig.axes)
        :param show_occ_region:
        :param show_grb_trigger:
        :param plot_sources:
        :param min_bin_width:
        :param change_time:
        :param show_model:
        :param data_color: the color of the data
        :param model_color: the color of the model
        :param step: (bool) create a step count histogram or interpolate the model
        :param show_data: (bool) show_the data with the model
        :param show_residuals: (bool) shoe the residuals
        :param show_legend: (bool) show legend
        :return:
        """

        # Change time reference to seconds since beginning of the day
        if change_time:
            time_ref = self._day_met
            time_frame = 'Seconds since midnight'
        else:
            time_ref = 0.
            time_frame = 'MET'

        model_label = "Background fit"

        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)


        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if (min_bin_width is not NO_REBIN) or (self._rebinner is None):


            this_rebinner = Rebinner(self._total_time_bins - time_ref, min_bin_width, self._saa_mask)

        else:

            # Use the rebinner already in the data
            this_rebinner = self._rebinner


        # Residuals

        # we need to get the rebinned counts
        self._rebinned_observed_counts, = this_rebinner.rebin(self._total_counts_all_echan[:,echan])

        # the rebinned counts expected from the model
        self._rebinned_model_counts, = this_rebinner.rebin(self.model_counts(echan))

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
        y = self.model_counts(echan) / self._total_time_bin_widths

        x = np.mean(self._total_time_bins - time_ref, axis=1)

        if show_model:
            residual_plot.add_model(x,
                                    y,
                                    label=model_label,
                                    color=model_color)

        if plot_sources:

            source_list = self._get_list_of_sources(self._total_time_bins - time_ref, echan, self._total_time_bin_widths)

            residual_plot.add_list_of_sources(x, source_list)

        # Add vertical lines for grb triggers

        if show_grb_trigger:
            residual_plot.add_vertical_line(self._grb_triggers, time_ref)

        if show_occ_region:
            residual_plot.add_occ_region(self._occ_region, time_ref)


        return residual_plot.finalize(xlabel="Time\n(%s)" %time_frame,
                                      ylabel="Count Rate\n(counts s$^{-1}$)",
                                      xscale='linear',
                                      yscale='linear',
                                      show_legend=show_legend)

    def _get_list_of_sources(self,time_bins, echan, time_bin_width=1.):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """
        source_list = []
        color_list = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
        i_index=0
        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, time_bins, self._saa_mask, echan)
            if np.sum(data) != 0:
                source_list.append({"label": source_name, "data": data / time_bin_width, "color": color_list[i_index]})
                i_index+=1
        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(i, time_bins, self._saa_mask, echan)
            source_list.append({"label": source_name, "data": data / time_bin_width, "color": color_list[i_index]})
            i_index += 1
        if self._use_SAA:
            saa_data = self._model.get_saa_counts(self._total_time_bins, self._saa_mask, echan)
            if np.sum(saa_data) != 0:
                source_list.append({"label": "SAA_decays", "data": saa_data / time_bin_width, "color": color_list[i_index]})
                i_index += 1
        point_source_data = self._model.get_point_source_counts(self._total_time_bins, self._saa_mask, echan)
        if np.sum(point_source_data) != 0:
            source_list.append({"label": "Point_sources", "data": point_source_data / time_bin_width, "color": color_list[i_index]})
            i_index += 1
        return source_list


    def add_grb_trigger(self, grb_name, trigger_time, time_format='UTC', time_offset= 0, color='b'):
        """
        Add a GRB Trigger to plot a vertical line
        The grb is added to a dictionary with the name as key and the time (met) and the color as values in a subdict
        A time offset can be used to add line in reference to a trigger
        :param grb_name: string
        :param trigger_time: string in UTC '00:23:11.997'
        :return:
        """
        if time_format == 'UTC':
            day = self._data.day
            year = '20%s'%day[:2]
            month = day[2:-2]
            dd = day[-2:]

            day_at = astro_time.Time("%s-%s-%sT%s(UTC)" % (year, month, dd, trigger_time))

            met = GBMTime(day_at).met + time_offset

        if time_format == 'MET':
            met = trigger_time

        self._grb_triggers[grb_name] = {'met': met, 'color': color}

    def add_occ_region(self, occ_name, time_start, time_stop, time_format='UTC', color='grey'):
        """

        :param occ_name:
        :param start_time:
        :param stop_time:
        :param color:
        :return:
        """
        if time_format == 'UTC':
            day = self._data.day
            year = '20%s' % day[:2]
            month = day[2:-2]
            dd = day[-2:]
            t_start = astro_time.Time("%s-%s-%sT%s(UTC)" % (year, month, dd, time_start))
            t_stop = astro_time.Time("%s-%s-%sT%s(UTC)" % (year, month, dd, time_stop))

            met_start = GBMTime(t_start).met
            met_stop = GBMTime(t_stop).met

        if time_format == 'MET':
            met_start = time_start
            met_stop = time_stop

        self._occ_region[occ_name] = {'met': (met_start, met_stop), 'color': color}

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

        self._set_free_parameters(fit_result)

        print("Fits file was successfully loaded and the free parameters set")

    @property
    def use_SAA(self):

        return self._use_SAA

    # define a function that return the residuals of the fit:
    def residuals(self, echan):
        significance_calc_return = Significance(self.data_counts,
                                                self.model_counts(echan) / self._total_scale_factor,
                                                self._total_scale_factor)
        self._residuals_return = significance_calc_return.known_background()
        return np.vstack((self._data.mean_time[2:-2], self._residuals_return)).T

    def residuals_rebinned(self, echan, min_bin_width=NO_REBIN):
        time_ref = 0.
        if (min_bin_width is not NO_REBIN) or (self._rebinner is None):
            this_rebinner = Rebinner(self._total_time_bins - time_ref, min_bin_width, self._total_mask)  # _saa_mask

        # we need to get the rebinned counts
        rebinned_observed_counts, = this_rebinner.rebin(self._total_counts_all_echan[:,echan])

        # the rebinned counts expected from the model
        rebinned_model_counts, = this_rebinner.rebin(self.model_counts(echan))

        rebinned_background_counts = np.zeros_like(rebinned_observed_counts)

        rebinned_mean_time = np.mean(this_rebinner.time_rebinned, axis=1)

        significance_calc_res_rebinned = Significance(rebinned_observed_counts,
                                                      rebinned_background_counts + rebinned_model_counts / self._total_scale_factor,
                                                      self._total_scale_factor)
        self._residuals_return_rebinned = significance_calc_res_rebinned.known_background()

        return np.vstack((rebinned_mean_time[2:-2], self._residuals_return_rebinned[2:-2])).T

    #test
    @property
    def time_rebinned(self):
        return self._rebinned_time_bins
    @property
    def res(self):
        return self._residuals
import numpy as np

from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
from os import getenv
from os.path import join, basename
import h2o
from pandas import DataFrame

NO_REBIN = 1E-99


class H2OAutoML(object):

    def __init__(self, data, file_dir=join(getenv('GBMDATA'), 'ml/models')):
        self._data = data
        self.file_dir = file_dir

        self._name = "Count rate detector %s" % self._data._det
        # The MET start time of the day
        self._day_met = self._data._day_met
        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = self._data.time_bins[2:-2]
        self._total_mean_times = self._data.mean_time[2:-2]
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        self.features = data.calc_features(self._total_mean_times)

        df = DataFrame(self.features, columns=['det_ra', 'det_dec', 'sc0', 'sc1', 'sc2', 'sc_height', 'earth_az',
                                               'earth_zen', 'sun_angle', 'q0', 'q1', 'q2', 'q3', 'mc_l'])

        print(df.shape)
        print(data.rates.shape)
        df['rates3'] = data.rates[:, 2][2:-2]

        self.hf = h2o.H2OFrame(df)

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
        self._grb_mask_calculated = False
        self._grb_triggers = {}
        self._occ_region = {}

        self.model = None
        self._model_counts = None
        self._model_count_rates = None
        self._rebinned_observed_counts = None
        self._rebinned_model_counts = None
        self._rebinned_model_count_rates = None
        self._rebinned_background_counts = None
        self._rebinned_time_bins = None
        self._rebinned_time_bin_widths = None
        self._residuals = None

    def predict(self, features):
        predict_hf = self.model.predict(self.hf)
        return np.array(predict_hf.as_data_frame()['predict'])

    def load_model(self, file_name):
        fname = join(self.file_dir, file_name)
        print("Loading model '{0:}'".format(basename(fname)))
        self.model = h2o.load_model(fname)
        return

    @property
    def model_count_rates(self):
        if self._model_count_rates is None:
            self._model_count_rates = self.predict(self.features)

            # The SAA sections will be set to zero if a saa_mask is provided
            if self._saa_mask is not None:
                assert len(self._model_count_rates) == len(self._saa_mask), "The time_bins and saa_mask should be of equal length"
                self._model_count_rates[np.where(~self._saa_mask)] = 0.
        return self._model_count_rates

    @property
    def model_counts(self):
        if self._model_counts is None:
            self._model_counts = self.predict(self.features) * self._total_time_bin_widths

            # The SAA sections will be set to zero if a saa_mask is provided
            if self._saa_mask is not None:
                assert len(self._model_counts) == len(self._saa_mask), "The time_bins and saa_mask should be of equal length"
                self._model_counts[np.where(~self._saa_mask)] = 0.

        return self._model_counts

    def display_model(self, echan, data_color='k', model_color='r', show_data=True, show_residuals=True,
                      show_legend=True, min_bin_width=1E-99, show_grb_trigger=False,
                      show_model=True, change_time=True, show_occ_region=False, **kwargs):

        """
        Plot the current model with or without the data and the residuals. Multiple models can be plotted by supplying
        a previous axis to 'model_subplot'.
        Example usage:
        fig = data.display_model()
        fig2 = data2.display_model(model_subplot=fig.axes)
        :param echan:
        :param show_occ_region:
        :param show_grb_trigger:
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

        model_label = "NN Background fit"
        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if (min_bin_width is not NO_REBIN) or (self._rebinner is None):

            this_rebinner = Rebinner(self._total_time_bins - time_ref, min_bin_width, self._saa_mask)

        else:
            # Use the rebinner already in the data
            this_rebinner = self._rebinner

        # Residuals

        # we need to get the rebinned counts
        self._rebinned_observed_counts, = this_rebinner.rebin(self._total_counts_all_echan[:, echan])

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

        residual_plot.add_data(np.mean(self._rebinned_time_bins, axis=1),
                               self._rebinned_observed_counts / self._rebinned_time_bin_widths,
                               self._residuals,
                               residual_yerr=residual_errors,
                               yerr=None,
                               xerr=None,
                               label=self._name,
                               color=data_color,
                               show_data=show_data)

        # We always plot the model un-rebinned here
        # Mask the array so we don't plot the model where data have been excluded
        # y = expected_model_rate / chan_width

        y = self.model_count_rates

        x = self._total_mean_times - time_ref

        if show_model:
            residual_plot.add_model(x,
                                    y,
                                    label=model_label,
                                    color=model_color)

        # Add vertical lines for grb triggers

        if show_grb_trigger:
            residual_plot.add_vertical_line(self._grb_triggers, time_ref)

        if show_occ_region:
            residual_plot.add_occ_region(self._occ_region, time_ref)

        return residual_plot.finalize(xlabel="Time\n(%s)" % time_frame,
                                      ylabel="Count Rate\n(counts s$^{-1}$)",
                                      xscale='linear',
                                      yscale='linear',
                                      show_legend=show_legend)

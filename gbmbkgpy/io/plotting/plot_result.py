import numpy as np
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
from matplotlib import pyplot as plt
import h5py
from datetime import datetime
from gbmgeometry import GBMTime
import astropy.time as astro_time

NO_REBIN = 1E-99


class ResultPlotGenerator(object):
    def __init__(self, data_path, plot_config, color_config):

        # Import general settings
        self.data_path = plot_config['data_path']
        self.bin_width = plot_config['bin_width']
        self.change_time = plot_config['change_time']
        self.xlim = plot_config['xlim']
        self.ylim = plot_config['ylim']
        self.dpi = plot_config['dpi']
        self.mpl_style = plot_config['mpl_style']
        self.legend_outside = plot_config['legend_outside']

        # Flags for activate/deactivate
        self.show_residuals = plot_config['show_residuals']
        self.show_data = plot_config['show_data']
        self.show_sources = plot_config['show_sources']
        self.show_model = plot_config['show_model']
        self.show_ppc = plot_config['show_ppc']
        self.show_legend = plot_config['show_legend']
        self.show_occ_region = plot_config['show_occ_region']
        self.show_grb_trigger = plot_config['show_grb_trigger']

        # Color settings
        self.model_color = color_config['model_color']
        self.source_colors = color_config['source_colors']
        self.ppc_colors = color_config['ppc_colors']
        self.data_color = color_config['data_color']

        if plot_config['mpl_style'] is not None:
            plt.style.use(plot_config['mpl_style'])

        # Save path basis
        self.save_path_basis = '/'.join(data_path.split('/')[:-1])

        self._grb_triggers = {}
        self._occ_region = {}

        self._dates = None
        self._day_start_times = None
        self._day_stop_times = None
        self._saa_mask = None
        self._model_counts = None
        self._observed_counts = None
        self._ppc_counts = None
        self._sources = None
        self._total_time_bins = None

    def create_plots(self):
        # Load data and start plotting
        with h5py.File(self.data_path, 'r') as f:
            keys = f.keys()
            det = np.array(f['general']['Detector'])
            self._dates = np.array(f['general']['Dates'])
            self._day_start_times = np.array(f['general']['day_start_times'])
            self._day_stop_times = np.array(f['general']['day_stop_times'])
            self._saa_mask = np.array(f['general']['saa_mask'])
            for i, day in enumerate(self._dates):

                for key in keys:
                    if key == 'general':
                        pass
                    else:
                        echan = key.split(' ')[1]
                        time_bins_start = np.array(f[key]['time_bins_start'])
                        time_bins_stop = np.array(f[key]['time_bins_stop'])
                        self._model_counts = np.array(f[key]['total_model_counts'])
                        self._observed_counts = np.where(self._saa_mask, np.array(f[key]['observed_counts']), 0)
                        self._ppc_counts = np.array(f[key]['PPC'])

                        self._sources = {}
                        for key_inter in f[key]['Sources']:
                            self._sources[key_inter] = np.array(f[key]['Sources'][key_inter])
                        self._total_time_bins = np.vstack((time_bins_start, time_bins_stop)).T
                        time_stamp = datetime.now().strftime('%y%m%d_%H%M')

                        self._create_model_plots(
                            which_day=i,
                            savepath='{}/plot_date_{}_det_{}_echan_{}__{}.pdf'.format(self.save_path_basis, day, det, echan, time_stamp),
                            dpi=self.dpi,
                        )

    def _create_model_plots(self,
                            which_day=0,
                            savepath='test.pdf',
                            **kwargs
                            ):
        """
        Plot the current model with or without the data and the residuals.
        :param echan:
        :param which_day:
        :param savepath:
        :param bin_width:
        :param change_time:
        :param show_residuals:
        :param show_data:
        :param show_sources:
        :param show_ppc:
        :param show_grb_trigger:
        :param times_mark:
        :param names_mark:
        :param xlim:
        :param ylim:
        :param legend_outside:
        :param dpi:
        :return:
        """

        # Change time reference to seconds since beginning of the day
        if self.change_time and which_day is not None:
            assert which_day < len(self._dates), 'Use a valid date...'
            self._time_ref = self._day_start_times[which_day]
            time_frame = 'Time since midnight [s]'
        else:
            self._time_ref = 0
            time_frame = 'MET [s]'

        residual_plot = ResidualPlot(show_residuals=self.show_residuals, **kwargs)

        this_rebinner = Rebinner((self._total_time_bins - self._time_ref), self.bin_width)

        self._rebinned_observed_counts, = this_rebinner.rebin(self._observed_counts)

        self._rebinned_model_counts, = this_rebinner.rebin(self._model_counts)

        self._rebinned_background_counts = np.zeros_like(self._rebinned_observed_counts)

        self._rebinned_time_bins = this_rebinner.time_rebinned

        self._rebinned_time_bin_widths = np.diff(self._rebinned_time_bins, axis=1)[:, 0]

        self._rebinned_time_bin_mean = np.mean(self._rebinned_time_bins, axis=1)

        significance_calc = Significance(self._rebinned_observed_counts,
                                         self._rebinned_background_counts + self._rebinned_model_counts,
                                         1)
        residual_errors = None
        self._residuals = significance_calc.known_background()

        residual_plot.add_data(np.mean(self._rebinned_time_bins, axis=1),
                               self._rebinned_observed_counts / self._rebinned_time_bin_widths,
                               self._residuals,
                               residual_yerr=residual_errors,
                               yerr=None,
                               xerr=None,
                               label='Observed Count Rates',
                               color=self.data_color,
                               show_data=self.show_data, marker_size=1.5)

        if self.show_model:
            residual_plot.add_model(self._rebinned_time_bin_mean - self._time_ref,
                                    self._rebinned_model_counts / self._rebinned_time_bin_widths,
                                    label='Best Fit',
                                    color=self.model_color)

        if self.show_sources:
            source_list = []

            for i, key in enumerate(self._sources.keys()):
                if 'L-parameter' in key:
                    label = 'Cosmic Rays'
                elif 'CGB' in key:
                    label = 'Cosmic Gamma-Ray Background'
                elif 'Earth' in key:
                    label = 'Earth Albedo'
                elif 'Constant' in key:
                    label = 'Constant'
                elif 'CRAB' in key:
                    label = 'Crab'
                elif 'sun' in key:
                    label = 'Sun'
                elif 'SAA_decays' in key:
                    label = 'SAA Exits'
                else:
                    label = key
                rebinned_source_counts = this_rebinner.rebin(self._sources[key])[0]

                source_list.append({
                    'data': rebinned_source_counts,
                    'label': label,
                    'color': self.source_colors[i]
                })

            residual_plot.add_list_of_sources(self._rebinned_time_bin_mean - self._time_ref, source_list)

        if self.show_ppc:
            rebinned_ppc_rates = []
            for j, ppc in enumerate(self._ppc_counts):
                rebinned_ppc_rates.append(this_rebinner.rebin(self._ppc_counts[j][2:-2]) / self._rebinned_time_bin_widths)
            rebinned_ppc_rates = np.array(rebinned_ppc_rates)

            residual_plot.add_ppc(rebinned_ppc_rates=rebinned_ppc_rates,
                                  rebinned_time_bins=self._rebinned_time_bins - self._time_ref,
                                  q_levels=[0.68, 0.95, 0.99],
                                  colors=self.ppc_colors,
                                  )

        # Add vertical lines for grb triggers
        if self.show_grb_trigger:
            residual_plot.add_vertical_line(self._grb_triggers, self._time_ref)

        if self.show_occ_region:
            residual_plot.add_occ_region(self._occ_region, self._time_ref)

        if self.xlim is None or self.ylim is None:
            xlim, ylim = self._calc_limits(which_day)
            self.xlim = xlim if self.xlim is None else self.xlim
            self.ylim = ylim if self.ylim is None else self.ylim

        final_plot = residual_plot.finalize(xlabel="Time\n(%s)" % time_frame,
                                            ylabel="Count Rate\n(counts s$^{-1}$)",
                                            xscale='linear',
                                            yscale='linear',
                                            show_legend=self.show_legend,
                                            xlim=self.xlim,
                                            ylim=self.ylim,
                                            legend_outside=self.legend_outside)

        final_plot.savefig(savepath, dpi=self.dpi)

    def add_grb_trigger(self, grb_name, trigger_time, time_format='UTC', time_offset=0, color='b'):
        """
        Add a GRB Trigger to plot a vertical line
        The grb is added to a dictionary with the name as key and the time (met) and the color as values in a subdict
        A time offset can be used to add line in reference to a trigger
        :param color:
        :param time_offset:
        :param time_format:
        :param grb_name: string
        :param trigger_time: string in UTC '2008-01-01T00:23:11.997'
        :return:
        """
        if time_format == 'UTC':
            day_at = astro_time.Time("%s(UTC)" % trigger_time)

            met = GBMTime(day_at).met + time_offset

        elif time_format == 'MET':
            met = trigger_time

        else:
            raise Exception('Not supported time format, please use MET or UTC')

        self._grb_triggers[grb_name] = {'met': met, 'color': color}

    def add_occ_region(self, occ_name, time_start, time_stop, time_format='UTC', color='grey'):
        """

        :param time_format:
        :param occ_name:
        :param time_start: string in MET or UTC '2008-01-01T00:23:11.997'
        :param time_stop: string in MET or UTC '2008-01-01T00:23:11.997'
        :param color:
        :return:
        """
        if time_format == 'UTC':
            t_start = astro_time.Time("%s(UTC)" % time_start)
            t_stop = astro_time.Time("%s(UTC)" % time_stop)

            met_start = GBMTime(t_start).met
            met_stop = GBMTime(t_stop).met

        elif time_format == 'MET':
            met_start = time_start
            met_stop = time_stop
        else:
            raise Exception('Not supported time format, please use MET or UTC')

        self._occ_region[occ_name] = {'met': (met_start, met_stop), 'color': color}

    def _calc_limits(self, which_day):
        min_time = self._day_start_times[which_day] - self._time_ref
        max_time = self._day_stop_times[which_day] - self._time_ref

        day_mask_larger = self._rebinned_time_bin_mean > min_time
        day_mask_smaller = self._rebinned_time_bin_mean < max_time

        day_mask_total = day_mask_larger * day_mask_smaller

        time_bins_masked = self._rebinned_time_bins[day_mask_total]
        obs_counts_masked = self._rebinned_observed_counts[day_mask_total]

        zero_counts_mask = obs_counts_masked > 1

        index_start = [0]
        index_stop = []

        for i in range(len(zero_counts_mask) - 1):
            if zero_counts_mask[i] is False and zero_counts_mask[i + 1] is True:
                index_stop.append(i - 1)
            if zero_counts_mask[i] is True and zero_counts_mask[i + 1] is False:
                index_start.append(i)
        if len(index_start) > len(index_stop):
            index_stop.append(-1)
        for i in range(len(index_stop) - 1):
            if time_bins_masked[:, 1][index_start[i + 1]] - time_bins_masked[:, 0][index_stop[i]] < 1000:
                zero_counts_mask[index_stop[i] - 5:index_start[i + 1] + 5] = np.ones_like(zero_counts_mask[index_stop[i] - 5:index_start[i + 1] + 5]) == 2

        time_bins_masked2 = time_bins_masked[zero_counts_mask]

        time_bins_intervals = []
        start = time_bins_masked2[0, 0]
        for i in range(len(time_bins_masked2) - 1):
            if time_bins_masked2[i + 1, 0] - time_bins_masked2[i, 0] > 5 * 60 * 60:
                stop = time_bins_masked2[i, 0] + 100
                time_bins_intervals.append((start, stop))
                start = time_bins_masked2[i + 1, 0] - 100
        time_bins_intervals.append((start, time_bins_masked2[-1, 0]))
        xlim = time_bins_intervals[0]

        obs_rates_masked2 = obs_counts_masked[zero_counts_mask] / np.diff(time_bins_masked2, axis=1)[0]
        high_lim = 1.5 * np.percentile(obs_rates_masked2, 99)
        ylim = (0, high_lim)

        return xlim, ylim

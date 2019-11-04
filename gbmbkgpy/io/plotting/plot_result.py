import numpy as np
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
from matplotlib import pyplot as plt
import h5py
from datetime import datetime
from gbmgeometry import GBMTime
import astropy.time as astro_time
from gbmbkgpy.utils.progress_bar import progress_bar

NO_REBIN = 1E-99


class ResultPlotGenerator(object):
    def __init__(self, plot_config, component_config, style_config, highlight_config={}):

        # Import plot settings
        self.data_path =        plot_config['data_path']
        self.bin_width =        plot_config.get('bin_width', 10)
        self.change_time =      plot_config.get('change_time', True)
        self.xlim =             plot_config.get('xlim', None)
        self.ylim =             plot_config.get('ylim', None)
        self.xscale =           plot_config.get('xscale', 'linear')
        self.yscale =           plot_config.get('yscale', 'linear')
        self.xlabel =           plot_config.get('xlabel', None)
        self.ylabel =           plot_config.get('ylabel', None)
        self.dpi =              plot_config.get('dpi', 400)
        self.show_legend =      plot_config.get('show_legend', True)
        self.legend_outside =   plot_config.get('legend_outside', True)

        # Import component settings
        self.show_data =        component_config.get('show_data', True)
        self.show_model =       component_config.get('show_model', True)
        self.show_ppc =         component_config.get('show_ppc', True)
        self.show_residuals =   component_config.get('show_residuals', False)

        self.show_all_sources = component_config.get('show_all_sources', True)
        self.show_earth =       component_config.get('show_earth', True)
        self.show_cgb =         component_config.get('show_cgb', True)
        self.show_sun =         component_config.get('show_sun', True)
        self.show_saa =         component_config.get('show_saa', True)
        self.show_cr =          component_config.get('show_cr', True)
        self.show_constant =    component_config.get('show_constant', True)
        self.show_crab =        component_config.get('show_crab', True)

        self.show_occ_region =  component_config.get('show_occ_region', False)
        self.show_grb_trigger = component_config.get('show_grb_trigger', False)

        # Import style settings
        self.model_color =      style_config['model']
        self.source_colors =    style_config['sources']
        self.ppc_colors =       style_config['ppc']
        self.data_color =       style_config['data']
        self.legend_kwargs =    style_config.get('legend_kwargs', None)

        if style_config['mpl_style'] is not None:
            plt.style.use(style_config['mpl_style'])

        # Save path basis
        self.save_path_basis = '/'.join(self.data_path.split('/')[:-1])

        self._grb_triggers = {}
        self._occ_region = {}

        if highlight_config != {}:
            if highlight_config['grb_trigger'] is not None:
                for grb_trigger in highlight_config['grb_trigger']:
                    self.add_grb_trigger(grb_name=grb_trigger['name'],
                                         trigger_time=grb_trigger['trigger_time'],
                                         time_format='UTC',
                                         time_offset=grb_trigger['time_offset'],
                                         color=grb_trigger.get('color', 'b'))

            if highlight_config['occ_region'] is not None:
                for occ_region in highlight_config['occ_region']:
                    self.add_occ_region(occ_name=occ_region['name'],
                                        time_start=occ_region['time_start'],
                                        time_stop=occ_region['time_stop'],
                                        time_format=occ_region.get('time_format', 'UTC'),
                                        color=occ_region.get('color', 'grey'))

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
        print('Load data and start plotting')

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

                        total_steps = 12 if self.show_ppc is False else 12 + len(self._ppc_counts)

                        with progress_bar(total_steps, title='Create Result plot') as p:
                            self._create_model_plots(
                                which_day=i,
                                savepath='{}/plot_date_{}_det_{}_echan_{}__{}.pdf'.format(self.save_path_basis, day, det, echan, time_stamp),
                                p_bar=p
                            )
        print('Success!')

    def _create_model_plots(self,
                            p_bar,
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
            time_frame = 'Time since midnight [h]'
        else:
            self._time_ref = 0
            time_frame = 'MET [s]'

        p_bar.increase()

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

        p_bar.increase()

        residual_plot.add_data(np.mean(self._rebinned_time_bins, axis=1),
                               self._rebinned_observed_counts / self._rebinned_time_bin_widths,
                               self._residuals,
                               residual_yerr=residual_errors,
                               yerr=None,
                               xerr=None,
                               label='Observed Count Rates',
                               color=self.data_color['color'],
                               alpha=self.data_color['alpha'],
                               show_data=self.show_data, marker_size=1.5)

        p_bar.increase()

        if self.show_model:
            residual_plot.add_model(self._rebinned_time_bin_mean,
                                    self._rebinned_model_counts / self._rebinned_time_bin_widths,
                                    label='Best Fit',
                                    color=self.model_color['color'],
                                    alpha=self.model_color['alpha']
                                    )

        p_bar.increase()

        source_list = []
        for i, key in enumerate(self._sources.keys()):
            if 'L-parameter' in key:
                label = 'Cosmic Rays'
                color_key = 'cr'
                if not self.show_all_sources and not self.show_cr:
                    continue

            elif 'CGB' in key:
                label = 'Cosmic Gamma-Ray Background'
                color_key = 'cgb'
                if not self.show_all_sources and not self.show_cgb:
                    continue

            elif 'Earth' in key:
                label = 'Earth Albedo'
                color_key = 'earth'
                if not self.show_all_sources and not self.show_earth:
                    continue

            elif 'Constant' in key:
                label = 'Constant'
                color_key = 'constant'
                if not self.show_all_sources and not self.show_constant:
                    continue

            elif 'CRAB' in key:
                label = 'Crab'
                color_key = 'crab'
                if not self.show_all_sources and not self.show_crab:
                    continue

            elif 'sun' in key:
                label = 'Sun'
                color_key = 'sun'
                if not self.show_all_sources and not self.show_sun:
                    continue

            elif 'SAA_decays' in key:
                label = 'SAA Exits'
                color_key = 'saa'
                if not self.show_all_sources and not self.show_saa:
                    continue

            else:
                label = key
                color_key = 'default'
                if not self.show_all_sources:
                    continue

            rebinned_source_counts = this_rebinner.rebin(self._sources[key])[0]

            source_list.append({
                'data': rebinned_source_counts / self._rebinned_time_bin_widths,
                'label': label,
                'color': self.source_colors[color_key]['color'] if not self.source_colors['use_global'] else None,
                'alpha': self.source_colors[color_key]['alpha'] if not self.source_colors['use_global'] else None,
            })

        if self.source_colors['use_global']:
            cmap = plt.get_cmap(self.source_colors['global']['cmap'])
            colors = cmap(np.linspace(0, 1, len(source_list)))

            for i, source in enumerate(source_list):
                source['color'] = colors[i]
                source['alpha'] = self.source_colors['global']['alpha']

        if len(source_list) > 0:
            residual_plot.add_list_of_sources(self._rebinned_time_bin_mean, source_list)

        p_bar.increase()

        if self.show_ppc:
            rebinned_ppc_rates = []
            for j, ppc in enumerate(self._ppc_counts):
                self._ppc_counts[j][np.where(~self._saa_mask)] = 0.
                rebinned_ppc_rates.append(this_rebinner.rebin(self._ppc_counts[j][2:-2]) / self._rebinned_time_bin_widths)
                p_bar.increase()
            rebinned_ppc_rates = np.array(rebinned_ppc_rates)

            residual_plot.add_ppc(rebinned_ppc_rates=rebinned_ppc_rates,
                                  rebinned_time_bin_mean=self._rebinned_time_bin_mean,
                                  q_levels=[0.68, 0.95, 0.99],
                                  colors=self.ppc_colors['color'],
                                  alpha=self.ppc_colors['alpha']
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

        xticks = []
        xtick_labels = []
        for xstep in range(int(self.xlim[0] / 3600), int(self.xlim[1] / 3600), 4):
            xticks.append(xstep * 3600)
            xtick_labels.append('%sh' % xstep)

        p_bar.increase()

        xlabel = "{}".format(time_frame) if self.xlabel is None else self.xlabel
        ylabel = "Count Rate [counts s$^{-1}$]" if self.ylabel is None else self.ylabel

        final_plot = residual_plot.finalize(xlabel=xlabel,
                                            ylabel=ylabel,
                                            xscale=self.xscale,
                                            yscale=self.yscale,
                                            xticks=xticks,
                                            xtick_labels=xtick_labels,
                                            show_legend=self.show_legend,
                                            xlim=self.xlim,
                                            ylim=self.ylim,
                                            legend_outside=self.legend_outside,
                                            legend_kwargs=self.legend_kwargs)

        p_bar.increase()

        final_plot.savefig(savepath, dpi=self.dpi)

        p_bar.increase()

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

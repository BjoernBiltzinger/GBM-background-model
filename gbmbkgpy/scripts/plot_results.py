import matplotlib
matplotlib.use('AGG')
import h5py
import numpy as np
import seaborn as sns
from gbmbkgpy.utils.binner import Rebinner
from gbmbkgpy.utils.statistics.stats_tools import Significance
from datetime import datetime
NO_REBIN = 1E-99

import matplotlib.pyplot as plt

# Path to data_for_plots.hdf5 file
path = '/data/bbiltzing/data/fits/mn_out/test_2days_nb_0/fit_10-14_10-03/data_for_plots.hdf5'

# Svae path basis
save_path_basis = '/'.join(path.split('/')[:-1]) 

def display_model(echan, which_day=0, add_sources=True, add_ppc=True, add_residuals=False, savepath='test.pdf', min_bin_width=120, legend_outside=False):

    if which_day!=None:
        assert which_day<len(dates), 'USe a valid date...'
        time_ref = day_start_times[which_day]
        time_frame = 'Time since midnight [s]'
    else:
        time_ref = 0
        time_frame = 'MET [s]'

    model_label = "Background fit"
    
    this_rebinner = Rebinner((total_time_bins - time_ref), min_bin_width)

    rebinned_observed_counts, = this_rebinner.rebin(observed_counts)

    rebinned_model_counts, = this_rebinner.rebin(model_counts)

    rebinned_time_bins = this_rebinner.time_rebinned

    rebinned_bin_length = np.diff(rebinned_time_bins, axis=1)[:, 0]

    rebinned_time_bin_mean = np.mean(rebinned_time_bins, axis=1)
    
    significance_calc = Significance(rebinned_observed_counts,
                                     rebinned_model_counts,
                                     1)

    residuals = significance_calc.known_background()

    residual_errors = None

    fig = plt.figure()
    with sns.axes_style("darkgrid"):
        if add_residuals:
            ax_main = fig.add_subplot(211)
            ax_residuals = fig.add_subplot(212, sharex=ax_main)
        else:
            ax_main = fig.add_subplot(111)

    if add_residuals:
        residual_yerr = np.ones_like(residuals)
        ax_residuals.axhline(0, linestyle='--', color='k')
        ax_residuals.errorbar(rebinned_time_bin_mean,
                              residuals,
                              yerr=residual_yerr,
                              capsize=0,
                              fmt='.',
                              elinewidth=1,
                              markersize=3,
                              color='k')

    ax_main.scatter(rebinned_time_bin_mean,
                    rebinned_observed_counts/rebinned_bin_length,
                    s=3,
                    alpha=.9,
                    label='Observed Count Rates',
                    color='k',
                    zorder=17)

    ax_main.plot(rebinned_time_bin_mean,
                 rebinned_model_counts/rebinned_bin_length,
                 label='Best Fit',
                 color = 'red',
                 alpha = 0.8,
                 zorder=20)
    
    if add_sources:
        
        source_colors = ['navy', 'magenta', 'cyan', 'salmon', 'saddlebrown', 'gold']

        for i, key in enumerate(sources.keys()):
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
            else:
                label = key
            rebinned_source_counts = this_rebinner.rebin(sources[key])[0]
            ax_main.plot(rebinned_time_bin_mean,
                         rebinned_source_counts/rebinned_bin_length,
                         label=label,
                         color=source_colors[i],
                         zorder=18,
                         alpha=0.8)
                     
    if add_ppc:
        q_levels = [0.68, 0.95, 0.99]
        ppc_colors = ['darkgreen', 'green', 'lightgreen'] 
        rebinned_ppc_rates = []
        for j, ppc in enumerate(ppc_counts):
            rebinned_ppc_rates.append(this_rebinner.rebin(ppc_counts[j][2:-2])/rebinned_bin_length)
        rebinned_ppc_rates = np.array(rebinned_ppc_rates)
        for i, level in enumerate(q_levels):
            low = np.percentile(rebinned_ppc_rates, 50-50*level, axis=0)[0]
            high = np.percentile(rebinned_ppc_rates, 50+50*level, axis=0)[0]
            ax_main.fill_between(rebinned_time_bin_mean, low, high, color=ppc_colors[i], zorder=5-i)

    min_time = day_start_times[which_day]-time_ref
    max_time = day_stop_times[which_day]-time_ref

    day_mask_larger = rebinned_time_bin_mean>min_time
    day_mask_smaller = rebinned_time_bin_mean<max_time
    
    
    day_mask_total = day_mask_larger * day_mask_smaller

    time_bins_masked = rebinned_time_bins[day_mask_total]
    obs_counts_masked = rebinned_observed_counts[day_mask_total]

    zero_counts_mask = obs_counts_masked>1

    index_start = [0]
    index_stop = []

    for i in range(len(zero_counts_mask)-1):
        if zero_counts_mask[i]==False and zero_counts_mask[i+1]==True:
            index_stop.append(i-1)
        if zero_counts_mask[i]==True and zero_counts_mask[i+1]==False:
            index_start.append(i)
    if len(index_start)>len(index_stop):
        index_stop.append(-1)
    for i in range(len(index_stop)-1):
        if time_bins_masked[:,1][index_start[i+1]]-time_bins_masked[:,0][index_stop[i]]<1000:
            zero_counts_mask[index_stop[i]-5:index_start[i+1]+5]=np.ones_like(zero_counts_mask[index_stop[i]-5:index_start[i+1]+5])==2

    time_bins_masked2 = time_bins_masked[zero_counts_mask]

    time_bins_intervals = []
    start = time_bins_masked2[0,0]
    for i in range(len(time_bins_masked2)-1):
        if time_bins_masked2[i+1,0]-time_bins_masked2[i,0] > 5*60*60:
            stop = time_bins_masked2[i,0] + 100
            time_bins_intervals.append((start, stop))
            start = time_bins_masked2[i+1,0]-100
    time_bins_intervals.append((start, time_bins_masked2[-1,0]))
    ax_main.set_xlim(time_bins_intervals[0])
    if add_residuals:
        ax_residuals.set_xlim(time_bins_intervals[0])

    obs_rates_masked2 = obs_counts_masked[zero_counts_mask]/np.diff(time_bins_masked2, axis=1)[0]
    high_lim = 1.5*np.percentile(obs_rates_masked2, 99)

    ax_main.set_ylim((0,high_lim))
    if add_residuals:
        ax_residuals.set_xlabel(time_frame)
        ax_residuals.set_ylabel('Residuals')
        ax_main.set_xticks([])

        res_high_lim = 1.5*np.percentile(residuals[residuals<100], 99)
        res_low_lim = 1.5*np.percentile(residuals[residuals<100], 1)

        ax_residuals.set_ylim((res_low_lim, res_high_lim))
    else:
        ax_main.set_xlabel(time_frame)
    ax_main.set_ylabel(r'Count Rates [1/s]')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    if legend_outside:
        box_main = ax_main.get_position()
        ax_main.set_position([box_main.x0, box_main.y0, box_main.width * 0.8, box_main.height])
        ax_main.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))
        if add_residuals:
             box_res = ax_residuals.get_position()
             ax_residuals.set_position([box_res.x0, box_res.y0, box_res.width * 0.8, box_res.height])
    else:
        ax_main.legend(fontsize='x-small', loc=0)
    
    fig.savefig(savepath, dpi=400)
    
with h5py.File(path, 'r') as f:
    keys = f.keys()
    det = np.array(f['general']['Detector'])
    dates =np.array(f['general']['Dates'])
    day_start_times = np.array(f['general']['day_start_times'])
    day_stop_times = np.array(f['general']['day_stop_times'])
    saa_mask = np.array(f['general']['saa_mask'])
    time_stamp = datetime.now().strftime('%y%m%d_%H%M')
    for i, day in enumerate(dates):
        
        for key in keys:
            if key=='general':
                pass
            else:

                echan = key.split(' ')[1]
                time_bins_start = np.array(f[key]['time_bins_start'])
                time_bins_stop = np.array(f[key]['time_bins_stop'])
                model_counts = np.array(f[key]['total_model_counts'])
                observed_counts = np.where(saa_mask, np.array(f[key]['observed_counts']), 0)

                ppc_counts = np.array(f[key]['PPC'])

                sources = {}
                for key_inter in f[key]['Sources']:
                    sources[key_inter] = np.array(f[key]['Sources'][key_inter])
                total_time_bins = np.vstack((time_bins_start, time_bins_stop)).T
                time_stamp = datetime.now().strftime('%y%m%d_%H%M')

                display_model(echan, which_day = i, savepath='{}/plot_{}_date_{}_echan_{}.pdf'.format(save_path_basis,
                                                                                                       time_stamp, day, echan),
                              min_bin_width=5, add_residuals=True) 

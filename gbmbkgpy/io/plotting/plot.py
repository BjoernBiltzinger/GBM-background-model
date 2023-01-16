import numpy as np
import copy
import matplotlib.pyplot as plt

from gbmgeometry import GBMTime
from astropy.time import Time

from gbmbkgpy.utils.statistics.stats_tools import Significance

def plot_lightcurve(model, ax=None, rates=True, eff_echan=None, bin_width=None,
                    show_data=True, data_color="black", data_alpha=0.9,
                    model_alpha=1, show_total_model=True, data_linewidth=1,
                    marker_size=3,
                    total_model_color="green", model_component_list=[], total_model_label=None,
                    model_component_colors=[], filename=None, norm_time=True,
                    t0=None, time_ticks=None, y_ticks=None, time_format="h",
                    xlim=(None, None), ylim=(0, None), plot_ppc=False, ppc_color="darkgreen",
                    plot_saa=True, ppc_percentile=99, ppc_alpha=0.6, saa_color="navy", time_marks={}):

    # TODO: SAA only one legend label
    # Other time bins

    if ax is None:
        fig, ax = plt.subplots()

    # rebin the time bins
    if bin_width is not None:
        # save the old binning to restore it later
        save_bin_width = model.data.min_bin_width

        model.data.rebin_data(bin_width)

    if rates:
        width = model._data.time_bin_width
        ax.set_ylabel("Count rates [cnts/s]")
    else:
        width = 1
        ax.set_ylabel("Counts [cnts]")
    # there are sometimes gaps in the time bins, we do not want to draw the
    # model in the gaps
    time_bins = model.data.time_bins
    times = model.data.mean_time

    # start time of this day
    date = model.data.date
    gbm_time = GBMTime(Time(f"20{date[:2]}-{date[2:4]}-{date[4:6]}T00:00:00", format="isot", scale='utc'))
    start_time_day = gbm_time.met
    if norm_time:
        if t0 is None:
            times -= start_time_day #times[0]
        else:
            times -= start_time_day + t0

    if time_format == 'h':
        times /= (3600)

    num_labels = 0

    if show_data:
        ax.scatter(times,
                   model.data.counts[:, eff_echan]/width,
                   s=marker_size,
                   linewidths=data_linewidth,
                   facecolors="none",
                   edgecolors="black",
                   alpha=data_alpha,
                   label="Data",
                   zorder=5,
                   rasterized=False,
                   )
        num_labels += 1

    # there are sometimes gaps in the time bins, we do not want to draw the
    # model in the gaps
    idxs = np.array([0])
    idxs = np.append(idxs,
                     np.argwhere(~np.isclose(model.data.time_bins[:-1, 1],
                                             model.data.time_bins[1:, 0],
                                             atol=10, rtol=0))+1)
    idxs = np.append(idxs, len(model.data.time_bins))


    #time_bins_split = []
    #for idx in idxs:
    #    time_bins_split.append()
    

    if show_total_model:

        for n, (start, stop) in enumerate(zip(idxs[:-1], idxs[1:])):

            if n == 0:
                if total_model_label is None:
                    label = "Total Model"
                else:
                    label = total_model_label
            else:
                label = None
            ax.plot(times[start:stop],
                    model.get_model_counts(time_bins=time_bins[start:stop])[:, eff_echan]/width[start:stop],
                    color=total_model_color, label=label,
                    alpha=model_alpha, zorder=10)
        num_labels += 1

    if plot_ppc:
        # save current parameter values to reset it after the ppc generation
        current_par_vals = []
        for p in model.parameter.values():
            current_par_vals.append(p.value)

        # samples from fit run
        samples = model.raw_samples
        if len(samples) > 300:
            mask = np.zeros(len(samples), dtype=bool)
            mask[:300] = True
            np.random.shuffle(mask)
            samples = samples[mask]

        # get the model counts for every sample
        model_counts = np.zeros((len(samples), len(times)))
        for i, s in enumerate(samples):
            model.set_parameters(s)
            model_counts[i] = model.get_model_counts(time_bins=time_bins)[:, eff_echan]

        # poisson noise
        model_counts = np.random.poisson(model_counts)

        min_p = np.percentile(model_counts, 50-ppc_percentile/2, axis=0)
        max_p = np.percentile(model_counts, 50+ppc_percentile/2, axis=0)

        for n, (start, stop) in enumerate(zip(idxs[:-1], idxs[1:])):
            if n == 0:
                label = "PPC"
            else:
                label = None
            ax.fill_between(times[start:stop],
                            min_p[start:stop]/width[start:stop],
                            max_p[start:stop]/width[start:stop],
                            color=ppc_color,
                            alpha=ppc_alpha, label=label, linewidth=0, zorder=-5)

        model.set_parameters(current_par_vals)

    for comp, color in zip(model_component_list, model_component_colors):

        for n, (start, stop) in enumerate(zip(idxs[:-1], idxs[1:])):
            if n == 0:
                label = comp
            else:
                label = None
            ax.plot(times[start:stop],
                    model.get_model_counts_given_source([comp], time_bins=time_bins[start:stop])[:, eff_echan]/width[start:stop],
                    color=color, label=label, alpha=model_alpha, zorder=9)
        num_labels += 1

    if plot_saa:
        saa_sources = []
        for source in model.sources:
            if "SAA" in source.name:
                saa_sources.append(source.name)

        if len(saa_sources) > 0:
            for n, (start, stop) in enumerate(zip(idxs[:-1], idxs[1:])):
                if n == 0:
                    label = "SAA"
                else:
                    label = None
                ax.plot(times[start:stop],
                        model.get_model_counts_given_source(saa_sources,
                                                            time_bins=time_bins[start:stop])[:, eff_echan]/width[start:stop],
                        color=saa_color, label=label, alpha=model_alpha,
                        zorder=9)
            num_labels += 1

    for name, mark in time_marks.items():
        time = mark["time"]
        if t0 is not None:
            time -= t0
        if time_format == 'h':
            time /= 3600
        ax.axvline(time, color=mark["color"], alpha=mark["alpha"], label=name)
        num_labels += 1

    fig = ax.get_figure()
    if num_labels < 4:
        ax.legend(
            loc="upper right",
            fancybox=True
        )
    else:
        ncol = 3
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(-0.05, -1.05, 1, 1),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            #mode="expand",
            fancybox=True
        )


    finalize_plot(ax, time_ticks, y_ticks, time_format, xlim, ylim)

    if filename is not None:
        fig.savefig(filename)

    if bin_width is not None:
        # reset time bin
        model.data.rebin_data(save_bin_width)

    return ax


def plot_residuals(model,
                   ax=None,
                   eff_echan=None,
                   alpha=1,
                   color="black",
                   filename=None,
                   norm_time=True,
                   t0=None,
                   time_ticks=None,
                   y_ticks=None,
                   time_format="h",
                   xlim=(None, None),
                   ylim=(None, None),
                   linewidth=1,
                   marker_size=3,
                   time_marks={},
                   bin_width=None):

    if ax is None:
        fig, ax = plt.subplots()

    # rebin the time bins
    if bin_width is not None:
        # save the old binning to restore it later
        save_bin_width = model.data.min_bin_width

        model.data.rebin_data(bin_width)

    time_bins = model.data.time_bins
    times = model.data.mean_time

    # calc residuals
    sign = Significance(model.data.counts[:, eff_echan],
                        model.get_model_counts(time_bins=time_bins)[:, eff_echan],
                        1
                        )

    residuals = sign.known_background()

    # standard 1 sigma error bar
    residual_yerr = np.ones_like(residuals)

    # 0 line
    ax.axhline(0, linestyle="--", color='k')

    date = model.data.date
    gbm_time = GBMTime(Time(f"20{date[:2]}-{date[2:4]}-{date[4:6]}T00:00:00", format="isot", scale='utc'))
    start_time_day = gbm_time.met

    times = model._data.mean_time
    if norm_time:
        if t0 is None:
            times -= start_time_day #times[0]
        else:
            times -= start_time_day + t0

    if time_format == 'h':
        times /= (3600)

    ax.errorbar(times,
                residuals,
                yerr=residual_yerr,
                capsize=0,
                fmt=".",
                elinewidth=linewidth,
                markersize=marker_size,
                color=color,
                alpha=alpha,
                rasterized=False,
                )

    finalize_plot(ax, time_ticks, y_ticks, time_format, xlim, ylim)

    ax.set_ylabel("Residuals [$\sigma $]")
    if filename is not None:
        fig.savefig(filename)

    for name, mark in time_marks.items():
        time = mark["time"]
        if t0 is not None:
            time -= t0
        if time_format == 'h':
            time /= 3600
        ax.axvline(time, color=mark["color"], alpha=mark["alpha"])

    if bin_width is not None:
        # reset time bin
        model.data.rebin_data(save_bin_width)

    return ax


def finalize_plot(ax, time_ticks, y_ticks, time_format, xlim, ylim):
    if time_ticks is not None:
        ax.set_xticks(time_ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if time_format == 'h':
        ax.set_xlabel("Time [h]")
    elif time_format == 's':
        ax.set_xlabel("Time [s]")
    else:
        NotImplementedError("Only s and h as time format supported!")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

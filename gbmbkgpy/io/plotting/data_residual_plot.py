import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from gbmbkgpy.utils.binner import Rebinner
from gbmbkgpy.utils.progress_bar import progress_bar

from gbmbkgpy.io.plotting.step_plots import step_plot
from gbmbkgpy.exceptions.custom_exceptions import custom_warnings

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:
        rank = 0
        using_mpi = False
except:
    rank = 0
    using_mpi = False


class ResidualPlot(object):
    def __init__(self, **kwargs):
        """
        A class that makes data/residual plots
        :param show_residuals: to show the residuals
        :param ratio_residuals: to use ratios instead of sigma
        :param model_subplot: and axis or list of axes to plot to rather than create a new one
        """

        self._ratio_residuals = False
        self._show_residuals = True
        self._ppc = False

        if "show_residuals" in kwargs:
            self._show_residuals = bool(kwargs.pop("show_residuals"))

        if "ratio_residuals" in kwargs:
            self._ratio_residuals = bool(kwargs.pop("ratio_residuals"))

        # this lets you overplot other fits

        if "model_subplot" in kwargs:

            model_subplot = kwargs.pop("model_subplot")

            # turn on or off residuals

            if self._show_residuals:

                assert (
                    type(model_subplot) == list
                ), "you must supply a list of axes to plot to residual"

                assert (
                    len(model_subplot) == 2
                ), "you have requested to overplot a model with residuals, but only provided one axis to plot"

                self._data_axis, self._residual_axis = model_subplot

            else:

                try:

                    self._data_axis = model_subplot

                    self._fig = self._data_axis.get_figure()

                except (AttributeError):

                    # the user supplied a list of axes

                    self._data_axis = model_subplot[0]

            # we will use the figure associated with
            # the data axis

            self._fig = self._data_axis.get_figure()

        else:

            # turn on or off residuals

            if self._show_residuals:
                self._fig, (self._data_axis, self._residual_axis) = plt.subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}, **kwargs
                )

            else:

                self._fig, self._data_axis = plt.subplots(**kwargs)

    @property
    def figure(self):
        """
        :return: the figure instance
        """

        return self._fig

    @property
    def data_axis(self):
        """
        :return: the top or data axis
        """

        return self._data_axis

    @property
    def residual_axis(self):
        """
        :return: the bottom or residual axis
        """

        assert self._show_residuals, "this plot has no residual axis"

        return self._residual_axis

    @property
    def show_residuals(self):
        return self._show_residuals

    @property
    def ratio_residuals(self):
        return self._ratio_residuals

    def add_model_step(self, xmin, xmax, xwidth, y, label, color):
        """
        Add a model but use discontinuous steps for the plotting.
        :param xmin: the low end boundaries
        :param xmax: the high end boundaries
        :param xwidth: the width of the bins
        :param y: the height of the bins
        :param label: the label of the model
        :param color: the color of the model
        :return: None
        """
        step_plot(
            np.asarray(zip(xmin, xmax)),
            y / xwidth,
            self._data_axis,
            alpha=0.8,
            label=label,
            color=color,
        )

    def add_vertical_line(self, grb_triggers, time_ref):
        """

        :param grb_triggers:
        :param time_ref:
        :return:
        """

        for key, value in grb_triggers.items():
            self._data_axis.axvline(
                x=value["met"] - time_ref,
                color=value.get("color", "b"),
                linestyle=value.get("linestyle", "-"),
                linewidth=value.get("linewidth", 1),
                alpha=value.get("alpha", 0.3),
                label=key,
            )

    def add_occ_region(self, occ_region, time_ref):
        """

        :param occ_region:
        :param time_ref:
        :return:
        """

        for key, value in occ_region.items():
            self._data_axis.axvspan(
                xmin=value["met"][0] - time_ref,
                xmax=value["met"][1] - time_ref,
                color=value.get("color", "grey"),
                alpha=value.get("alpha", 0.1),
                label=key,
            )

    def add_model(self, x, y, label, color, alpha=0.6, linewidth=2):
        """
        Add a model and interpolate it across the time span for the plotting.
        :param alpha:
        :param x: the evaluation energies
        :param y: the model values
        :param label: the label of the model
        :param color: the color of the model
        :return: None
        """
        self._data_axis.plot(
            x, y, label=label, color=color, alpha=alpha, zorder=20, linewidth=linewidth
        )

    def add_posteriour(self, x, y, color="grey", alpha=0.002):
        """

        :param x:
        :param y:
        :param lable:
        :param color:
        :return:
        """
        self._data_axis.plot(x, y, color=color, alpha=alpha, zorder=19)

    def add_list_of_sources(self, x, source_list):
        """
        Add a list of model sources and interpolate them across the time span for the plotting.
        :param source_list:
        :param x: the evaluation energies
        :param y: the model values
        :param label: the label of the model
        :param color: the color of the model
        :return: None
        """
        for i, source in enumerate(source_list):
            alpha = source.get("alpha", 0.6)
            linewidth = source.get("linewidth", 2)
            self._data_axis.plot(
                x,
                source["data"],
                color=source["color"],
                label=source["label"],
                alpha=alpha,
                zorder=18,
                linewidth=linewidth,
            )

    def add_data(
        self,
        x,
        y,
        residuals,
        label,
        xerr=None,
        yerr=None,
        residual_yerr=None,
        color="r",
        alpha=0.9,
        show_data=True,
        marker_size=3,
        linewidth=1,
        elinewidth=1,
        rasterized=False,
    ):
        """
        Add the data for the this model
        :param x: energy of the data
        :param y: value of the data
        :param residuals: the residuals for the data
        :param label: label of the data
        :param xerr: the error in energy (or bin width)
        :param yerr: the errorbars of the data
        :param color: color of the
        :return:
        """

        # if we want to show the data

        if show_data:
            self._data_axis.scatter(
                x,
                y,
                s=marker_size,
                linewidths=linewidth,
                facecolors="none",
                edgecolors=color,
                alpha=alpha,
                label=label,
                zorder=15,
                rasterized=rasterized,
            )

        # if we want to show the residuals

        if self._show_residuals:

            # normal residuals from the likelihood

            if not self.ratio_residuals:
                residual_yerr = np.ones_like(residuals)

            self._residual_axis.axhline(0, linestyle="--", color="k")

            self._residual_axis.errorbar(
                x,
                residuals,
                yerr=residual_yerr,
                capsize=0,
                fmt=".",
                elinewidth=elinewidth,
                markersize=marker_size,
                color=color,
                alpha=alpha,
                rasterized=rasterized,
            )

    def add_ppc(
        self,
        rebinned_ppc_rates=None,
        rebinned_time_bin_mean=None,
        result_dir=None,
        model=None,
        plotter=None,
        time_bins=None,
        saa_mask=None,
        echan=None,
        q_levels=[0.68],
        colors=["lightgreen"],
        alpha=0.5,
        bin_width=1e-99,
        n_params=1,
        time_ref=0,
    ):
        """
        Add ppc plot
        :param result_dir: path to result directory
        :param model: Model object
        :param time_bin: Time bins where to compute ppc steps
        :param saa_mask: Mask which time bins are set to zero
        :param echan: Which echan
        :param q_levels: At which levels the ppc should be plotted
        :param colors: colors for the different q_level
        """
        assert len(q_levels) == len(
            colors
        ), "q_levels and colors must have same length!"

        q_levels.sort(reverse=True)

        if rebinned_ppc_rates is None or rebinned_time_bin_mean is None:
            # Get Analyze object from results file of Multinest Fit
            import pymultinest

            analyzer = pymultinest.analyse.Analyzer(n_params, result_dir)

            # Make a mask with 300 random True to choose 300 random samples
            N_samples = 200
            rates = []
            a = np.zeros(
                len(analyzer.get_equal_weighted_posterior()[:, :-1]), dtype=int
            )
            a[:N_samples] = 1
            np.random.shuffle(a)
            a = a.astype(bool)

            # For these 300 random samples calculate the corresponding rates for all time bins
            # with the parameters of this sample
            if using_mpi:
                points_per_rank = float(N_samples) / float(size)
                points_lower_index = int(np.floor(points_per_rank * rank))
                points_upper_index = int(np.floor(points_per_rank * (rank + 1)))
                if rank == 0:
                    with progress_bar(
                        len(
                            analyzer.get_equal_weighted_posterior()[:, :-1][a][
                                points_lower_index:points_upper_index
                            ]
                        ),
                        title="Calculating PPC. This shows the progress of rank 0. All other should be about the same.",
                    ) as p:

                        for i, sample in enumerate(
                            analyzer.get_equal_weighted_posterior()[:, :-1][a][
                                points_lower_index:points_upper_index
                            ]
                        ):
                            synth_data = plotter.get_synthetic_data(sample, model)
                            this_rebinner = Rebinner(
                                synth_data.time_bins - time_ref, bin_width
                            )
                            rebinned_time_bins = this_rebinner.time_rebinned
                            rebinned_counts = this_rebinner.rebin(
                                synth_data.counts[:, echan]
                            )
                            rebinned_bin_length = np.diff(rebinned_time_bins, axis=1).T[
                                0
                            ]
                            rates.append(rebinned_counts / rebinned_bin_length)
                            p.increase()

                else:
                    for i, sample in enumerate(
                        analyzer.get_equal_weighted_posterior()[:, :-1][a][
                            points_lower_index:points_upper_index
                        ]
                    ):
                        synth_data = plotter.get_synthetic_data(sample, model)
                        this_rebinner = Rebinner(
                            synth_data.time_bins - time_ref, bin_width
                        )
                        rebinned_time_bins = this_rebinner.time_rebinned
                        rebinned_counts = this_rebinner.rebin(
                            synth_data.counts[:, echan]
                        )
                        rebinned_bin_length = np.diff(rebinned_time_bins, axis=1).T[0]
                        rates.append(rebinned_counts / rebinned_bin_length)

                rates = np.array(rates)
                print(rates.shape)
                rates_g = comm.gather(rates, root=0)
                if rank == 0:
                    rates_g = np.concatenate(rates_g)
                rates = comm.bcast(rates_g, root=0)
                if rank == 0:
                    for i, level in enumerate(q_levels):
                        low = np.percentile(rates, 50 - 50 * level, axis=0)[0]
                        high = np.percentile(rates, 50 + 50 * level, axis=0)[0]
                        self._data_axis.fill_between(
                            np.mean(rebinned_time_bins, axis=1),
                            low,
                            high,
                            color=colors[i],
                            alpha=alpha,
                        )
            else:
                for i, sample in enumerate(
                    analyzer.get_equal_weighted_posterior()[:, :-1][a]
                ):
                    synth_data = plotter.get_synthetic_data(sample, model)
                    this_rebinner = Rebinner(synth_data.time_bins - time_ref, bin_width)
                    rebinned_time_bins = this_rebinner.time_rebinned
                    rebinned_counts = this_rebinner.rebin(synth_data.counts[:, echan])
                    rebinned_bin_length = np.diff(rebinned_time_bins, axis=1).T[0]
                    rates.append(rebinned_counts / rebinned_bin_length)

                rates = np.array(rates)
                # Plot the q_level areas around median fit
                for i, level in enumerate(q_levels):
                    low = np.percentile(rates, 50 - 50 * level, axis=0)[0]
                    high = np.percentile(rates, 50 + 50 * level, axis=0)[0]
                    self._data_axis.fill_between(
                        np.mean(rebinned_time_bins, axis=1),
                        low,
                        high,
                        color=colors[i],
                        alpha=alpha,
                    )

        else:
            if rank == 0:
                for i, level in enumerate(q_levels):
                    low = np.percentile(rebinned_ppc_rates, 50 - 50 * level, axis=0)
                    high = np.percentile(rebinned_ppc_rates, 50 + 50 * level, axis=0)
                    self._data_axis.fill_between(
                        rebinned_time_bin_mean, low, high, color=colors[i], alpha=alpha
                    )

        # Set Plot range
        # total_mean_rate = np.mean(np.array(rates))
        # self._data_axis.set_ylim((0,3*total_mean_rate))
        # self._data_axis.set_xlim((70000, 80000))

    def finalize(
        self,
        xlabel="x",
        ylabel="y",
        xscale="log",
        yscale="log",
        xticks=None,
        xtick_labels=None,
        show_legend=True,
        invert_y=False,
        xlim=None,
        ylim=None,
        legend_outside=False,
        legend_kwargs=None,
        axis_title=None,
        show_title=False,
        residual_ylim=None,
    ):
        """
        :param xlabel:
        :param ylabel:
        :param xscale:
        :param yscale:
        :param show_legend:
        :return:
        """

        if show_title and axis_title is not None:
            self._data_axis.set_title(axis_title)

        if show_legend and legend_kwargs is not None:
            if "bbox_transform" in legend_kwargs:
                legend_kwargs.pop("bbox_transform")
                self._data_axis.legend(
                    bbox_transform=self._fig.transFigure, **legend_kwargs
                )
            else:
                self._data_axis.legend(**legend_kwargs)

        elif show_legend and legend_outside:
            box = self._data_axis.get_position()
            self._data_axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            self._data_axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        elif show_legend:
            self._data_axis.legend(fontsize="x-small", loc=0)

        self._data_axis.set_ylabel(ylabel)

        self._data_axis.set_xscale(xscale)
        if yscale == "log":

            self._data_axis.set_yscale(yscale, nonposy="clip")

        else:

            self._data_axis.set_yscale(yscale)

        if self._show_residuals:

            self._residual_axis.set_xscale(xscale)

            locator = MaxNLocator(prune="upper", nbins=5)
            self._residual_axis.yaxis.set_major_locator(locator)

            self._residual_axis.set_xlabel(xlabel)

            if self.ratio_residuals:
                custom_warnings.warn(
                    "Residuals plotted as ratios: beware that they are not statistical quantites, and can not be used to asses fit quality"
                )
                self._residual_axis.set_ylabel("Residuals\n(fraction of model)")
            else:
                self._residual_axis.set_ylabel("Residuals\n($\sigma$)")

            if xticks is not None:
                assert len(xticks) == len(xtick_labels)
                self._residual_axis.set_xticks(xticks)
                self._residual_axis.set_xticklabels(xtick_labels)

        else:

            self._data_axis.set_xlabel(xlabel)

            if xticks is not None:
                assert len(xticks) == len(xtick_labels)
                self._data_axis.set_xticks(xticks)
                self._data_axis.set_xticklabels(xtick_labels)

            # This takes care of making space for all labels around the figure

        self._fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        self._fig.subplots_adjust(hspace=0)

        if invert_y:
            self._data_axis.set_ylim(self._data_axis.get_ylim()[::-1])
        if xlim is not None:
            self._data_axis.set_xlim(xlim)
        if ylim is not None:
            self._data_axis.set_ylim(ylim)

        # self._data_axis.set_yscale('log')
        # self._data_axis.set_ylim(bottom=1)
        return self._fig

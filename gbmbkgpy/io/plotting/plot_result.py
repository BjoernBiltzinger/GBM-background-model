import numpy as np
import h5py
import yaml
from datetime import datetime
from matplotlib import pyplot as plt
from gbmgeometry import GBMTime
import astropy.time as astro_time

from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner

NO_REBIN = 1e-9

try:
    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
        rank = 0
except:
    using_mpi = False
    rank = 0


class ResultPlotGenerator(object):
    def __init__(self, config_file, result_dict):

        if isinstance(config_file, dict):
            config = config_file
        else:
            # Load the config.yml
            with open(config_file) as f:
                config = yaml.load(f)

        self._result_dict = result_dict

        self.bin_width = config["plot"].get("bin_width", 10)
        self.change_time = config["plot"].get("change_time", True)
        self.time_since_midnight = config["plot"].get("time_since_midnight", True)
        self.time_format = config["plot"].get("time_format", "h")
        self.time_t0 = config["plot"].get("time_t0", None)
        self.set_axis_limits = config["plot"].get("set_axis_limits", False)
        self.xlim = config["plot"].get("xlim", None)
        self.ylim = config["plot"].get("ylim", None)
        self.residual_ylim = config["plot"].get("residual_ylim", None)
        self.xscale = config["plot"].get("xscale", "linear")
        self.yscale = config["plot"].get("yscale", "linear")
        self.xlabel = config["plot"].get("xlabel", None)
        self.ylabel = config["plot"].get("ylabel", None)
        self.dpi = config["plot"].get("dpi", 400)
        self.show_legend = config["plot"].get("show_legend", True)
        self.show_title = config["plot"].get("show_title", True)
        self.axis_title = config["plot"].get("axis_title", None)
        self.legend_outside = config["plot"].get("legend_outside", True)

        # Import component settings
        self.show_data = config["component"].get("show_data", True)
        self.show_model = config["component"].get("show_model", True)
        self.show_ppc = config["component"].get("show_ppc", True)
        self.show_residuals = config["component"].get("show_residuals", False)
        self.show_all_sources = config["component"].get("show_all_sources", True)
        self.show_earth = config["component"].get("show_earth", True)
        self.show_cgb = config["component"].get("show_cgb", True)
        self.show_sun = config["component"].get("show_sun", True)
        self.show_saa = config["component"].get("show_saa", True)
        self.show_cr = config["component"].get("show_cr", True)
        self.show_constant = config["component"].get("show_constant", True)
        self.show_crab = config["component"].get("show_crab", True)
        self.show_occ_region = config["component"].get("show_occ_region", False)
        self.show_grb_trigger = config["component"].get("show_grb_trigger", False)

        # Import style settings
        self.model_styles = config["style"]["model"]
        self.source_styles = config["style"]["sources"]
        self.ppc_styles = config["style"]["ppc"]
        self.data_styles = config["style"]["data"]
        self.legend_kwargs = config["style"].get("legend_kwargs", None)

        if config["style"]["mpl_style"] is not None:
            plt.style.use(config["style"]["mpl_style"])

        self._grb_triggers = {}
        self._occ_region = {}

        if config.get("highlight", None) is not None:
            if config["highlight"]["grb_trigger"] is not None:
                for grb_trigger in config["highlight"]["grb_trigger"]:
                    self.add_grb_trigger(
                        grb_name=grb_trigger["name"],
                        trigger_time=grb_trigger["trigger_time"],
                        time_format="UTC",
                        time_offset=grb_trigger["time_offset"],
                        color=grb_trigger.get("color", "b"),
                        alpha=grb_trigger.get("alpha", 0.3),
                        linestyle=grb_trigger.get("linestyle", "-"),
                        linewidth=grb_trigger.get("linewidth", 0.8),
                    )
            if config["highlight"]["occ_region"] is not None:
                for occ_region in config["highlight"]["occ_region"]:
                    self.add_occ_region(
                        occ_name=occ_region["name"],
                        time_start=occ_region["time_start"],
                        time_stop=occ_region["time_stop"],
                        time_format=occ_region.get("time_format", "UTC"),
                        color=occ_region.get("color", "grey"),
                        alpha=occ_region.get("alpha", 0.1),
                    )
        self._plot_path_list = []

    @classmethod
    def from_result_file(cls, config_file, result_data_file):

        result_dict = {}

        print("Load result file for plotting from: {}".format(result_data_file))

        with h5py.File(result_data_file, "r") as f:

            result_dict["dates"] = f.attrs["dates"]
            result_dict["detectors"] = f.attrs["detectors"]
            result_dict["echans"] = f.attrs["echans"]
            result_dict["param_names"] = f.attrs["param_names"]
            result_dict["best_fit_values"] = f.attrs["best_fit_values"]

            result_dict["day_start_times"] = f["day_start_times"][()]
            result_dict["day_stop_times"] = f["day_stop_times"][()]
            result_dict["time_bins_start"] = f["time_bins_start"][()]
            result_dict["time_bins_stop"] = f["time_bins_stop"][()]
            result_dict["total_time_bins"] = np.vstack(
                (result_dict["time_bins_start"], result_dict["time_bins_stop"])
            ).T
            result_dict["saa_mask"] = f["saa_mask"][()]

            result_dict["model_counts"] = f["model_counts"][()]
            result_dict["observed_counts"] = f["observed_counts"][()]

            result_dict["sources"] = {}

            for source_name in f["sources"].keys():
                result_dict["sources"][source_name] = f["sources"][source_name][()]

            result_dict["ppc_counts"] = f["ppc_counts"][()]

            result_dict["time_stamp"] = datetime.now().strftime("%y%m%d_%H%M")

        return cls(config_file=config_file, result_dict=result_dict)

    @classmethod
    def from_result_instance(cls, config_file, data, model, saa_object):
        result_dict = {}

        result_dict["detectors"] = data.detectors
        result_dict["dates"] = data.dates
        result_dict["echans"] = data.echans
        result_dict["day_start_times"] = data.day_start_times
        result_dict["day_stop_times"] = data.day_stop_times
        result_dict["total_time_bins"] = data.time_bins
        result_dict["saa_mask"] = saa_object.saa_mask

        result_dict["observed_counts"] = set_saa_zero(
            data.counts, saa_mask=result_dict["saa_mask"]
        )

        result_dict["model_counts"] = model.get_counts(
            time_bins=result_dict["total_time_bins"], saa_mask=result_dict["saa_mask"]
        )

        result_dict["sources"] = {}

        for i, source_name in enumerate(model.continuum_sources):
            data = model.get_continuum_counts(
                i, result_dict["total_time_bins"], result_dict["saa_mask"]
            )
            if np.sum(data) != 0:
                result_dict["sources"][source_name] = data

        for i, source_name in enumerate(model.global_sources):
            data = model.get_global_counts(
                i, result_dict["total_time_bins"], result_dict["saa_mask"]
            )
            if np.sum(data) != 0:
                result_dict["sources"][source_name] = data

        for i, source_name in enumerate(model.fit_spectrum_sources):
            data = model.get_fit_spectrum_counts(
                i, result_dict["total_time_bins"], result_dict["saa_mask"]
            )
            if np.sum(data) != 0:
                result_dict["sources"][source_name] = data

        saa_data = model.get_saa_counts(
            result_dict["total_time_bins"], result_dict["saa_mask"]
        )
        if np.sum(saa_data) != 0:
            result_dict["sources"]["SAA_decays"] = saa_data

        for i, point_source in enumerate(model.point_sources):
            data = model.get_point_source_counts(
                i, result_dict["total_time_bins"], result_dict["saa_mask"]
            )
            if np.sum(data) != 0:
                result_dict["sources"][point_source] = data

        result_dict["time_stamp"] = datetime.now().strftime("%y%m%d_%H%M")

        # TODO: Add PPC calc
        result_dict["ppc_counts"] = []

        return cls(config_file=config_file, result_dict=result_dict)

    def create_plots(self, output_dir):

        for day_idx, day in enumerate(self._result_dict["dates"]):

            for det_idx, det in enumerate(self._result_dict["detectors"]):

                for echan_idx, echan in enumerate(self._result_dict["echans"]):

                    total_steps = (
                        12
                        if self.show_ppc is False
                        else 12 + len(self._result_dict["ppc_counts"])
                    )

                    time_stamp = datetime.now().strftime("%y%m%d_%H%M")

                    plot_path = (
                        f"{output_dir}/"
                        f"plot_date_{day}_"
                        f"det_{det}_"
                        f"echan_{echan}__"
                        f"{time_stamp}.pdf"
                    )

                    self._plot_path_list.append(plot_path)

                    with progress_bar(total_steps, title="Create Result plot") as p:
                        self._create_model_plots(
                            p_bar=p,
                            det=det,
                            det_idx=det_idx,
                            echan=echan,
                            echan_idx=echan_idx,
                            day_idx=day_idx,
                            savepath=plot_path,
                        )

    def _create_model_plots(
        self,
        p_bar,
        det,
        det_idx,
        echan,
        echan_idx,
        day_idx=None,
        savepath="test.pdf",
        **kwargs,
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

        if self.time_since_midnight and self.time_t0 is not None:
            raise ValueError(
                "You selected time since midnight and passed a t0, you should choose one..."
            )

        # Change time reference to seconds since beginning of the day
        if self.time_since_midnight and day_idx is not None:
            assert day_idx < len(self._result_dict["dates"]), "Use a valid date..."
            self._time_ref = self._result_dict["day_start_times"][day_idx]
            time_frame = "Time since midnight [{}]".format(self.time_format)

        elif self.time_t0 is not None:
            # if it is of type string, assume we are dealing with a UTC timestamp
            if isinstance(self.time_t0, str):

                day_at = astro_time.Time("%s(UTC)" % self.time_t0)

                self._time_ref = GBMTime(day_at).met

            # if it is of type float, assume it is in MET
            elif isinstance(self.time_t0, float):
                self._time_ref = self.time_t0

            time_frame = "t-t$_0$ [{}]".format(self.time_format)

        else:
            self._time_ref = 0
            time_frame = "MET [s]"

        p_bar.increase()

        residual_plot = ResidualPlot(show_residuals=self.show_residuals, **kwargs)

        if self.bin_width > NO_REBIN:

            this_rebinner = Rebinner(
                (self._result_dict["total_time_bins"] - self._time_ref),
                self.bin_width,
                self._result_dict["saa_mask"],
            )

            self._rebinned_observed_counts = this_rebinner.rebin(
                self._result_dict["observed_counts"][:, det_idx, echan_idx]
            )[0]

            self._rebinned_model_counts = this_rebinner.rebin(
                self._result_dict["model_counts"][:, det_idx, echan_idx]
            )[0]

            self._rebinned_time_bins = this_rebinner.time_rebinned

            rebin = True

        else:

            self._rebinned_observed_counts = self._result_dict["observed_counts"][
                :, det_idx, echan_idx
            ]

            self._rebinned_model_counts = self._result_dict["model_counts"][
                :, det_idx, echan_idx
            ]

            self._rebinned_time_bins = (
                self._result_dict["total_time_bins"] - self._time_ref
            )

            rebin = False

        self._rebinned_background_counts = np.zeros_like(self._rebinned_observed_counts)

        self._rebinned_time_bin_widths = np.diff(self._rebinned_time_bins, axis=1)[:, 0]

        self._rebinned_time_bin_mean = np.mean(self._rebinned_time_bins, axis=1)

        significance_calc = Significance(
            self._rebinned_observed_counts,
            self._rebinned_background_counts + self._rebinned_model_counts,
            1,
        )

        residual_errors = None
        self._residuals = significance_calc.known_background()

        p_bar.increase()

        residual_plot.add_data(
            self._rebinned_time_bin_mean,
            self._rebinned_observed_counts / self._rebinned_time_bin_widths,
            self._residuals,
            residual_yerr=residual_errors,
            yerr=None,
            xerr=None,
            label="Obs. Count Rates"
            if self.data_styles.get("show_label", True)
            else None,
            color=self.data_styles.get("color", "black"),
            alpha=self.data_styles.get("alpha", 0.6),
            show_data=self.show_data,
            marker_size=self.data_styles.get("marker_size", 0.5),
            linewidth=self.data_styles.get("linewidth", 0.2),
            elinewidth=self.data_styles.get("elinewidth", 0.5),
            rasterized=self.data_styles.get("rasterized", False),
        )
        p_bar.increase()

        if self.show_model:
            residual_plot.add_model(
                self._rebinned_time_bin_mean,
                self._rebinned_model_counts / self._rebinned_time_bin_widths,
                label="Best Fit" if self.model_styles.get("show_label", True) else None,
                color=self.model_styles.get("color", "red"),
                alpha=self.model_styles.get("alpha", 0.9),
                linewidth=self.model_styles.get("linewidth", 0.8),
            )

        p_bar.increase()

        src_list = []
        for i, (key, value) in enumerate(self._result_dict["sources"].items()):
            if "L-parameter" in key or "BGO_CR_Approx" in key:
                label = "Cosmic Rays"
                style_key = "cr"
                sort_idx = 0
                if not self.show_all_sources and not self.show_cr:
                    continue

            elif "Earth" in key:
                label = "Earth Albedo"
                style_key = "earth"
                sort_idx = 1
                if not self.show_all_sources and not self.show_earth:
                    continue

            elif "CGB" in key:
                label = "CGB"
                style_key = "cgb"
                sort_idx = 2
                if not self.show_all_sources and not self.show_cgb:
                    continue

            elif "Constant" in key:
                label = "Constant"
                style_key = "constant"
                sort_idx = 3
                if not self.show_all_sources and not self.show_constant:
                    continue

            elif "SAA_decays" in key:
                label = "SAA Exits"
                style_key = "saa"
                sort_idx = 4
                if not self.show_all_sources and not self.show_saa:
                    continue

            elif "CRAB" in key:
                label = "Crab"
                style_key = "crab"
                sort_idx = 5
                if not self.show_all_sources and not self.show_crab:
                    continue

            elif "sun" in key:
                label = "Sun"
                style_key = "sun"
                sort_idx = 6
                if not self.show_all_sources and not self.show_sun:
                    continue

            else:
                label = key
                style_key = "default"
                sort_idx = i + len(self._result_dict["sources"].items())
                if not self.show_all_sources:
                    continue

            if rebin:
                rebinned_source_counts = this_rebinner.rebin(
                    self._result_dict["sources"][key][:, det_idx, echan_idx]
                )[0]
            else:
                rebinned_source_counts = self._result_dict["sources"][key][
                    :, det_idx, echan_idx
                ]

            if np.sum(rebinned_source_counts) > 0.0:

                src_list.append(
                    {
                        "data": rebinned_source_counts / self._rebinned_time_bin_widths,
                        "label": label
                        if self.source_styles[style_key].get("show_label", True)
                        or self.source_styles["use_global"]
                        else None,
                        "color": self.source_styles[style_key]["color"]
                        if not self.source_styles["use_global"]
                        else None,
                        "alpha": self.source_styles[style_key]["alpha"]
                        if not self.source_styles["use_global"]
                        else None,
                        "linewidth": self.source_styles[style_key]["linewidth"]
                        if not self.source_styles["use_global"]
                        else None,
                        "sort_idx": sort_idx,
                    }
                )

        self._source_list = sorted(src_list, key=lambda src: src["sort_idx"])

        if self.source_styles["use_global"]:
            cmap = plt.get_cmap(self.source_styles["global"]["cmap"])
            colors = cmap(np.linspace(0, 1, len(self._source_list)))

            for i, source in enumerate(self._source_list):
                source["color"] = colors[i]
                source["alpha"] = self.source_styles["global"]["alpha"]
                source["linewidth"] = self.source_styles["global"]["linewidth"]
                source["label"] = (
                    source["label"]
                    if self.source_styles["global"]["show_label"]
                    else None
                )

        if len(self._source_list) > 0:
            residual_plot.add_list_of_sources(
                self._rebinned_time_bin_mean, self._source_list
            )

        p_bar.increase()

        if self.show_ppc:
            rebinned_ppc_rates = []

            ppc_counts_det_echan = self._result_dict["ppc_counts"][
                :, :, det_idx, echan_idx
            ]

            for j, ppc in enumerate(ppc_counts_det_echan):
                set_saa_zero(
                    ppc_counts_det_echan[j], saa_mask=self._result_dict["saa_mask"],
                )
                if rebin:
                    rebinned_ppc_rates.append(
                        this_rebinner.rebin(ppc_counts_det_echan[j])
                        / self._rebinned_time_bin_widths
                    )
                else:
                    rebinned_ppc_rates.append(
                        ppc_counts_det_echan[j] / self._rebinned_time_bin_widths
                    )

                p_bar.increase()
            rebinned_ppc_rates = np.array(rebinned_ppc_rates)

            residual_plot.add_ppc(
                rebinned_ppc_rates=rebinned_ppc_rates,
                rebinned_time_bin_mean=self._rebinned_time_bin_mean,
                q_levels=[0.68, 0.95, 0.99],
                colors=self.ppc_styles["color"],
                alpha=self.ppc_styles["alpha"],
            )

        # Add vertical lines for grb triggers
        if self.show_grb_trigger:
            residual_plot.add_vertical_line(self._grb_triggers, self._time_ref)

        if self.show_occ_region:
            residual_plot.add_occ_region(self._occ_region, self._time_ref)

        if self.set_axis_limits:
            if self.xlim is None or self.ylim is None:
                xlim, ylim = self._calc_limits(day_idx)
                self.xlim = xlim if self.xlim is None else self.xlim
                self.ylim = ylim if self.ylim is None else self.ylim

            if self.time_format == "h":
                xticks = []
                xtick_labels = []
                for xstep in range(
                    int(self.xlim[0] / 3600), int((self.xlim[1] + 500) / 3600) + 1, 4
                ):
                    xticks.append(xstep * 3600)
                    xtick_labels.append("%s" % xstep)

            elif self.time_format == "s":
                xticks = None
                xtick_labels = None

            else:
                raise NotImplementedError(
                    "Please provide a valid time format: ['h', 's']"
                )
        else:
            xticks = None
            xtick_labels = None

        p_bar.increase()

        xlabel = "{}".format(time_frame) if self.xlabel is None else self.xlabel
        ylabel = "Count Rate [counts s$^{-1}$]" if self.ylabel is None else self.ylabel

        axis_title = (
            "Detector: {} Date: {} Energy: {}".format(
                det, self._result_dict["dates"][day_idx], self._get_echan_str(echan)
            )
            if self.axis_title is None
            else self.axis_title
        )

        final_plot = residual_plot.finalize(
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=self.xscale,
            yscale=self.yscale,
            xticks=xticks,
            xtick_labels=xtick_labels,
            show_legend=self.show_legend,
            xlim=self.xlim,
            ylim=self.ylim,
            residual_ylim=self.residual_ylim,
            legend_outside=self.legend_outside,
            legend_kwargs=self.legend_kwargs,
            axis_title=axis_title,
            show_title=self.show_title,
        )

        p_bar.increase()

        if rank == 0:
            final_plot.savefig(savepath, dpi=self.dpi)

        p_bar.increase()

    def add_grb_trigger(
        self,
        grb_name,
        trigger_time,
        time_format="UTC",
        time_offset=0,
        color="b",
        alpha=0.3,
        linestyle="-",
        linewidth=1,
    ):
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
        if time_format == "UTC":
            day_at = astro_time.Time("%s(UTC)" % trigger_time)

            met = GBMTime(day_at).met + time_offset

        elif time_format == "MET":
            met = trigger_time

        else:
            raise Exception("Not supported time format, please use MET or UTC")

        self._grb_triggers[grb_name] = {
            "met": met,
            "color": color,
            "alpha": alpha,
            "linestyle": linestyle,
            "linewidth": linewidth,
        }

    def add_occ_region(
        self,
        occ_name,
        time_start,
        time_stop,
        time_format="UTC",
        color="grey",
        alpha=0.1,
    ):
        """

        :param time_format:
        :param occ_name:
        :param time_start: string in MET or UTC '2008-01-01T00:23:11.997'
        :param time_stop: string in MET or UTC '2008-01-01T00:23:11.997'
        :param color:
        :return:
        """
        if time_format == "UTC":
            t_start = astro_time.Time("%s(UTC)" % time_start)
            t_stop = astro_time.Time("%s(UTC)" % time_stop)

            met_start = GBMTime(t_start).met
            met_stop = GBMTime(t_stop).met

        elif time_format == "MET":
            met_start = time_start
            met_stop = time_stop
        else:
            raise Exception("Not supported time format, please use MET or UTC")

        self._occ_region[occ_name] = {
            "met": (met_start, met_stop),
            "color": color,
            "alpha": alpha,
        }

    def _calc_limits(self, which_day):
        min_time = self._result_dict["day_start_times"][which_day] - self._time_ref
        max_time = self._result_dict["day_stop_times"][which_day] - self._time_ref

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
            if (
                time_bins_masked[:, 1][index_start[i + 1]]
                - time_bins_masked[:, 0][index_stop[i]]
                < 1000
            ):
                zero_counts_mask[index_stop[i] - 5 : index_start[i + 1] + 5] = (
                    np.ones_like(
                        zero_counts_mask[index_stop[i] - 5 : index_start[i + 1] + 5]
                    )
                    == 2
                )

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

        if self.show_data or self.show_model:
            obs_rates_masked2 = (
                obs_counts_masked[zero_counts_mask]
                / np.diff(time_bins_masked2, axis=1)[0]
            )
            high_lim = 1.5 * np.percentile(obs_rates_masked2, 99)
        else:
            high_lim = 0.0
            for source in self._source_list:
                source_rates_masked = source["data"][zero_counts_mask]
                h_lim = 1.5 * np.percentile(source_rates_masked, 99)
                high_lim = h_lim if h_lim > high_lim else high_lim

        ylim = (0, high_lim)
        return xlim, ylim

    def _get_echan_str(self, echan):
        # TODO: Add information for CSPEC
        echan_dict = {
            "0": "4 keV - 12 keV",
            "1": "12 keV - 27 keV",
            "2": "27 keV - 50 keV",
            "3": "50 keV - 102 keV",
            "4": "102 keV - 295 keV",
            "5": "295 keV - 540 keV",
            "6": "540 keV - 985 keV",
            "7": "985 keV - 2 MeV",
        }

        return echan_dict[str(echan)]


def set_saa_zero(vector, saa_mask):
    vector[np.where(~saa_mask)] = 0.0
    return vector

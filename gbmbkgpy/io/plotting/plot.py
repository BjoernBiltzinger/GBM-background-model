import numpy as np
import copy
from gbmbkgpy.utils.statistics.stats_tools import Significance
from gbmbkgpy.io.plotting.data_residual_plot import ResidualPlot
from gbmbkgpy.utils.binner import Rebinner
import h5py
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmgeometry import GBMTime
import astropy.time as astro_time

NO_REBIN = 1e-99

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
except:

    using_mpi = False


class Plotter(object):
    def __init__(self, data, model, saa_object, echan_list):

        self._data = data
        self._model = model  # type: Model
        self._echan_list = np.arange(
            len(echan_list)
        )  # list of all echans which should be fitted

        self._name = "Count rate detector %s" % data._det

        # The MET start time of the first used day
        self._day_met = data.day_met[0]

        self._free_parameters = self._model.free_parameters
        self._parameters = self._model.parameters

        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = data.time_bins
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        # Get the SAA and GRB mask:
        self._saa_mask = saa_object.saa_mask
        self._grb_mask = np.ones(
            len(self._total_time_bins), dtype=bool
        )  # np.full(len(self._total_time_bins), True)
        # An entry in the total mask is False when one of the two masks is False
        self._total_mask = ~np.logical_xor(self._saa_mask, self._grb_mask)

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._total_mask]

        # Extract the counts from the data object. should be same size as time bins. For all echans together
        self._counts_all_echan = data.counts[self._total_mask]
        self._total_counts_all_echan = data.counts

        self._total_scale_factor = 1.0
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False

        self._grb_triggers = {}
        self._occ_region = {}

    def display_model(
        self,
        index,
        data_color="k",
        model_color="r",
        step=True,
        show_data=True,
        show_residuals=True,
        show_legend=True,
        min_bin_width=1e-99,
        plot_sources=False,
        show_grb_trigger=False,
        show_model=True,
        change_time=False,
        show_occ_region=False,
        posteriour=None,
        ppc=False,
        result_dir=None,
        xlim=None,
        ylim=None,
        legend_outside=False,
        **kwargs
    ):

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
        :param ppc: (bool) show ppc
        :return:
        """

        # Change time reference to seconds since beginning of the day
        if change_time:
            time_ref = self._day_met
            time_frame = "Seconds since midnight"
        else:
            time_ref = 0.0
            time_frame = "MET"

        model_label = "Background fit"

        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if min_bin_width != NO_REBIN:
            this_rebinner = Rebinner(
                (self._total_time_bins - time_ref), min_bin_width, self._saa_mask
            )

            # we need to get the rebinned counts

            (self._rebinned_observed_counts,) = this_rebinner.rebin(
                self._total_counts_all_echan[:, index]
            )

            # the rebinned counts expected from the model
            (self._rebinned_model_counts,) = this_rebinner.rebin(
                self._model.get_counts(
                    self._total_time_bins, index, saa_mask=self._saa_mask
                )
            )

            self._rebinned_background_counts = np.zeros_like(
                self._rebinned_observed_counts
            )

            self._rebinned_time_bins = this_rebinner.time_rebinned

            self._rebinned_time_bin_widths = np.diff(self._rebinned_time_bins, axis=1)[
                :, 0
            ]

        else:
            self._rebinned_observed_counts = self._total_counts_all_echan[:, index]

            # the rebinned counts expected from the model
            self._rebinned_model_counts = self._model.get_counts(
                self._total_time_bins, index, saa_mask=self._saa_mask
            )

            self._rebinned_background_counts = np.zeros_like(
                self._rebinned_observed_counts
            )

            self._rebinned_time_bins = self._total_time_bins - time_ref

            self._rebinned_time_bin_widths = self._total_time_bin_widths

        # Residuals
        significance_calc = Significance(
            self._rebinned_observed_counts,
            self._rebinned_background_counts
            + self._rebinned_model_counts / self._total_scale_factor,
            self._total_scale_factor,
        )

        residual_errors = None
        self._residuals = significance_calc.known_background()
        if ppc:
            if result_dir == None:
                print(
                    "No ppc possible, no results directonary given to display method!"
                )
            else:
                ppc_model = copy.deepcopy(self._model)
                n_params = len(self._model.free_parameters)
                residual_plot.add_ppc(
                    result_dir=result_dir,
                    model=ppc_model,
                    plotter=self,
                    time_bins=self._total_time_bins - time_ref,
                    saa_mask=self._saa_mask,
                    echan=index,
                    q_levels=[0.68, 0.95, 0.99],
                    colors=["lightgreen", "green", "darkgreen"],
                    bin_width=min_bin_width,
                    n_params=n_params,
                    time_ref=time_ref,
                )

        residual_plot.add_data(
            np.mean(self._rebinned_time_bins, axis=1),
            self._rebinned_observed_counts / self._rebinned_time_bin_widths,
            self._residuals,
            residual_yerr=residual_errors,
            yerr=None,
            xerr=None,
            label=self._name,
            color=data_color,
            show_data=show_data,
            marker_size=1.5,
        )

        y = (
            self._model.get_counts(
                self._total_time_bins, index, saa_mask=self._saa_mask
            )
        ) / self._total_time_bin_widths

        x = np.mean(self._total_time_bins - time_ref, axis=1)

        if show_model:
            residual_plot.add_model(x, y, label=model_label, color=model_color)
        if posteriour is not None:
            # Make a copy of the model for plotting
            plot_model = copy.deepcopy(self._model)

            # Use every tenth result to save memory
            posterior_sample = posteriour[::10]
            for j in range(len(posterior_sample)):
                plot_model.set_free_parameters(
                    posterior_sample[j][2:]
                )  # The first 2 values are not the parameters

                post_model_counts = plot_model.get_counts(
                    self._total_time_bins, saa_mask=self._saa_mask
                )

                (rebinned_post_model_counts,) = this_rebinner.rebin(post_model_counts)

                x_post = np.mean(self._rebinned_time_bins, axis=1)
                y_post = rebinned_post_model_counts / self._rebinned_time_bin_widths

                residual_plot.add_posteriour(x_post, y_post, alpha=0.02)

        if plot_sources:
            source_list = self._get_list_of_sources(
                self._total_time_bins - time_ref, index, self._total_time_bin_widths
            )

            residual_plot.add_list_of_sources(x, source_list)

        # Add vertical lines for grb triggers

        if show_grb_trigger:
            residual_plot.add_vertical_line(self._grb_triggers, time_ref)

        if show_occ_region:
            residual_plot.add_occ_region(self._occ_region, time_ref)

        return residual_plot.finalize(
            xlabel="Time\n(%s)" % time_frame,
            ylabel="Count Rate\n(counts s$^{-1}$)",
            xscale="linear",
            yscale="linear",
            show_legend=show_legend,
            xlim=xlim,
            ylim=ylim,
            legend_outside=legend_outside,
        )

    def _save_plotting_data(self, path, result_dir, echan_list):
        """
        Function to save the data needed to create the plots.
        """
        ppc_counts_all = []
        for index in self._echan_list:
            ppc_counts_all.append(self.ppc_data(result_dir, index))
        if rank == 0:
            with h5py.File(path, "w") as f1:

                group_general = f1.create_group("general")

                group_general.create_dataset("Detector", data=self._data.det)
                group_general.create_dataset("Dates", data=self._data.day)
                group_general.create_dataset(
                    "day_start_times", data=self._data.day_start_times
                )
                group_general.create_dataset(
                    "day_stop_times", data=self._data.day_stop_times
                )
                group_general.create_dataset(
                    "saa_mask",
                    data=self._saa_mask,
                    compression="gzip",
                    compression_opts=9,
                )

                for j, index in enumerate(self._echan_list):
                    source_list = self.get_counts_of_sources(
                        self._total_time_bins, index
                    )

                    model_counts = self._model.get_counts(
                        self._total_time_bins, index, saa_mask=self._saa_mask
                    )

                    time_bins = self._total_time_bins

                    ppc_counts = ppc_counts_all[j]

                    observed_counts = self._total_counts_all_echan[:, index]

                    group_echan = f1.create_group("Echan {}".format(echan_list[j]))

                    # group_ppc = group_echan.create_group('PPC data')

                    group_sources = group_echan.create_group("Sources")

                    group_echan.create_dataset(
                        "time_bins_start",
                        data=time_bins[:, 0],
                        compression="gzip",
                        compression_opts=9,
                    )
                    group_echan.create_dataset(
                        "time_bins_stop",
                        data=time_bins[:, 1],
                        compression="gzip",
                        compression_opts=9,
                    )
                    group_echan.create_dataset(
                        "total_model_counts",
                        data=model_counts,
                        compression="gzip",
                        compression_opts=9,
                    )
                    group_echan.create_dataset(
                        "observed_counts",
                        data=observed_counts,
                        compression="gzip",
                        compression_opts=9,
                    )
                    for source in source_list:
                        group_sources.create_dataset(
                            source["label"],
                            data=source["data"],
                            compression="gzip",
                            compression_opts=9,
                        )
                    group_echan.create_dataset(
                        "PPC", data=ppc_counts, compression="gzip", compression_opts=9
                    )

    def ppc_data(self, result_dir, echan):
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
        import pymultinest

        analyzer = pymultinest.analyse.Analyzer(1, result_dir)
        mn_posteriour_samples = analyzer.get_equal_weighted_posterior()[:, :-1]

        counts = []

        # Make a mask with 500 random True to choose N_samples random samples,
        # if multinest returns less then 500 posterior samples use next smaller *00
        N_samples = (
            500
            if len(mn_posteriour_samples) > 500
            else int(len(mn_posteriour_samples) / 100) * 100
        )

        a = np.zeros(len(mn_posteriour_samples), dtype=int)
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
                        mn_posteriour_samples[a][points_lower_index:points_upper_index]
                    ),
                    title="Calculating PPC for echan {}".format(echan),
                ) as p:

                    for i, sample in enumerate(
                        mn_posteriour_samples[a][points_lower_index:points_upper_index]
                    ):
                        synth_data = self.get_synthetic_data(sample)
                        counts.append(synth_data.counts[:, echan])
                        p.increase()

            else:
                for i, sample in enumerate(
                    mn_posteriour_samples[a][points_lower_index:points_upper_index]
                ):
                    synth_data = self.get_synthetic_data(sample)
                    counts.append(synth_data.counts[:, echan])
            counts = np.array(counts)
            counts_g = comm.gather(counts, root=0)

            if rank == 0:
                counts_g = np.concatenate(counts_g)
            counts = comm.bcast(counts_g, root=0)
        else:
            for i, sample in enumerate(
                analyzer.get_equal_weighted_posterior()[:, :-1][a]
            ):
                synth_data = self.get_synthetic_data(sample)
                counts.append(synth_data.counts[:, echan])
            counts = np.array(counts)
        return counts

    def _get_list_of_sources(self, time_bins, echan, time_bin_width=1.0):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """
        source_list = []
        color_list = ["b", "g", "c", "m", "y", "k", "navy", "darkgreen", "cyan"]
        i_index = 0
        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, time_bins, self._saa_mask, echan)
            if np.sum(data) != 0:
                source_list.append(
                    {
                        "label": source_name,
                        "data": data / time_bin_width,
                        "color": color_list[i_index],
                    }
                )
                i_index += 1
        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(i, time_bins, self._saa_mask, echan)
            source_list.append(
                {
                    "label": source_name,
                    "data": data / time_bin_width,
                    "color": color_list[i_index],
                }
            )
            i_index += 1

        for i, source_name in enumerate(self._model.fit_spectrum_sources):
            data = self._model.get_fit_spectrum_counts(
                i, time_bins, self._saa_mask, echan
            )
            source_list.append(
                {
                    "label": source_name,
                    "data": data / time_bin_width,
                    "color": color_list[i_index],
                }
            )
            i_index += 1

        saa_data = self._model.get_saa_counts(
            self._total_time_bins, self._saa_mask, echan
        )
        if np.sum(saa_data) != 0:
            source_list.append(
                {
                    "label": "SAA_decays",
                    "data": saa_data / time_bin_width,
                    "color": color_list[i_index],
                }
            )
            i_index += 1
        point_source_data = self._model.get_point_source_counts(
            self._total_time_bins, self._saa_mask, echan
        )
        if np.sum(point_source_data) != 0:
            source_list.append(
                {
                    "label": "Point_sources",
                    "data": point_source_data / time_bin_width,
                    "color": color_list[i_index],
                }
            )
            i_index += 1
        return source_list

    def add_grb_trigger(
        self, grb_name, trigger_time, time_format="UTC", time_offset=0, color="b"
    ):
        """
        Add a GRB Trigger to plot a vertical line
        The grb is added to a dictionary with the name as key and the time (met) and the color as values in a subdict
        A time offset can be used to add line in reference to a trigger
        :param grb_name: string
        :param trigger_time: string in UTC '00:23:11.997'
        :return:
        """
        if time_format == "UTC":
            day = self._data.day
            year = "20%s" % day[:2]
            month = day[2:-2]
            dd = day[-2:]

            day_at = astro_time.Time(
                "%s-%s-%sT%s(UTC)" % (year, month, dd, trigger_time)
            )

            met = GBMTime(day_at).met + time_offset

        if time_format == "MET":
            met = trigger_time

        self._grb_triggers[grb_name] = {"met": met, "color": color}

    def add_occ_region(
        self, occ_name, time_start, time_stop, time_format="UTC", color="grey"
    ):
        """

        :param occ_name:
        :param start_time:
        :param stop_time:
        :param color:
        :return:
        """
        if time_format == "UTC":
            day = self._data.day
            year = "20%s" % day[:2]
            month = day[2:-2]
            dd = day[-2:]
            t_start = astro_time.Time(
                "%s-%s-%sT%s(UTC)" % (year, month, dd, time_start)
            )
            t_stop = astro_time.Time("%s-%s-%sT%s(UTC)" % (year, month, dd, time_stop))

            met_start = GBMTime(t_start).met
            met_stop = GBMTime(t_stop).met

        if time_format == "MET":
            met_start = time_start
            met_stop = time_stop

        self._occ_region[occ_name] = {"met": (met_start, met_stop), "color": color}

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

        for i, parameter in enumerate(synth_model.free_parameters.values()):
            parameter.value = synth_parameters[i]

        for echan in self._echan_list:
            synth_data.counts[:, echan] = np.random.poisson(
                synth_model.get_counts(synth_data.time_bins, echan)
            )

        return synth_data

    def get_counts_of_sources(self, time_bins, echan):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """

        source_list = []
        color_list = ["b", "g", "c", "m", "y", "k", "navy", "darkgreen", "cyan"]
        i_index = 0
        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, time_bins, self._saa_mask, echan)
            if np.sum(data) != 0:
                source_list.append(
                    {"label": source_name, "data": data, "color": color_list[i_index]}
                )
                i_index += 1
        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(i, time_bins, self._saa_mask, echan)
            source_list.append(
                {"label": source_name, "data": data, "color": color_list[i_index]}
            )
            i_index += 1

        for i, source_name in enumerate(self._model.fit_spectrum_sources):
            data = self._model.get_fit_spectrum_counts(
                i, time_bins, self._saa_mask, echan
            )
            source_list.append(
                {"label": source_name, "data": data, "color": color_list[i_index]}
            )
            i_index += 1

        saa_data = self._model.get_saa_counts(
            self._total_time_bins, self._saa_mask, echan
        )
        if np.sum(saa_data) != 0:
            source_list.append(
                {"label": "SAA_decays", "data": saa_data, "color": color_list[i_index]}
            )
            i_index += 1
        point_source_data = self._model.get_point_source_counts(
            self._total_time_bins, self._saa_mask, echan
        )
        if np.sum(point_source_data) != 0:
            source_list.append(
                {
                    "label": "Point_sources",
                    "data": point_source_data,
                    "color": color_list[i_index],
                }
            )
            i_index += 1
        return source_list

import os
import numpy as np
import copy

from gbmbkgpy.utils.pha import SPECTRUM, PHAII
import h5py
from gbmbkgpy.utils.progress_bar import progress_bar
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


class DataExporter(object):
    def __init__(self, model_generator, best_fit_values):

        self._best_fit_values = best_fit_values
        self._total_scale_factor = 1.0

        # Create a copy of the response precalculation
        response_precalculation = model_generator._resp

        # Create a copy of the geomtry precalculation
        geometry_precalculation = model_generator._geom

        config = model_generator.config

        if (
            config["export"]["save_unbinned"]
            and config["general"]["min_bin_width"] > 1e-9
        ):

            # Create copy of config dictionary
            config_export = config

            config_export["general"]["min_bin_width"] = 1e-9

            # Create a new model generator instance of the same type
            model_generator_unbinned = type(model_generator)()

            model_generator_unbinned.from_config_dict(
                config=config_export,
                response=response_precalculation,
                geometry=geometry_precalculation,
            )

            model_generator_unbinned.likelihood.set_free_parameters(best_fit_values)

            self._data = model_generator_unbinned.data
            self._model = model_generator_unbinned.model
            self._time_bins = model_generator_unbinned.data.time_bins
            self._saa_mask = model_generator_unbinned.saa_calc.saa_mask

        else:

            self._data = model_generator.data
            self._model = model_generator.model
            self._time_bins = model_generator.data.time_bins
            self._saa_mask = model_generator.saa_calc.saa_mask

        self._ppc_model = None
        self._ppc_time_bins = None

    def save_data(self, file_path, result_dir, save_ppc=True):
        """
        Function to save the data needed to create the plots.
        """
        # Calculate the PPC
        ppc_counts, ppc_counts_binned = self._ppc_data(result_dir)

        if rank == 0:
            print("Save fit result to: {}".format(file_path))

            # Get the model counts
            model_counts = self._model.get_counts(time_bins=self._time_bins)

            # Get the counts of the individual sources
            source_list = self.get_counts_of_sources()

            # Get the statistical error from the posterior samples
            low = np.percentile(ppc_counts, 50 - 50 * 0.68, axis=0)
            high = np.percentile(ppc_counts, 50 + 50 * 0.68, axis=0)
            stat_err = high - low

            with h5py.File(file_path, "w") as f:

                f.attrs["dates"] = self._data.dates

                if hasattr(self._data, "trigger"):
                    f.attrs["trigger"] = self._data.trigger
                    f.attrs["trigger_time"] = self._data.trigtime

                f.attrs["data_type"] = self._data.data_type

                f.attrs["detectors"] = self._data.detectors
                f.attrs["echans"] = self._data.echans
                f.attrs["param_names"] = self._model.parameter_names
                f.attrs["best_fit_values"] = self._best_fit_values

                f.create_dataset("day_start_times", data=self._data.day_start_times)
                f.create_dataset("day_stop_times", data=self._data.day_stop_times)
                f.create_dataset(
                    "saa_mask", data=self._saa_mask, compression="lzf",
                )
                f.create_dataset(
                    "time_bins_start", data=self._time_bins[:, 0], compression="lzf",
                )
                f.create_dataset(
                    "time_bins_stop", data=self._time_bins[:, 1], compression="lzf",
                )
                f.create_dataset(
                    "observed_counts", data=self._data.counts, compression="lzf",
                )

                f.create_dataset(
                    "model_counts", data=model_counts, compression="lzf",
                )
                f.create_dataset("stat_err", data=stat_err, compression="lzf")

                group_sources = f.create_group("sources")
                for source in source_list:
                    group_sources.create_dataset(
                        source["label"], data=source["data"], compression="lzf",
                    )

                if save_ppc:
                    f.create_dataset(
                        "ppc_time_bins", data=self._ppc_time_bins, compression="lzf"
                    )
                    f.create_dataset(
                        "ppc_counts", data=ppc_counts_binned, compression="lzf"
                    )

            print("File sucessfully saved!")

    def get_counts_of_sources(self):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """

        source_list = []
        i_index = 0

        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(
                i, self._data.time_bins, self._saa_mask
            )
            if np.sum(data) != 0:
                source_list.append({"label": source_name, "data": data})
                i_index += 1

        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(
                i, self._data.time_bins, self._saa_mask
            )
            source_list.append({"label": source_name, "data": data})
            i_index += 1

        for i, source_name in enumerate(self._model.fit_spectrum_sources):
            data = self._model.get_fit_spectrum_counts(
                i, self._data.time_bins, self._saa_mask
            )
            source_list.append({"label": source_name, "data": data})
            i_index += 1

        saa_data = self._model.get_saa_counts(self._data.time_bins, self._saa_mask)
        if np.sum(saa_data) != 0:
            source_list.append({"label": "SAA_decays", "data": saa_data})
            i_index += 1

        # point_source_data = self._model.get_point_source_counts(self._data.time_bins, self._saa_mask)
        # if np.sum(point_source_data) != 0:
        #     source_list.append(
        #         {"label": "Point_sources", "data": point_source_data, "color": color_list[i_index]})
        #     i_index += 1

        return source_list

    def _ppc_data(self, result_dir):
        """
        Add ppc plot
        :param result_dir: path to result directory
        :param echan: Which echan
        """

        data_rebinner = Rebinner(self._time_bins, min_bin_width=60, mask=self._saa_mask)

        self._ppc_time_bins = data_rebinner.time_rebinned

        import pymultinest

        analyzer = pymultinest.analyse.Analyzer(1, result_dir)
        mn_posteriour_samples = analyzer.get_equal_weighted_posterior()[:, :-1]

        counts = []
        counts_binned = []

        # Make a mask with 500 random True to choose N_samples random samples,
        # if multinest returns less then 500 posterior samples the use the maximal possible number
        N_samples = (
            500 if len(mn_posteriour_samples) > 500 else len(mn_posteriour_samples)
        )

        random_mask = np.zeros(len(mn_posteriour_samples), dtype=int)
        random_mask[:N_samples] = 1
        np.random.shuffle(random_mask)
        random_mask = random_mask.astype(bool)

        # For these N_samples random samples calculate the corresponding rates for all time bins
        # with the parameters of this sample
        if using_mpi:

            points_per_rank = float(N_samples) / float(size)
            points_lower_index = int(np.floor(points_per_rank * rank))
            points_upper_index = int(np.floor(points_per_rank * (rank + 1)))

            hidden = False if rank == 0 else True

            with progress_bar(
                len(
                    mn_posteriour_samples[random_mask][
                        points_lower_index:points_upper_index
                    ]
                ),
                title="Calculating PPC",
                hidden=hidden,
            ) as p:

                for i, sample in enumerate(
                    mn_posteriour_samples[random_mask][
                        points_lower_index:points_upper_index
                    ]
                ):
                    synth_counts = self.get_synthetic_data(sample).astype(np.uint32)
                    counts.append(synth_counts)

                    rebinned_counts = data_rebinner.rebin(synth_counts)[0]

                    counts_binned.append(rebinned_counts)

                    p.increase()

            # Now combine and brodcast the results in small packages,
            # because mpi can't handle arrays that big
            ppc_counts = None
            ppc_counts_binned = None

            with progress_bar(len(counts), title="Gather PCC", hidden=hidden) as p:
                for i, cnts in enumerate(counts):

                    counts_g = comm.gather(cnts, root=0)

                    counts_binned_g = comm.gather(counts_binned[i], root=0)

                    if rank == 0:

                        counts_g = np.array(counts_g)

                        counts_binned_g = np.array(counts_binned_g)

                        if ppc_counts is None:

                            ppc_counts = counts_g

                            ppc_counts_binned = counts_binned_g

                        else:

                            ppc_counts = np.append(ppc_counts, counts_g, axis=0)

                            ppc_counts_binned = np.append(
                                ppc_counts_binned, counts_binned_g, axis=0
                            )

                    p.increase()

        else:
            with progress_bar(
                len(mn_posteriour_samples[random_mask]), title="Calculation PPC",
            ) as p:

                for i, sample in enumerate(mn_posteriour_samples[random_mask]):
                    synth_counts = self.get_synthetic_data(sample)
                    counts.append(synth_counts)

                    rebinned_counts = data_rebinner.rebin(synth_counts)[0]

                    counts_binned.append(rebinned_counts)

                    p.increase()

            ppc_counts = np.array(counts)
            ppc_counts_binned = np.array(counts_binned)

        return ppc_counts, ppc_counts_binned

    def get_synthetic_data(self, synth_parameters):
        """
        Creates a ContinousData object with synthetic data based on the total counts from the synth_model
        If no synth_model is passed it makes a deepcopy of the existing model
        :param synth_parameters:
        :return:
        """
        if self._ppc_model is None:

            self._ppc_model = copy.deepcopy(self._model)

        for i, parameter in enumerate(self._ppc_model.free_parameters.values()):
            parameter.value = synth_parameters[i]

        synth_counts = np.random.poisson(
            self._ppc_model.get_counts(self._data.time_bins)
        )

        return synth_counts


class PHAExporter(DataExporter):
    def __init__(self, *args, **kwargs):
        super(PHAExporter, self).__init__(*args, **kwargs)

    def save_pha(self, path, result_dir):
        model_counts = np.zeros(
            (len(self._total_time_bin_widths), len(self._echan_names))
        )
        stat_err = np.zeros_like(model_counts)

        # Get the model counts
        for echan in self._echan_names:
            model_counts[:, echan] = self._model.get_counts(
                self._total_time_bins, echan, saa_mask=self._saa_mask
            )
        model_rates = model_counts / self._total_time_bin_widths

        # Get the statistical error from the posterior samples
        for echan in self._echan_names:
            counts = self._ppc_data(result_dir, echan)[:, 2:-2]
            rates = counts / self._total_time_bin_widths

            low = np.percentile(rates, 50 - 50 * 0.68, axis=0)[0]
            high = np.percentile(rates, 50 + 50 * 0.68, axis=0)[0]

            stat_err[:, echan] = high - low

        spectrum = SPECTRUM(
            tstart=self._total_time_bins[:, 1],
            telapse=self._total_time_bin_widths,
            channel=self._echan_names,
            rate=model_rates,
            quality=np.zeros_like(model_rates, dtype=int),
            grouping=np.ones_like(self._echan_names),
            exposure=self._total_time_bin_widths,
            backscale=None,
            respfile=None,
            ancrfile=None,
            back_file=None,
            sys_err=None,
            stat_err=stat_err,
            is_poisson=False,
        )

        spectrum.hdu.dump(path)


det_name_lookup = {
    "n0": "NAI_00",
    "n1": "NAI_01",
    "n2": "NAI_02",
    "n3": "NAI_03",
    "n4": "NAI_04",
    "n5": "NAI_05",
    "n6": "NAI_06",
    "n7": "NAI_07",
    "n8": "NAI_08",
    "n9": "NAI_09",
    "na": "NAI_10",
    "nb": "NAI_11",
    "b0": "BGO_00",
    "b1": "BGO_01",
}

valid_det_names = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]


class PHAWriter(object):
    """
    Class to load multiple result files for the same date or the same trigger
    and combine them in a PHAII background file.
    """

    def __init__(
        self,
        dates,
        trigger,
        trigger_time,
        data_type,
        echans,
        detectors,
        time_bins,
        observed_counts,
        model_counts,
        stat_err,
        det_echan_loaded=None
    ):
        self._dates = dates
        self._trigger = trigger
        self._trigger_time = trigger_time
        self._data_type = data_type
        self._echans = echans
        self._detectors = detectors
        self._time_bins = time_bins
        self._observed_counts = observed_counts
        self._model_counts = model_counts
        self._stat_err = stat_err
        self._det_echan_loaded = det_echan_loaded

    @classmethod
    def from_result_files(cls, result_file_list):
        detectors = []
        echans = []
        det_echan_loaded = []

        for i, file in enumerate(result_file_list):
            with h5py.File(file, "r") as f:
                dates_f = f.attrs["dates"]
                data_type_f = f.attrs["data_type"]
                trigger_f = f.attrs.get("trigger", None)
                trigger_time_f = f.attrs.get("trigger_time", 0.0)
                detectors_f = f.attrs["detectors"]
                echans_f = f.attrs["echans"]
                time_bins_start_f = f["time_bins_start"][()]
                time_bins_stop_f = f["time_bins_stop"][()]

                observed_counts_f = f["observed_counts"][()]
                model_counts_f = f["model_counts"][()]
                stat_err_f = f["stat_err"][()]

            if i == 0:
                dates = dates_f
                data_type = data_type_f
                trigger = trigger_f
                trigger_time = trigger_time_f
                time_bins = np.vstack((time_bins_start_f, time_bins_stop_f)).T
                observed_counts = np.zeros((len(time_bins_stop_f), 12, 8))
                model_counts = np.zeros((len(time_bins_stop_f), 12, 8))
                stat_err = np.zeros_like(model_counts)

            else:
                assert dates == dates_f
                assert data_type == data_type_f
                assert trigger == trigger_f
                assert trigger_time == trigger_time_f
                assert np.array_equal(
                    time_bins, np.vstack((time_bins_start_f, time_bins_stop_f)).T,
                )

            for det_tmp_idx, det in enumerate(detectors_f):

                if det not in detectors:
                    detectors.append(det)

                det_idx = valid_det_names.index(det)

                for echan_idx, echan in enumerate(echans_f):

                    det_echan = "{}_{}".format(det, echan)

                    if echan not in echans:
                        echans.append(echan)

                    assert (
                        det_echan not in det_echan_loaded
                    ), "{}-{} already loaded, you have to resolve the conflict by hand".format(
                        det, echan
                    )

                    # Combine the observed counts
                    observed_counts[:, det_idx, echan] = observed_counts_f[
                        :, det_tmp_idx, echan_idx
                    ]

                    # Combine the model counts
                    model_counts[:, det_idx, echan] = model_counts_f[
                        :, det_tmp_idx, echan_idx
                    ]

                    # Combine the statistical error of the fit
                    stat_err[:, det_idx, echan] = stat_err_f[:, det_tmp_idx, echan_idx]

                    # Append the det_echan touple to avoid overloading
                    det_echan_loaded.append(det_echan)

        echans.sort()
        detectors.sort()
        det_echan_loaded.sort()

        return cls(
            dates,
            trigger,
            trigger_time,
            data_type,
            echans,
            detectors,
            time_bins,
            observed_counts,
            model_counts,
            stat_err,
            det_echan_loaded
        )

    @classmethod
    def from_combined_hdf5(cls, file_path):

        with h5py.File(file_path, "r") as f:

            dates = f.attrs["dates"]

            trigger = f.attrs["trigger"]

            trigger_time = f.attrs["trigger_time"]

            data_type = f.attrs["data_type"]

            echans = f.attrs["echans"]

            detectors = f.attrs["detectors"]

            time_bins = f["time_bins"][()]

            observed_counts = f["observed_counts"][()]

            model_counts = f["model_counts"][()]

            stat_err = f["stat_err"][()]

        if trigger == "None":
            trigger = None

        return cls(
            dates,
            trigger,
            trigger_time,
            data_type,
            echans,
            detectors,
            time_bins,
            observed_counts,
            model_counts,
            stat_err,
        )

    def save_combined_hdf5(self, output_path):

        with h5py.File(output_path, "w") as f:

            f.attrs["dates"] = self._dates

            f.attrs["trigger"] = self._trigger if self._trigger is not None else "None"

            f.attrs["trigger_time"] = self._trigger_time

            f.attrs["data_type"] = self._data_type

            f.attrs["echans"] = self._echans

            f.attrs["detectors"] = self._detectors

            f.create_dataset(
                "time_bins", data=self._time_bins, compression="lzf",
            )

            f.create_dataset(
                "observed_counts", data=self._observed_counts, compression="lzf",
            )

            f.create_dataset(
                "model_counts", data=self._model_counts, compression="lzf",
            )

            f.create_dataset("stat_err", data=self._stat_err, compression="lzf")

    def write_pha(
            self, output_dir, active_time_start, active_time_end, trigger_time=None, file_name=None, overwrite=False
    ):
        """
        Creates saves a background file for each detector
        """

        if trigger_time is not None:
            trigtime = trigger_time
        else:
            trigtime = self._trigger_time

        time_bins = self._time_bins - trigtime

        idx_min_time = time_bins[:, 0] >= active_time_start
        idx_max_time = time_bins[:, 1] <= active_time_end
        idx_valid_bin = idx_min_time * idx_max_time

        for det in self._detectors:
            det_idx = valid_det_names.index(det)

            tstart = time_bins[idx_valid_bin][0, 0]

            telapse = (
                time_bins[idx_valid_bin][-1, 1] - time_bins[idx_valid_bin][0, 0],
            )

            observed_counts = np.sum(
                self._observed_counts[idx_valid_bin, det_idx, :], axis=0
            )

            observed_rate = observed_counts / telapse

            model_counts = np.sum(self._model_counts[idx_valid_bin, det_idx, :], axis=0)

            model_rate = model_counts / telapse

            stat_err = np.sqrt(
                np.sum(np.square(self._stat_err[idx_valid_bin, det_idx, :]), axis=0)
            )

            # Calculate the dead time of the detector:
            # Each event in the echans 0-6 gives a dead time of 2.6 μs
            # Each event in the over flow channel 7 gives a dead time of 10 μs
            dead_time = (
                np.sum(observed_counts[0:7]) * 2.6 * 1e-6
                + observed_counts[7] * 1e-5
            )

            # Write observed spectrum to PHA file
            observed_spectrum = PHAII(
                instrument_name="GBM_{}".format(det_name_lookup[det]),
                telescope_name="Fermi",
                tstart=tstart,
                telapse=telapse,
                channel=self._echans,
                rate=observed_rate,
                quality=np.zeros_like(observed_rate, dtype=int),
                grouping=np.ones_like(self._echans),
                exposure=telapse - dead_time,
                backscale=1.,
                respfile=None,
                ancrfile=None,
                back_file=None,
                sys_err=np.zeros_like(observed_rate),
                stat_err=None,
                is_poisson=True,
            )

            # Write background spectrum to PHA file
            background_spectrum = PHAII(
                instrument_name="GBM_{}".format(det_name_lookup[det]),
                telescope_name="Fermi",
                tstart=tstart,
                telapse=telapse,
                channel=self._echans,
                rate=model_rate,
                quality=np.zeros_like(model_rate, dtype=int),
                grouping=np.ones_like(self._echans),
                exposure=telapse - dead_time,
                backscale=1.,
                respfile=None,
                ancrfile=None,
                back_file=None,
                sys_err=np.zeros_like(model_rate),
                stat_err=stat_err,
                is_poisson=False,
            )

            if file_name is None:

                if self._trigger is None:

                    file_name = "_".join(self._dates)

                else:

                    file_name = self._trigger

            obs_file_path = os.path.join(
                output_dir,
                "{}_{}.pha".format(file_name, det)
            )
            bkg_file_path = os.path.join(
                output_dir,
                "{}_{}_bak.pha".format(file_name, det)
            )

            observed_spectrum.writeto(obs_file_path, overwrite=overwrite)
            background_spectrum.writeto(bkg_file_path, overwrite=overwrite)

import numpy as np
from scipy.interpolate import interp1d
import arviz as av
import matplotlib.pyplot as plt


class StanDataConstructor(object):
    """
    Object to construct the data dictionary for stan!
    """

    def __init__(
        self,
        data=None,
        model=None,
        response=None,
        geometry=None,
        model_generator=None,
        threads_per_chain=1,
    ):
        """
        Init with data, model, response and geometry object or model_generator object
        """

        if model_generator is None:
            self._data = data
            self._model = model
            self._response = response
            self._geometry = geometry
        else:
            self._data = model_generator.data
            self._model = model_generator.model
            self._response = model_generator.response
            self._geometry = model_generator.geometry

        self._threads = threads_per_chain

        self._dets = self._data.detectors
        self._echans = self._data.echans
        self._time_bins = self._data.time_bins

        self._time_bin_edges = np.append(self._time_bins[:, 0], self._time_bins[-1:1])

        self._ndets = len(self._dets)
        self._nechans = len(self._echans)
        self._ntimebins = len(self._time_bins)

    def global_sources(self):
        """
        Fixed photon sources (e.g. point sources or CGB/Earth if spectrum not fitted)
        """

        s = self._model.global_sources

        if len(s) == 0:
            self._global_counts = None
            return None

        global_counts = np.zeros((len(s), self._ntimebins, self._ndets, self._nechans))

        for i, k in enumerate(s.keys()):
            global_counts[i] = s[k].get_counts(self._time_bins)

        # Flatten along time, detectors and echans
        global_counts = global_counts[:, 2:-2].reshape(len(s), -1)

        self._global_counts = global_counts

    def continuum_sources(self):
        """
        Sources with an independent norm per echan and detector (Cosmic rays).
        At the moment hard coded for 2 sources (Constant and CosmicRay)
        """

        if len(self._model.continuum_sources) == 0:
            self._cont_counts = None
            return None

        # In the python code we have an individual source for every echan. For the stan code we need one in total.
        num_cont_sources = 2

        continuum_counts = np.zeros(
            (num_cont_sources, self._ntimebins, self._ndets, self._nechans)
        )

        for i, s in enumerate(list(self._model.continuum_sources.values())):
            if "Constant" in s.name:
                index = 0
            else:
                index = 1
            continuum_counts[index, :, :, s.echan] = s.get_counts(self._time_bins)

        self._cont_counts = continuum_counts[:, 2:-2].reshape(2, -1)

    def free_spectrum_sources(self):
        """
        Free spectrum sources
        """

        s = self._model.fit_spectrum_sources

        self._Ebins_in = np.vstack(
            (
                self._response.responses[self._dets[0]].Ebin_in_edge[:-1],
                self._response.responses[self._dets[0]].Ebin_in_edge[1:],
            )
        )

        self._num_Ebins_in = len(self._Ebins_in[0])

        self._base_response_array_earth = None
        self._base_response_array_cgb = None
        self._base_response_array_ps = None

        if len(s) == 0:
            return None

        base_response_array_earth = None
        base_response_array_cgb = None
        base_rsp_ps_free = None

        for k in s.keys():
            rsp_detectors = s[k]._shape._effective_responses
            ar = np.zeros(
                (
                    self._ndets,
                    len(self._geometry.geometry_times),
                    self._num_Ebins_in,
                    self._nechans,
                )
            )
            for i, det in enumerate(self._dets):
                ar[i] = rsp_detectors[det]
            if k == "Earth occultation":
                base_response_array_earth = ar
            if k == "CGB":
                base_response_array_cgb = ar
            else:
                if base_rsp_ps_free is not None:
                    base_rsp_ps_free = np.append(base_rsp_ps_free, np.array([ar]))
                else:
                    base_rsp_ps_free = np.array([ar])

        if base_response_array_earth is not None:

            eff_rsp_new_earth = interp1d(
                self._geometry.geometry_times, base_response_array_earth, axis=1
            )

            rsp_all_earth = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_earth(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                0,
                1,
            )

            # Trapz integrate over time bins
            base_response_array_earth = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_earth[:, :-1] + rsp_all_earth[:, 1:])
            )

            self._base_response_array_earth = base_response_array_earth[2:-2].reshape(
                -1, self._num_Ebins_in
            )

        if base_response_array_cgb is not None:

            eff_rsp_new_cgb = interp1d(
                self._geometry.geometry_times, base_response_array_cgb, axis=1
            )

            rsp_all_cgb = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_cgb(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                0,
                1,
            )

            # Trapz integrate over time bins
            base_response_array_cgb = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_cgb[:-1] + rsp_all_cgb[1:])
            )

            self._base_response_array_cgb = base_response_array_cgb[2:-2].reshape(
                -1, self._num_Ebins_in
            )

        if base_rsp_ps_free is not None:

            eff_rsp_new_free_ps = interp1d(
                self._geometry.geometry_times, base_rsp_ps_free, axis=2
            )

            rsp_all_ps = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_free_ps(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                1,
                2,
            )

            # Trapz integrate over time bins
            base_rsp_ps_free = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_ps[:, :-1] + rsp_all_ps[:, 1:])
            )

            self._base_response_array_ps = base_rsp_ps_free[:, 2:-2].reshape(
                base_rsp_ps_free.shape[0], -1, self._num_Ebins_in
            )

    def saa_sources(self):
        """
        The Saa exit sources
        """
        # One source per exit (not per exit and echan like in the python code)
        self._num_saa_exits = int(len(self._model.saa_sources) / self._nechans)
        saa_start_times = np.zeros(self._num_saa_exits)
        for i, s in enumerate(
            list(self._model.saa_sources.values())[: self._num_saa_exits]
        ):
            saa_start_times[i] = s._shape._saa_exit_time[0]

        self._saa_start_times = saa_start_times

    def construct_data_dict(self):
        self.global_sources()
        self.continuum_sources()
        self.saa_sources()
        self.free_spectrum_sources()

        data_dict = {}

        data_dict["num_time_bins"] = self._ntimebins - 4
        data_dict["num_dets"] = self._ndets
        data_dict["num_echans"] = self._nechans

        data_dict["time_bins"] = self._time_bins[2:-2]
        data_dict["counts"] = np.array(self._data.counts[2:-2], dtype=int).flatten()

        data_dict["rsp_num_Ein"] = self._num_Ebins_in
        data_dict["Ebins_in"] = self._Ebins_in

        # Global sources
        if self._global_counts is not None:
            data_dict["num_fixed_comp"] = len(self._global_counts)
            data_dict["base_counts_array"] = self._global_counts
        else:
            raise NotImplementedError

        if self._base_response_array_ps is not None:
            data_dict["base_response_array_free_ps"] = self._base_response_array_ps
        if self._base_response_array_earth is not None:
            data_dict["base_response_array_earth"] = self._base_response_array_earth
        if self._base_response_array_cgb is not None:
            data_dict["base_response_array_cgb"] = self._base_response_array_cgb

        if self._base_response_array_cgb is not None:
            data_dict["earth_cgb_free"] = 1
        else:
            data_dict["earth_gb_free"] = 0

        data_dict["num_saa_exits"] = self._num_saa_exits
        data_dict["saa_start_times"] = self._saa_start_times

        if self._cont_counts is not None:
            data_dict["num_cont_comp"] = 2
            data_dict["base_counts_array_cont"] = self._cont_counts

        # Stan grainsize for reduced_sum
        if self._threads == 1:
            data_dict["grainsize"] = 1
        else:
            data_dict["grainsize"] = int(
                (self._ntimebins - 4) * self._ndets * self._nechans / self._threads
            )
        return data_dict


class ReadStanArvizResult(object):
    def __init__(self, nc_files):
        for i, nc_file in enumerate(nc_files):
            if i == 0:
                self._arviz_result = av.from_netcdf(nc_file)
            else:
                self._arviz_result = av.concat(
                    self._arviz_result, av.from_netcdf(nc_file), dim="chain"
                )

        self._model_parts = self._arviz_result.predictions.keys()

        self._dets = self._arviz_result.constant_data["dets"].values
        self._echans = self._arviz_result.constant_data["echans"].values

        self._ndets = len(self._dets)
        self._nechans = len(self._echans)

        self._time_bins = self._arviz_result.constant_data["time_bins"].values
        self._time_bins -= self._time_bins[0, 0]
        self._bin_width = self._time_bins[:, 1] - self._time_bins[:, 0]

        self._counts = self._arviz_result.observed_data["counts"].values

        predictions = self._arviz_result.predictions.stack(sample=("chain", "draw"))
        self._parts = {}
        for key in self._model_parts:
            self._parts[key] = predictions[key].values

        self._ppc = self._arviz_result.posterior_predictive.stack(
            sample=("chain", "draw")
        )["ppc"].values

    def ppc_plots(self, save_dir):

        colors = {
            "f_fixed": "red",
            "f_ps": "red",
            "f_saa": "navy",
            "f_cont": "magenta",
            "f_earth": "purple",
            "f_cgb": "cyan",
        }

        for d_index, d in enumerate(self._dets):
            for e_index, e in enumerate(self._echans):

                mask = np.arange(len(self._counts), dtype=int)[
                    e_index + d_index * self._ndets :: self._ndets * self._nechans
                ]
                fig, ax = plt.subplots()

                for i in np.linspace(0, self._ppc.shape[1] - 1, 30, dtype=int):
                    if i == 0:
                        ax.scatter(
                            np.mean(self._time_bins, axis=1),
                            self._ppc[mask][:, i] / self._bin_width,
                            color="darkgreen",
                            alpha=0.025,
                            edgecolor="green",
                            facecolor="none",
                            lw=0.9,
                            s=2,
                            label="PPC",
                        )
                    else:
                        ax.scatter(
                            np.mean(self._time_bins, axis=1),
                            self._ppc[mask][:, i] / self._bin_width,
                            color="darkgreen",
                            alpha=0.025,
                            edgecolor="darkgreen",
                            facecolor="none",
                            lw=0.9,
                            s=2,
                        )

                    for key in self._parts.keys():
                        # Check if there are several sources in this class
                        if len(self._parts[key].shape) == 3:
                            for k in range(len(self._parts[key])):
                                if k == 0 and i == 0:
                                    ax.scatter(
                                        np.mean(self._time_bins, axis=1),
                                        self._parts[key][k][mask][:, i]
                                        / self._bin_width,
                                        alpha=0.025,
                                        edgecolor=colors[key],
                                        facecolor="none",
                                        lw=0.9,
                                        s=2,
                                        label=key,
                                    )
                                else:
                                    ax.scatter(
                                        np.mean(self._time_bins, axis=1),
                                        self._parts[key][k][mask][:, i]
                                        / self._bin_width,
                                        alpha=0.025,
                                        edgecolor=colors[key],
                                        facecolor="none",
                                        lw=0.9,
                                        s=2,
                                    )
                        else:
                            if i == 0:
                                ax.scatter(
                                    np.mean(self._time_bins, axis=1),
                                    self._parts[key][mask][:, i] / self._bin_width,
                                    alpha=0.025,
                                    edgecolor=colors[key],
                                    facecolor="none",
                                    lw=0.9,
                                    s=2,
                                    label=key,
                                )
                            else:
                                ax.scatter(
                                    np.mean(self._time_bins, axis=1),
                                    self._parts[key][mask][:, i] / self._bin_width,
                                    alpha=0.025,
                                    edgecolor=colors[key],
                                    facecolor="none",
                                    lw=0.9,
                                    s=2,
                                )

                ax.scatter(
                    np.mean(self._time_bins, axis=1),
                    self._counts[mask] / self._bin_width,
                    color="darkgreen",
                    alpha=0.25,
                    edgecolor="black",
                    facecolor="none",
                    lw=0.9,
                    s=2,
                    label="Data",
                )
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
                lgd = fig.legend(loc="center right")  # , bbox_to_anchor=(1, 0.5))
                for lh in lgd.legendHandles:
                    lh.set_alpha(1)
                t = fig.suptitle(f"Detector {d} - Echan {e}")
                fig.savefig(
                    f"ppc_result_det_{d}_echan_{e}.png"
                )  # , bbox_extra_artists=(lgd,t), dpi=450, bbox_inches='tight')

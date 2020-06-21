import yaml
import pandas as pd
import os

from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.data.external_prop import ExternalProps
from gbmbkgpy.data.trigger_data import TrigData
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.fitting.background_like import BackgroundLike
from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.geometry_calc import Geometry
from gbmbkgpy.utils.response_precalculation import Response_Precalculation
from gbmbkgpy.modeling.setup_sources import Setup
from gbmbkgpy.modeling.albedo_cgb import Albedo_CGB_fixed, Albedo_CGB_free
from gbmbkgpy.modeling.sun import Sun
from gbmbkgpy.io.package_data import get_path_of_external_data_dir

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


NO_REBIN = 1e-9


def print_progress(text):
    """
    Helper function that prints the input text only with rank 0
    """
    if rank == 0:
        print(text)


class BackgroundModelGenerator(object):
    def __init__(self):
        pass

    def from_config_file(self, config_yml):
        with open(config_yml) as f:
            config = yaml.load(f)

        self.from_config_dict(config)

    def from_config_dict(self, config, response=None, geometry=None):

        self._config = config

        self._instantiate_data_class(config)

        self._instantiate_saa(config)

        if config["general"].get("min_bin_width", NO_REBIN) > NO_REBIN:
            self._rebinn_data(config)

        self._instantiate_ext_properties(config)

        if response is None:

            self._precalc_repsonse(config)

        else:

            self._resp = response

        if geometry is None:

            self._precalc_geometry(config)

        else:

            self._geom = geometry

        self._mask_valid_time_bins()

        self._setup_sources(config)

        self._instantiate_model(config)

        self._build_parameter_bounds(config)

        self._instantiate_likelihood(config)

    def _instantiate_data_class(self, config):
        print_progress("Prepare data...")
        self._data = Data(
            dates=config["general"]["dates"],
            detectors=config["general"]["detectors"],
            data_type=config["general"]["data_type"],
            echans=config["general"]["echans"],
            simulation=config["general"].get("simulation", False),
        )
        print_progress("Done")

    def _instantiate_saa(self, config):
        # Build the SAA object
        if config["saa"]["time_after_saa"] is not None:
            print_progress(
                "Precalculate SAA times and SAA mask. {} seconds after every SAA exit are excluded from fit...".format(
                    config["saa"]["time_after_saa"]
                )
            )
        else:
            print_progress("Precalculate SAA times and SAA mask...")

        self._saa_calc = SAA_calc(
            data=self._data,
            time_after_SAA=config["saa"]["time_after_saa"],
            short_time_intervals=config["saa"]["short_time_intervals"],
            nr_decays=config["saa"]["nr_decays_per_exit"],
        )

        print_progress("Done")

    def _rebinn_data(self, config):
        self._data.rebinn_data(
            config["general"]["min_bin_width"], self._saa_calc.saa_mask
        )
        self._saa_calc.set_rebinned_saa_mask(self._data.rebinned_saa_mask)

    def _instantiate_ext_properties(self, config):
        # Create external properties object (for McIlwain L-parameter)
        print_progress("Download and prepare external properties...")

        self._ep = ExternalProps(
            dates=config["general"]["dates"],
            detectors=config["general"]["detectors"],
            bgo_cr_approximation=config["setup"]["bgo_cr_approximation"],
        )

        print_progress("Done")

    def _precalc_repsonse(self, config):
        # Create a Response precalculation object, that precalculates the responses on a spherical grid arount the detector.
        # These calculations use the full DRM's and thus include sat. scattering and partial loss of energy by the photons.
        print_progress(
            "Precalculate responses for {} points on sphere around detector...".format(
                config["response"]["Ngrid"]
            )
        )

        self._resp = Response_Precalculation(
            detectors=config["general"]["detectors"],
            dates=config["general"]["dates"],
            echans=config["general"]["echans"],
            Ngrid=config["response"]["Ngrid"],
            data_type=config["general"]["data_type"],
            simulation=config["general"].get("simulation", False),
        )

        print_progress("Done")

    def _precalc_geometry(self, config):
        print_progress(
            "Precalculate geometry for {} times during the day...".format(
                config["geometry"]["n_bins_to_calculate"]
            )
        )

        self._geom = Geometry(
            data=self._data,
            dates=config["general"]["dates"],
            n_bins_to_calculate_per_day=config["geometry"]["n_bins_to_calculate"],
        )

        print_progress("Done")

    def _mask_valid_time_bins(self):

        self._data.mask_invalid_bins(geometry_times=self._geom.geometry_times)

        self._saa_calc.mask_invalid_bins(
            valid_time_mask=self._data.valid_time_mask,
            valid_rebinned_time_mask=self._data.valid_rebinned_time_mask,
        )

    def _setup_sources(self, config):
        # Create all individual sources and add them to a list
        assert (config["setup"]["fix_earth"] and config["setup"]["fix_cgb"]) or (
            not config["setup"]["fix_earth"] and not config["setup"]["fix_cgb"]
        ), "At the moment albeod and cgb spectrum have to be either both fixed or both free"

        if config["setup"]["fix_earth"]:
            self._albedo_cgb_obj = Albedo_CGB_fixed(self._resp, self._geom)
        else:
            self._albedo_cgb_obj = Albedo_CGB_free(self._resp, self._geom)

        if config["setup"]["use_sun"]:
            self._sun_obj = Sun(self._resp, self._geom, config["general"]["echan_list"])
        else:
            self._sun_obj = None

        print_progress("Create Source list...")

        self._source_list = Setup(
            data=self._data,
            saa_object=self._saa_calc,
            ep=self._ep,
            geometry=self._geom,
            sun_object=self._sun_obj,
            echans=config["general"]["echans"],
            det_responses=self._resp,
            albedo_cgb_object=self._albedo_cgb_obj,
            use_saa=config["setup"]["use_saa"],
            use_constant=config["setup"]["use_constant"],
            use_cr=config["setup"]["use_cr"],
            use_earth=config["setup"]["use_earth"],
            use_cgb=config["setup"]["use_cgb"],
            point_source_list=config["setup"]["ps_list"],
            fix_earth=config["setup"]["fix_earth"],
            fix_cgb=config["setup"]["fix_cgb"],
            use_sun=config["setup"]["use_sun"],
            saa_decays_per_exit=config["saa"]["nr_decays_per_exit"],
            saa_decay_at_day_start=config["saa"]["decay_at_day_start"],
            saa_decay_per_detector=config["saa"]["decay_per_detector"],
            saa_decay_model=config["saa"].get("decay_model", "exponential"),
            bgo_cr_approximation=config["setup"]["bgo_cr_approximation"],
            use_numba=config["fit"].get("use_numba", False),
        )

        print_progress("Done")

    def _instantiate_model(self, config):
        print_progress("Build model with source_list...")
        self._model = Model(
            *self._source_list,
            echans=config["general"]["echans"],
            detectors=config["general"]["detectors"],
            use_eff_area_correction=config["setup"]["use_eff_area_correction"],
        )
        print_progress("Done")

    def _build_parameter_bounds(self, config):
        parameter_bounds = {}

        # Echan individual sources
        for e in config["general"]["echans"]:

            if config["setup"]["use_saa"]:

                # If fitting only one day add additional 'SAA' decay to account for leftover excitation
                if (
                    config["saa"]["decay_at_day_start"]
                    and len(config["general"]["dates"]) == 1
                ):
                    offset = config["saa"]["nr_decays"]
                else:
                    offset = 0

                for saa_nr in range(self._saa_calc.num_saa + offset):

                    if config["saa"]["decay_per_detector"]:

                        for det in self._data.detectors:

                            parameter_bounds[
                                "norm_saa-{}_det-{}_echan-{}".format(saa_nr, det, e)
                            ] = {
                                "bounds": config["bounds"]["saa_bound"]["norm"],
                                "gaussian_parameter": config["gaussian_bounds"][
                                    "saa_bound"
                                ]["norm"],
                            }
                            parameter_bounds[
                                "decay_saa-{}_det-{}_echan-{}".format(saa_nr, det, e)
                            ] = {
                                "bounds": config["bounds"]["saa_bound"]["decay"],
                                "gaussian_parameter": config["gaussian_bounds"][
                                    "saa_bound"
                                ]["decay"],
                            }

                    else:

                        parameter_bounds[
                            "norm_saa-{}_det-all_echan-{}".format(saa_nr, e)
                        ] = {
                            "bounds": config["bounds"]["saa_bound"]["norm"],
                            "gaussian_parameter": config["gaussian_bounds"][
                                "saa_bound"
                            ]["norm"],
                        }
                        parameter_bounds[
                            "decay_saa-{}_det-all_echan-{}".format(saa_nr, e)
                        ] = {
                            "bounds": config["bounds"]["saa_bound"]["decay"],
                            "gaussian_parameter": config["gaussian_bounds"][
                                "saa_bound"
                            ]["decay"],
                        }

            if config["setup"]["use_constant"]:
                parameter_bounds["constant_echan-{}".format(e)] = {
                    "bounds": config["bounds"]["cr_bound"]["const"],
                    "gaussian_parameter": config["gaussian_bounds"]["cr_bound"][
                        "const"
                    ],
                }

            if config["setup"]["use_cr"]:
                parameter_bounds["norm_magnetic_echan-{}".format(e)] = {
                    "bounds": config["bounds"]["cr_bound"]["norm"],
                    "gaussian_parameter": config["gaussian_bounds"]["cr_bound"]["norm"],
                }

        if config["setup"]["use_sun"]:
            parameter_bounds["sun_norm"] = {
                "bounds": config["bounds"]["sun_bound"]["norm"],
                "gaussian_parameter": config["gaussian_bounds"]["sun_bound"]["norm"],
            }
            parameter_bounds["sun_index"] = {
                "bounds": config["bounds"]["sun_bound"]["index"],
                "gaussian_parameter": config["gaussian_bounds"]["sun_bound"]["index"],
            }
        # Global sources for all echans

        # If PS spectrum is fixed only the normalization, otherwise C, index
        for i, ps in enumerate(config["setup"]["ps_list"]):
            if ps == "auto_swift":
                filepath = os.path.join(
                    get_path_of_external_data_dir(), ".tmp", "ps_auto_swift.dat"
                )
                # Read it as pandas
                ps_df_add = pd.read_table(filepath, names=["name", "ra", "dec"])
                exclude = [
                    entry.upper() for entry in config["setup"]["ps_list"][ps]["exclude"]
                ]
                for row in ps_df_add.itertuples():
                    if row[1].upper() not in exclude:
                        parameter_bounds[f"norm_point_source-{row[1]}_pl"] = {
                            "bounds": config["bounds"]["ps_fixed_bound"]["pl"]["norm"],
                            "gaussian_parameter": config["gaussian_bounds"][
                                "ps_fixed_bound"
                            ]["pl"]["norm"],
                        }
            else:
                if config["setup"]["ps_list"][ps]["fixed"]:
                    if ps[:4] != "list":
                        for entry in config["setup"]["ps_list"][ps]["spectrum"]:
                            parameter_bounds[f"norm_point_source-{ps}_{entry}"] = {
                                "bounds": config["bounds"]["ps_fixed_bound"][entry][
                                    "norm"
                                ],
                                "gaussian_parameter": config["gaussian_bounds"][
                                    "ps_fixed_bound"
                                ][entry]["norm"],
                            }
                    else:
                        ps_df_add = pd.read_table(
                            config["setup"]["ps_list"][ps]["path"],
                            names=["name", "ra", "dec"],
                        )
                        for row in ps_df_add.itertuples():
                            for entry in config["setup"]["ps_list"][ps]["spectrum"]:
                                parameter_bounds[
                                    f"norm_point_source-{row[1]}_{entry}"
                                ] = {
                                    "bounds": config["bounds"]["ps_fixed_bound"][entry][
                                        "norm"
                                    ],
                                    "gaussian_parameter": config["gaussian_bounds"][
                                        "ps_fixed_bound"
                                    ][entry]["norm"],
                                }

                else:
                    if ps[:4] != "list":
                        for entry in config["setup"]["ps_list"][ps]["spectrum"]:
                            if entry == "pl":
                                parameter_bounds[
                                    "ps_{}_spectrum_fitted_norm_pl".format(ps)
                                ] = {
                                    "bounds": config["bounds"]["ps_free_bound"][entry][
                                        "norm"
                                    ],
                                    "gaussian_parameter": config["gaussian_bounds"][
                                        "ps_free_bound"
                                    ][entry]["norm"],
                                }
                                parameter_bounds[
                                    "ps_{}_spectrum_fitted_index".format(ps)
                                ] = {
                                    "bounds": config["bounds"]["ps_free_bound"][entry][
                                        "index"
                                    ],
                                    "gaussian_parameter": config["gaussian_bounds"][
                                        "ps_free_bound"
                                    ][entry]["index"],
                                }
                            elif entry == "bb":

                                parameter_bounds[
                                    "ps_{}_spectrum_fitted_norm_bb".format(ps)
                                ] = {
                                    "bounds": config["bounds"]["ps_free_bound"][entry][
                                        "norm"
                                    ],
                                    "gaussian_parameter": config["gaussian_bounds"][
                                        "ps_free_bound"
                                    ][entry]["norm"],
                                }
                                parameter_bounds[
                                    "ps_{}_spectrum_fitted_temp".format(ps)
                                ] = {
                                    "bounds": config["bounds"]["ps_free_bound"][entry][
                                        "temp"
                                    ],
                                    "gaussian_parameter": config["gaussian_bounds"][
                                        "ps_free_bound"
                                    ][entry]["temp"],
                                }
                    else:
                        ps_df_add = pd.read_table(
                            config["setup"]["ps_list"][ps]["path"],
                            names=["name", "ra", "dec"],
                        )
                        for row in ps_df_add.itertuples():
                            for entry in config["setup"]["ps_list"][ps]["spectrum"]:
                                if entry == "pl":
                                    parameter_bounds[
                                        f"ps_{row[1]}_spectrum_fitted_norm_pl"
                                    ] = {
                                        "bounds": config["bounds"]["ps_free_bound"][
                                            entry
                                        ]["norm"],
                                        "gaussian_parameter": config["gaussian_bounds"][
                                            "ps_free_bound"
                                        ][entry]["norm"],
                                    }
                                    parameter_bounds[
                                        f"ps_{row[1]}_spectrum_fitted_index"
                                    ] = {
                                        "bounds": config["bounds"]["ps_free_bound"][
                                            entry
                                        ]["index"],
                                        "gaussian_parameter": config["gaussian_bounds"][
                                            "ps_free_bound"
                                        ][entry]["index"],
                                    }
                                elif entry == "bb":

                                    parameter_bounds[
                                        f"ps_{row[1]}_spectrum_fitted_norm_bb"
                                    ] = {
                                        "bounds": config["bounds"]["ps_free_bound"][
                                            entry
                                        ]["norm"],
                                        "gaussian_parameter": config["gaussian_bounds"][
                                            "ps_free_bound"
                                        ][entry]["norm"],
                                    }
                                    parameter_bounds[
                                        f"ps_{row[1]}_spectrum_fitted_temp"
                                    ] = {
                                        "bounds": config["bounds"]["ps_free_bound"][
                                            entry
                                        ]["temp"],
                                        "gaussian_parameter": config["gaussian_bounds"][
                                            "ps_free_bound"
                                        ][entry]["temp"],
                                    }

        if config["setup"]["use_earth"]:
            # If earth spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
            if config["setup"]["fix_earth"]:
                parameter_bounds["norm_earth_albedo"] = {
                    "bounds": config["bounds"]["earth_fixed_bound"]["norm"],
                    "gaussian_parameter": config["gaussian_bounds"][
                        "earth_fixed_bound"
                    ]["norm"],
                }
            else:
                parameter_bounds["earth_albedo_spectrum_fitted_norm"] = {
                    "bounds": config["bounds"]["earth_free_bound"]["norm"],
                    "gaussian_parameter": config["gaussian_bounds"]["earth_free_bound"][
                        "norm"
                    ],
                }
                parameter_bounds["earth_albedo_spectrum_fitted_index1"] = {
                    "bounds": config["bounds"]["earth_free_bound"]["alpha"],
                    "gaussian_parameter": config["gaussian_bounds"]["earth_free_bound"][
                        "alpha"
                    ],
                }
                parameter_bounds["earth_albedo_spectrum_fitted_index2"] = {
                    "bounds": config["bounds"]["earth_free_bound"]["beta"],
                    "gaussian_parameter": config["gaussian_bounds"]["earth_free_bound"][
                        "beta"
                    ],
                }
                parameter_bounds["earth_albedo_spectrum_fitted_break_energy"] = {
                    "bounds": config["bounds"]["earth_free_bound"]["Eb"],
                    "gaussian_parameter": config["gaussian_bounds"]["earth_free_bound"][
                        "Eb"
                    ],
                }

        if config["setup"]["use_cgb"]:
            # If cgb spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
            if config["setup"]["fix_cgb"]:
                parameter_bounds["norm_cgb"] = {
                    "bounds": config["bounds"]["cgb_fixed_bound"]["norm"],
                    "gaussian_parameter": config["gaussian_bounds"]["cgb_fixed_bound"][
                        "norm"
                    ],
                }
            else:
                parameter_bounds["CGB_spectrum_fitted_norm"] = {
                    "bounds": config["bounds"]["cgb_free_bound"]["norm"],
                    "gaussian_parameter": config["gaussian_bounds"]["cgb_free_bound"][
                        "norm"
                    ],
                }
                parameter_bounds["CGB_spectrum_fitted_index1"] = {
                    "bounds": config["bounds"]["cgb_free_bound"]["alpha"],
                    "gaussian_parameter": config["gaussian_bounds"]["cgb_free_bound"][
                        "alpha"
                    ],
                }
                parameter_bounds["CGB_spectrum_fitted_index2"] = {
                    "bounds": config["bounds"]["cgb_free_bound"]["beta"],
                    "gaussian_parameter": config["gaussian_bounds"]["cgb_free_bound"][
                        "beta"
                    ],
                }
                parameter_bounds["CGB_spectrum_fitted_break_energy"] = {
                    "bounds": config["bounds"]["cgb_free_bound"]["Eb"],
                    "gaussian_parameter": config["gaussian_bounds"]["cgb_free_bound"][
                        "Eb"
                    ],
                }

        if config["setup"]["use_eff_area_correction"]:
            for det in sorted(config["general"]["detectors"])[1:]:
                parameter_bounds[f"eff_area_corr_{det}"] = {
                    "bounds": config["bounds"]["eff_area_correction"],
                    "gaussian_parameter": config["gaussian_bounds"][
                        "eff_area_correction"
                    ],
                }

        self._parameter_bounds = parameter_bounds

        # Add bounds to the parameters for multinest
        self._model.set_parameter_bounds(self._parameter_bounds)

    def _instantiate_likelihood(self, config):
        # Class that calcualtes the likelihood
        print_progress("Create BackgroundLike class that conects model and data...")
        self._background_like = BackgroundLike(
            data=self._data,
            model=self._model,
            saa_object=self._saa_calc,
            use_numba=config["fit"].get("use_numba", False),
        )
        print_progress("Done")

    @property
    def data(self):
        return self._data

    @property
    def external_properties(self):
        return self._ep

    @property
    def saa_calc(self):
        return self._saa_calc

    @property
    def response(self):
        return self._resp

    @property
    def geometry(self):
        return self._geom

    @property
    def albedo_cgb(self):
        return self._albedo_cgb_obj

    @property
    def source_list(self):
        return self._source_list

    @property
    def model(self):
        return self._model

    @property
    def likelihood(self):
        return self._background_like

    @property
    def parameter_bounds(self):
        return self._parameter_bounds

    @property
    def config(self):
        return self._config


class TrigdatBackgroundModelGenerator(BackgroundModelGenerator):
    def _instantiate_data_class(self, config):
        print_progress("Prepare data...")
        self._data = TrigData(
            trigger=config["general"]["trigger"],
            detectors=config["general"]["detectors"],
            data_type=config["general"]["data_type"],
            echans=config["general"]["echans"],
            test=config["general"].get("test", False),
        )
        print_progress("Done")

    def _precalc_repsonse(self, config):
        # Create a Response precalculation object, that precalculates the responses on a spherical grid arount the detector.
        # These calculations use the full DRM's and thus include sat. scattering and partial loss of energy by the photons.
        print_progress(
            "Precalculate responses for {} points on sphere around detector...".format(
                config["response"]["Ngrid"]
            )
        )

        self._resp = Response_Precalculation(
            detectors=config["general"]["detectors"],
            dates=config["general"]["dates"],
            echans=config["general"]["echans"],
            Ngrid=config["response"]["Ngrid"],
            data_type=config["general"]["data_type"],
            trigger=config["general"]["trigger"],
        )

        print_progress("Done")

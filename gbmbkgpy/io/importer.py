import numpy as np
import copy

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.pha import SPECTRUM
import h5py
from gbmbkgpy.utils.progress_bar import progress_bar

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
        rank = 0
except:
    using_mpi = False
    rank = 0


class FitImporter(object):
    """
    This will be used to import a saved background fit for further analysis or trigger detection
    """

    def __init__(self, config, fit_result_hdf5):
        self._instantiate_model(config)
        self._load_result_file(fit_result_hdf5)

    def _load_result_file(self, fit_result_hdf5):
        print("Load result file")

        if rank == 0:
            with h5py.File(fit_result_hdf5, "r") as f:
                detectors = f.attrs["detectors"]
                dates = f.attrs["dates"]
                param_names = f.attrs["param_names"]
                best_fit_values = f.attrs["best_fit_values"]
                stat_err = f["stat_err"][()]

            assert np.array_equal(
                detectors, self.data.detectors
            ), "Detector in fit result file is inconsistent with detector in config.yml"
            assert np.array_equal(
                dates, self.data.dates
            ), "Dates in fit result file is inconsistent with  dates in config.yml"
            assert np.array_equal(
                param_names, self.model.parameter_names
            ), "The parameters in the result files do not match the parameters of the model"
        else:
            best_fit_values = None

        if using_mpi:
            comm.Barrier()
            best_fit_values = comm.bcast(best_fit_values, root=0)

        self.likelihood.set_free_parameters(best_fit_values)
        self._best_fit_values = best_fit_values

        print("Successfully loaded result file!")

    def _instantiate_model(self, config):
        self._model_generator = BackgroundModelGenerator()

        if isinstance(config, dict):
            self._model_generator.from_config_dict(config)
        else:
            self._model_generator.from_config_file(config)

    @property
    def data(self):
        return self._model_generator.data

    @property
    def model(self):
        return self._model_generator.model

    @property
    def likelihood(self):
        return self._model_generator.likelihood

    @property
    def config(self):
        return self._model_generator.config

    @property
    def saa_calc(self):
        return self._model_generator.saa_calc

    @property
    def best_fit_values(self):
        return self._best_fit_values

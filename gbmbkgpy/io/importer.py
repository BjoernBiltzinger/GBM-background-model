import numpy as np
import copy

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.pha import SPECTRUM
import h5py
from gbmbkgpy.utils.progress_bar import progress_bar

NO_REBIN = 1E-99

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
    def __init__(self, config_yml, fit_result_hdf5):
        self._det = None
        self._dates = None
        self._day_start_times = None
        self._day_stop_times = None
        self._time_bins_start = None
        self._time_bins_stop = None
        self._saa_mask = None
        self._best_fit_values = None
        self._covariance_matrix = None
        self._param_names = None
        self._model_counts = None
        self._stat_err = None

        self._instantiate_model(config_yml)
        self._load_result_file(fit_result_hdf5)

    def _load_result_file(self, fit_result_hdf5):
        print('Load data and start plotting')

        if rank == 0:
            with h5py.File(fit_result_hdf5, 'r') as f:
                det =                 np.array(f['general']['detector'])
                dates =               np.array(f['general']['dates'])
                best_fit_values =     np.array(f['general']['best_fit_values'])
                param_names =         np.array(f['general']['param_names'])
                stat_err =            np.array(f['general']['stat_err'])

            assert det == self.data.det, 'Detector in fit result file is inconsistent with detector in config.yml'
            assert dates == self.data.dates, 'Dates in fit result file is inconsistent with  dates in config.yml'
            assert param_names == self.model.parameter_names, 'The parameters in the result files do not match the parameters of the model'

        if using_mpi:
            comm.Barrier()
            best_fit_values = comm.bcast(best_fit_values, root=0)

        self.likelihood.set_free_parameters(best_fit_values)

        print('Successfully loaded result file!')

    def _instantiate_model(self, config_yml):
        self._model_generator = BackgroundModelGenerator()

        self._model_generator.from_config_file(config_yml)

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

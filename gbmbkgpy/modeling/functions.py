from gbmbkgpy.modeling.parameter import Parameter
from gbmbkgpy.utils.spectrum import (
    _spec_integral_pl,
    _spec_integral_bb_pl,
    _spec_integral_bpl,
    _spec_integral_bb)
import numpy as np
import numexpr as ne

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

import collections

import scipy.integrate as integrate
import scipy.interpolate as interpolate

try:
    from numba import njit, float64, prange

    has_numba = True
except:
    has_numba = False

if has_numba:
    # Import Interpolation class
    from gbmbkgpy.utils.interpolation import Interp1D

    # Numba response folding
    @njit(cache=True)
    def _dot_numba(vec, A):
        """
        :param vec: True flux; shape(j)
        :param A: array of response arrays; shape (i,j,k)
        :return: dot product of vec and A
        """
        res = np.zeros((A.shape[0], A.shape[2]))
        for i in prange(A.shape[0]):
            res[i] = np.dot(vec, A[i])
        return res

    # Numba trapz integration
    @njit([float64[:, :, :](float64[:, :, :, :], float64[:, :])], cache=True)
    def _trapz_numba(y, x):
        """
        Trapz integration of matrix
        :param x: x-coords to integrate; trapz_int from x[i,0] to x[i,1]; shape (len_time_bins, 2)
        :param y: Folded differential counts [1/s] at the ebin edges; shape (len_time_bins, num_dets, num_echan, 2)
        :return: Counts in the time bins, for all dets and echans; shape (len_time_bins, num_dets, num_echan)
        """
        res = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
        for i in prange(len(y)):
            for j in prange(len(y[0])):
                for k in prange(len(y[0, 0])):
                    res[i, j, k] = (
                        (x[i, 1] - x[i, 0]) * (y[i, j, k, 1] + y[i, j, k, 0]) / 2
                    )
        return res

class Function(object):
    def __init__(self, *parameters, use_numba=False):
        """
        Init function of source
        :param parameters: parameters of the source
        """

        self._use_numba = use_numba
        if self._use_numba:
            assert has_numba, "Numba not installed!"

        parameter_dict = collections.OrderedDict()

        for parameter in parameters:
            parameter_dict[parameter.name] = parameter

        self._parameter_dict = parameter_dict

        for key, value in self._parameter_dict.items():
            self.__dict__[key] = value

    @property
    def parameter_value(self):
        """
        Returns the current parameter values
        :return:
        """

        return [par.value for par in self._parameter_dict.values()]

    def __call__(self):
        """
        Starts the evaluation of the counts per time bin with the current parameters
        :param echan: echan for which the counts should be returned
        :return:
        """

        return self._evaluate(*self.parameter_value)

    def recalculate_counts(self):
        """
        Function needed for sources that change spectrum during the fit. This recalculates the folding
        of the assumed photon spectrum (with the new spectral parameters) with the precalculated response
        :return:
        """

        self._fold_spectrum(*self.parameter_value)

    @property
    def parameters(self):
        """
        Returns the dictionary with the parameters
        :return:
        """

        return self._parameter_dict

    def _evaluate(self, *parameter_values):
        raise NotImplementedError("This method has to be defined in a subclass")

    def _fold_spectrum(self, *parameter_values):
        raise NotImplementedError("This method has to be defined in a subclass")


class Solar_Flare(Function):
    def __init__(self):
        K = Parameter(
            "norm_flare",
            initial_value=1.0,
            min_value=0,
            max_value=None,
            delta=0.1,
            normalization=True,
        )
        decay_constant = Parameter(
            "decay_flare", initial_value=-0.01, min_value=-1, max_value=0, delta=0.1
        )

        super(Solar_Flare, self).__init__(K, decay_constant)

    def _evaluate(self, x, K, decay_constant, echan=None):
        return K * np.exp(-x / decay_constant)


class SAA_Decay(Function):
    def __init__(self, saa_number, echan, model="exponential", detector="all", det_idx=None):
        A = Parameter(
            "norm_saa-{}_det-{}_echan-{}".format(saa_number, detector, echan),
            initial_value=1.0,
            min_value=0,
            max_value=None,
            delta=0.1,
            normalization=True,
            prior="log_uniform",
        )
        saa_decay_constant = Parameter(
            "decay_saa-{}_det-{}_echan-{}".format(saa_number, detector, echan),
            initial_value=0.01,
            min_value=0.0,
            max_value=1.0,
            delta=0.1,
            prior="log_uniform",
        )

        self._model = model

        self._det_idx = det_idx

        self._build_decay_function()

        super(SAA_Decay, self).__init__(A, saa_decay_constant)

    def set_saa_exit_time(self, time):
        self._saa_exit_time = time

    def set_time_bins(self, time_bins):
        self._time_bins = time_bins

    def set_nr_detectors(self, nr_detectors):
        self._nr_detectors = nr_detectors

    def set_det_idx(self, det_idx):
        self._det_idx = det_idx

    def precalulate_time_bins_integral(self):
        """
        This function is needed to do all the precalculations one can do for the later evaluation. One can precalulate
        the which time bins are after the SAA exit.
        :return:
        """

        self._t0 = self._saa_exit_time
        self._idx_start = self._time_bins[:, 0] < self._t0

        self._tstart = self._time_bins[:, 0][~self._idx_start]
        self._tstop = self._time_bins[:, 1][~self._idx_start]

        self._out = np.zeros_like(self._time_bins[:, 0])

    def _build_decay_function(self):

        if self._model == "exponential":

            def _decay_function(t0, tstart, tstop, A, saa_decay_constant):
                """
                Calculates the exponential decay for the SAA exit
                The the values are calculated for the start and stop times of the bins with the analytic solution of the integral
                for a function A*exp(-saa_decay_constant*(t-t0)) which is -A/saa_decay_constant *
                (exp(-saa_decay_constant*(tend_bin-to) - exp(-saa_decay_constant*(tstart_bin-to))
                :param A:
                :param saa_decay_constant:
                :return:
                """
                return ne.evaluate(
                    "-A / saa_decay_constant*(exp((t0-tstop)*abs(saa_decay_constant)) - exp((t0 - tstart)*abs(saa_decay_constant)))"
                )

        elif self._model == "linear":

            def _decay_function(t0, tstart, tstop, A, saa_decay_constant):
                """
                Calculates the linear decay for the SAA exit
                The the values are calculated for the start and stop times of the bins with the analytic solution for the integral
                of the function A - saa_decay_constany(t-t0)
                :param A:
                :param saa_decay_constant:
                :return:
                """
                return ne.evaluate(
                    "-A*tstart - saa_decay_constant*t0*tstart + (saa_decay_constant*tstart**2)/2 + A*tstop + saa_decay_constant*t0*tstop - (saa_decay_constant*tstop**2)/2"
                )

        else:
            raise NotImplementedError("The model is not implemented")

        self._decay_function = _decay_function


    def _evaluate(self, A, saa_decay_constant):
        """
        Calculates the exponential decay for the SAA exit
        The the values are calculated for the start and stop times of the bins with the analytic solution of the integral
        for a function A*exp(-saa_decay_constant*(t-t0)) which is -A/saa_decay_constant *
        (exp(-saa_decay_constant*(tend_bin-to) - exp(-saa_decay_constant*(tstart_bin-to))
        :param A:
        :param saa_decay_constant:
        :return:
        """
        t0 = self._t0
        tstart = self._tstart
        tstop = self._tstop

        self._out[~self._idx_start] = self._decay_function(t0, tstart, tstop, A, saa_decay_constant)

        if self._det_idx is None:

            return np.tile(self._out, (self._nr_detectors, 1)).T

        else:

            out_matrix = np.zeros((
                len(self._time_bins[:, 0]),
                self._nr_detectors
            ))

            out_matrix[:, self._det_idx] = self._out

            return out_matrix

class ContinuumFunction(Function):
    def __init__(self, coefficient_name):
        """
        A continuum function that is parametrized by a constant multiplied by
        an interpolated function. No echan connection. Every echan gets an independent
        normalization. Used for cosmic rays mostly.
        :param coefficient_name: the name of the coefficient
        """

        assert isinstance(coefficient_name, str)

        K = Parameter(
            coefficient_name,
            initial_value=1.0,
            min_value=0,
            max_value=None,
            delta=0.1,
            normalization=True,
        )

        super(ContinuumFunction, self).__init__(K)

    def set_function_array(self, function_array):
        """
        Set the temporal interpolation of the count rates that will be used for the function

        :param function_array: a scipy interpolation function
        :return:
        """

        self._function_array = function_array

    def set_saa_zero(self, saa_mask):
        """
        Set the SAA sections in the function array to zero
        :param saa_mask:
        :return:
        """
        self._function_array[~saa_mask] = 0.0

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """
        for det_idx in range(self._function_array.shape[1]):
            self._function_array[:, det_idx, :][
                self._function_array[:, det_idx, :] > 0
            ] = self._function_array[:, det_idx, :][
                self._function_array[:, det_idx, :] > 0
            ] - np.min(
                self._function_array[:, det_idx, :][
                    self._function_array[:, det_idx, :] > 0
                ]
            )

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """
        tiled_time_bins = np.tile(time_bins, (self._function_array.shape[1], 1, 1))

        tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 1)

        self._source_counts = np.trapz(self._function_array, tiled_time_bins)

    def _evaluate(self, K):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the current parameter value for K
        :return:
        """
        source_counts = self._source_counts
        return ne.evaluate("K*source_counts")

    def __call__(self):
        return self._evaluate(*self.parameter_value)


class GlobalFunction(Function):
    """
    A class in which a global constant can be generated which is the same for all Echans.
    Used for photon sources with fixed spectrum to predict the count rates in all echans simultaneously
    """

    def __init__(self, coefficient_name):
        """
        Init one Parameter K
        :param coefficient_name:
        """

        K = Parameter(
            coefficient_name,
            initial_value=1.0,
            min_value=0,
            max_value=None,
            delta=0.1,
            normalization=True,
        )

        super(GlobalFunction, self).__init__(K)

    def info(self):
        print("############### Class definition ###################")
        print("This is a global class with no spectral fitting")
        print("Sources that use this class have a physical photon spectrum")
        print("and we only fit the normalization of this photon spectrum")
        print("to the data. This spectrum gets folded with the response to")
        print("get the counts in the energy bins of GBM")
        print("############### Paramerters ########################")
        print("In this object these parameter are stored:")
        for key, value in self._parameter_dict.items():
            print(key)

    def set_function_array(self, function_array):
        """
        Set the temporal interpolation that will be used for the function
        Here the function_array is a list with as many entries as echans fitted together!
        :param function_array: a scipy interpolation function
        :return:
        """

        self._function_array = function_array

    def set_saa_zero(self, saa_mask):
        """
        Set the SAA sections in the function array to zero
        :param saa_mask:
        :return:
        """
        self._function_array[~saa_mask] = 0.0

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """
        for det_idx in range(self._function_array.shape[1]):
            self._function_array[:, det_idx, :, :][
                self._function_array[:, det_idx, :, :] > 0
            ] = self._function_array[:, det_idx, :, :][
                self._function_array[:, det_idx, :, :] > 0
            ] - np.min(
                self._function_array[:, det_idx, :, :][
                    self._function_array[:, det_idx, :, :] > 0
                ]
            )

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        tiled_time_bins = np.tile(
            time_bins,
            (self._function_array.shape[1], self._function_array.shape[2], 1, 1),
        )

        tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 2)
        tiled_time_bins = np.swapaxes(tiled_time_bins, 1, 2)

        self._source_counts = np.trapz(self._function_array, tiled_time_bins)

    def _evaluate(self, K):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the fitted parameter
        :return:
        """
        source_counts = self._source_counts
        return ne.evaluate("K*source_counts")

    def __call__(self):
        return self._evaluate(*self.parameter_value)


class GlobalFunctionSpectrumFit(Function):
    """
    A class in which a global constant and spectral parameters can be generated which is the same for all Echans.
    Use this if you want a source with free spectral parameters. Is computational much more expensive than the fixed
    spectrum!
    """

    def __init__(self, coefficient_name, spectrum="bpl", E_norm=1.0, use_numba=False):
        """
        Init the parameters of a broken power law
        :param coefficient_name:
        """
        self._E_norm = E_norm
        self._spec = spectrum
        if self._spec == "bpl":
            C = Parameter(
                coefficient_name + "_norm",
                initial_value=1.0,
                min_value=0,
                max_value=None,
                delta=0.1,
                normalization=True,
                prior="log_uniform",
            )
            index1 = Parameter(
                coefficient_name + "_index1",
                initial_value=-1.0,
                min_value=-10,
                max_value=5,
                mu=1,
                sigma=1,
                delta=0.1,
                normalization=False,
                prior="truncated_gaussian",
            )
            index2 = Parameter(
                coefficient_name + "_index2",
                initial_value=2.0,
                min_value=0.1,
                max_value=5,
                mu=1,
                sigma=1,
                delta=0.1,
                normalization=False,
                prior="truncated_gaussian",
            )
            break_energy = Parameter(
                coefficient_name + "_break_energy",
                initial_value=-1.0,
                min_value=-10,
                max_value=5,
                delta=0.1,
                normalization=False,
                prior="log_uniform",
            )

            super(GlobalFunctionSpectrumFit, self).__init__(
                C, index1, index2, break_energy, use_numba=use_numba
            )

        elif self._spec == "pl":

            C = Parameter(
                coefficient_name + "_norm_pl",
                initial_value=1.0,
                min_value=0,
                max_value=None,
                delta=0.1,
                normalization=True,
                prior="log_uniform",
            )
            index = Parameter(
                coefficient_name + "_index",
                initial_value=1.0,
                min_value=0,
                max_value=3,
                delta=0.1,
                mu=1,
                sigma=1,
                normalization=False,
                prior="truncated_gaussian",
            )

            super(GlobalFunctionSpectrumFit, self).__init__(
                C, index, use_numba=use_numba
            )

        elif self._spec == "bb+pl":

            C_pl = Parameter(
                coefficient_name + "_norm_pl",
                initial_value=1.0,
                min_value=0,
                max_value=None,
                delta=0.1,
                normalization=True,
                prior="log_uniform",
            )

            index = Parameter(coefficient_name + "_index",
                              initial_value=1.0,
                              min_value=0,
                              max_value=3,
                              delta=0.1,
                              mu=1,
                              sigma=1,
                              normalization=False,
                              prior="truncated_gaussian",
            )

            C_bb = Parameter(
                coefficient_name + "_norm_bb",
                initial_value=1.0,
                min_value=0,
                max_value=None,
                delta=0.1,
                normalization=True,
                prior="log_uniform",
            )

            temp = Parameter(
                coefficient_name + "_temp",
                initial_value=1.0,
                min_value=0,
                max_value=5,
                delta=0.1,
                mu=1,
                sigma=1,
                normalization=False,
                prior="truncated_gaussian",
            )

            super(GlobalFunctionSpectrumFit, self).__init__(
                C_pl, index, C_bb, temp, use_numba=use_numba
            )
        elif self._spec == 'bb':

            C = Parameter(
                coefficient_name + "_norm_bb",
                initial_value=1.0,
                min_value=0,
                max_value=None,
                delta=0.1,
                normalization=True,
                prior="log_uniform",
            )

            temp = Parameter(
                coefficient_name + "_temp",
                initial_value=1.0,
                min_value=0,
                max_value=5,
                delta=0.1,
                mu=1,
                sigma=1,
                normalization=False,
                prior="truncated_gaussian",
            )

            super(GlobalFunctionSpectrumFit, self).__init__(
                C, temp, use_numba=use_numba
                )
        else:

            raise ValueError(
                "Spectrum must be bpl, pl or bb+pl at the moment. But is {}".format(self._spec)
            )

    def set_dets_echans(self, detectors, echans):

        self._detectors = detectors
        self._echans = echans

    def set_effective_responses(self, effective_responses):
        """
        effective response sum for all times for which the geometry was calculated (NO INTERPOLATION HERE)
        :param response_array:
        :return:
        """
        self._effective_responses = effective_responses

    def set_interpolation_times(self, interpolation_times):
        """
        times for which the geometry was calculated
        :param interpolation_times:
        :return:
        """
        self._interpolation_times = interpolation_times
        try:
            self.set_interpolation()
        except:
            pass

    def set_time_bins(self, time_bins):
        """
        Basis array that has the length as the time_bins array with all entries 1
        :param time_bins:
        :return:
        """
        self._time_bins = time_bins

        tiled_time_bins = np.tile(
            time_bins, (len(self._detectors), len(self._echans), 1, 1)
        )

        tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 2)
        tiled_time_bins = np.swapaxes(tiled_time_bins, 1, 2)

        self._tiled_time_bins = tiled_time_bins

        try:
            self.set_interpolation()
        except:
            pass

    def set_interpolation(self):
        self._interp1d = Interp1D(self._time_bins, self._interpolation_times)

    def set_saa_mask(self, saa_mask):
        """
        Set the SAA sections in the function array to zero
        :param saa_mask:
        :return:
        """
        self._saa_mask = saa_mask

    def set_responses(self, responses):
        """
        Energie bundaries for the incoming photon spectrum (defined in the response precalculation)
        :param energy_bins:
        :return:
        """
        self._responses = responses

    def integrate_array(self):
        """
        Integrate the count rates to get the counts in each time bin. Can not be precalcualted here as the
        spectral form of the source changes and not only a normalization
        :param time_bins: The time bins of the data
        :return:
        """
        # Get the flux for all times
        if self._use_numba:
            folded_flux_all_dets = self._folded_flux_inter  # (self._time_bins)
        else:
            folded_flux_all_dets = self._folded_flux_inter(self._time_bins)
        # The interpolated flux has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        folded_flux_all_dets = np.swapaxes(folded_flux_all_dets, 1, 2)
        folded_flux_all_dets = np.swapaxes(folded_flux_all_dets, 2, 3)

        if self._use_numba:
            self._source_counts = _trapz_numba(folded_flux_all_dets, self._time_bins)
        else:
            self._source_counts = integrate.trapz(
                folded_flux_all_dets, self._tiled_time_bins
            )
        self._source_counts[~self._saa_mask] = 0.0

    def build_spec_integral(self):

        if self._spec == "bpl":

            def _integral(e1, e2):

                return _spec_integral_bpl(
                    e1, e2, self._C, self._break_energy, self._index1, self._index2
                )

        elif self._spec == "pl":

            def _integral(e1, e2):

                return _spec_integral_pl(
                    e1, e2, self._C, self._E_norm, self._index
                )

        elif self._spec == 'bb+pl':

            def _integral(e1, e2):

                return _spec_integral_bb_pl(
                    e1, e2, self._C_pl, self._E_norm, self._index, self._C_bb, self._temp
                )

        elif self._spec == 'bb':

            def _integral(e1, e2):

                return _spec_integral_bb(
                    e1, e2, self._C, self._temp
                )

        self._spec_integral = _integral


    def _fold_spectrum(self, *parameters):
        """
        Function to fold the spectrum defined by the current parameter values with the precalculated effective response
        :param C:
        :param index1:
        :param index2:
        :param break_energy:
        :return:
        """
        if self._spec == "bpl":

            self._C = parameters[0]
            self._index1 = parameters[1]
            self._index2 = parameters[2]
            self._break_energy = parameters[3]

        elif self._spec == "pl":

            self._C = parameters[0]
            self._index = parameters[1]

        elif self._spec == 'bb+pl':
            self._C_pl = parameters[0]
            self._index = parameters[1]
            self._C_bb = parameters[2]
            self._temp = parameters[3]

        elif self._spec == 'bb':
            self._C = parameters[0]
            self._temp = parameters[1]

        folded_flux = np.zeros(
            (len(self._interpolation_times), len(self._detectors), len(self._echans),)
        )

        for det_idx, det in enumerate(self._detectors):
            true_flux = self._spec_integral(
                self._responses[det].Ebin_in_edge[:-1],
                self._responses[det].Ebin_in_edge[1:],
            )
            if self._use_numba:
                folded_flux[:, det_idx, :] = _dot_numba(
                    true_flux, self._effective_responses[det]
                )
            else:
                folded_flux[:, det_idx, :] = np.dot(
                    true_flux, self._effective_responses[det]
                )
        if self._use_numba:
            self._folded_flux_inter = self._interp1d(folded_flux)
        else:
            self._folded_flux_inter = interpolate.interp1d(
                self._interpolation_times, folded_flux, axis=0
            )

        self.integrate_array()

    def _evaluate(self, *paramter):
        """
        Evalute the function, the params are only passed as dummies
        as the source_counts are already calculated.
        :param paramter: paramters of the source_function
        :param echan: echan
        :return:
        """

        return self._source_counts

    def __call__(self):

        return self._evaluate(*self.parameter_value)

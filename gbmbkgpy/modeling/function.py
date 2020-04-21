import collections
from gbmbkgpy.modeling.parameter import Parameter
import numpy as np
import scipy.integrate as integrate
import numexpr as ne
import scipy.interpolate as interpolate


class Function(object):

    def __init__(self, *parameters):
        """
        Init function of source
        :param parameters: parameters of the source
        """

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


class ContinuumFunction(Function):

    def __init__(self, coefficient_name):
        """
        A continuum function that is parametrized by a constant multiplied by
        a an interpolated function
        :param coefficient_name: the name of the coefficient
        """

        assert isinstance(coefficient_name, str)

        K = Parameter(coefficient_name, initial_value=1., min_value=0, max_value=None, delta=0.1,
                      normalization=True)

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
        self._function_array[~saa_mask] = 0.

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """
        for det_idx in range(self._function_array.shape[1]):
            self._function_array[:, det_idx, :][self._function_array[:, det_idx, :] > 0] = \
                self._function_array[:, det_idx, :][self._function_array[:, det_idx, :] > 0] - \
                np.min(self._function_array[:, det_idx, :][self._function_array[:, det_idx, :] > 0])

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """
        tiled_time_bins = np.tile(
            time_bins,
            (self._function_array.shape[1], 1, 1)
        )

        tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 1)

        self._source_counts = integrate.cumtrapz(self._function_array, tiled_time_bins)[:, :, 0]

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

        K = Parameter(coefficient_name, initial_value=1., min_value=0, max_value=None, delta=0.1,
                      normalization=True)

        super(GlobalFunction, self).__init__(K)

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
        self._function_array[~saa_mask] = 0.

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """
        for det_idx in range(self._function_array.shape[1]):
            self._function_array[:, det_idx, :, :][self._function_array[:, det_idx, :, :] > 0] = \
                self._function_array[:, det_idx, :, :][self._function_array[:, det_idx, :, :] > 0] - \
                np.min(self._function_array[:, det_idx, :, :][self._function_array[:, det_idx, :, :] > 0])

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        tiled_time_bins = np.tile(
            time_bins,
            (self._function_array.shape[1], self._function_array.shape[2], 1, 1)
        )

        tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 2)
        tiled_time_bins = np.swapaxes(tiled_time_bins, 1, 2)

        self._source_counts = integrate.cumtrapz(self._function_array, tiled_time_bins)[:, :, :, 0]

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

    def __init__(self, coefficient_name, spectrum='bpl', E_norm=1):
        """
        Init the parameters of a broken power law
        :param coefficient_name:
        """
        self._E_norm = E_norm
        self._spec = spectrum
        if self._spec == 'bpl':
            C = Parameter(coefficient_name + '_C', initial_value=1., min_value=0, max_value=None, delta=0.1,
                          normalization=True, prior='log_uniform')
            index1 = Parameter(coefficient_name + '_index1', initial_value=-1., min_value=-10, max_value=5, mu=1,
                               sigma=1, delta=0.1, normalization=False, prior='truncated_gaussian')
            index2 = Parameter(coefficient_name + '_index2', initial_value=2., min_value=0.1, max_value=5, mu=1,
                               sigma=1, delta=0.1, normalization=False, prior='truncated_gaussian')
            break_energy = Parameter(coefficient_name + '_break_energy', initial_value=-1., min_value=-10, max_value=5,
                                     delta=0.1, normalization=False, prior='log_uniform')

            super(GlobalFunctionSpectrumFit, self).__init__(C, index1, index2, break_energy)

        elif self._spec == 'pl':

            C = Parameter(coefficient_name + '_C', initial_value=1., min_value=0, max_value=None, delta=0.1,
                          normalization=True, prior='log_uniform')
            index = Parameter(coefficient_name + '_index', initial_value=-1., min_value=0, max_value=3, delta=0.1,
                              mu=1, sigma=1, normalization=False, prior='truncated_gaussian')

            super(GlobalFunctionSpectrumFit, self).__init__(C, index)

        else:

            raise ValueError('Spectrum must be bpl or pl at the moment. But is {}'.format(self._spec))

        self._evaluate = self.build_evaluation_function()


    def set_dets_echans(self, detectors, echans):

        self._nr_detectors = len(detectors)
        self._nr_echans = len(echans)

    def set_effective_response(self, effective_response):
        """
        effective response sum for all times for which the geometry was calculated (NO INTERPOLATION HERE)
        :param response_array:
        :return:
        """
        self.effective_response = effective_response

    def set_interpolation_times(self, interpolation_times):
        """
        times for which the geometry was calculated
        :param interpolation_times:
        :return:
        """
        self._interpolation_times = interpolation_times

    def set_time_bins(self, time_bins):
        """
        Basis array that has the length as the time_bins array with all entries 1
        :param time_bins:
        :return:
        """
        self._time_bins = time_bins

        tiled_time_bins = np.tile(
            time_bins,
            (self._nr_detectors, self._nr_echans, 1, 1)
        )

        self._tiled_time_bins = np.swapaxes(tiled_time_bins, 0, 2)

    def set_saa_mask(self, saa_mask):
        """
        Set the SAA sections in the function array to zero
        :param saa_mask:
        :return:
        """
        self._saa_mask = saa_mask

    def energy_boundaries(self, energy_bins):
        """
        Energie bundaries for the incoming photon spectrum (defined in the response precalculation)
        :param energy_bins:
        :return:
        """
        self._energy_bins = energy_bins

    def integrate_array(self):
        """
        Integrate the count rates to get the counts in each time bin. Can not be precalcualted here as the
        spectral form of the source changes and not only a normalization
        :param time_bins: The time bins of the data
        :return:
        """
        # Get the flux for all times
        folded_flux_all_dets = self._folded_flux_inter(self._time_bins)

        # The interpolated flux has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        folded_flux_all_dets = np.swapaxes(folded_flux_all_dets, 1, 2)
        folded_flux_all_dets = np.swapaxes(folded_flux_all_dets, 2, 3)

        self._source_counts = integrate.cumtrapz(folded_flux_all_dets, self._tiled_time_bins)[:, :, :, 0]

        self._source_counts[~self._saa_mask] = 0.

    def _spectrum(self, energy):
        """
        Defines spectrum of source
        :param energy:
        :return:
        """
        if self._spec == 'bpl':

            return self._C / ((energy / self._break_energy) ** self._index1 + (energy / self._break_energy) ** self._index2)

        elif self._spec == 'pl':

            return self._C / (energy/self._E_norm) ** self._index

    def _integral(self, e1, e2):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return:
        """
        return (e2 - e1) / 6.0 * (
                self._spectrum(e1) + 4 * self._spectrum((e1 + e2) / 2.0) +
                self._spectrum(e2))

    def _fold_spectrum(self, *parameters):
        """
        Function to fold the spectrum defined by the current parameter values with the precalculated effective response
        :param C:
        :param index1:
        :param index2:
        :param break_energy:
        :return:
        """
        if self._spec == 'bpl':

            self._C = parameters[0]
            self._index1 = parameters[1]
            self._index2 = parameters[2]
            self._break_energy = parameters[3]

        elif self._spec == 'pl':

            self._C = parameters[0]
            self._index = parameters[1]


        folded_flux = np.zeros((
            len(self._geom[self._detectors[0]].time),
            len(self._detectors),
            len(self._echans),
        ))

        for det_idx, det in enumerate(self._detectors):
            true_flux = self._integral(
                self._rsp[det].Ebin_in_edge[:-1],
                self._rsp[det].Ebin_in_edge[1:]
            )

            folded_flux[:, det_idx, :] = np.dot(true_flux, self._effective_response[det])

        self._folded_flux_inter = interpolate.interp1d(
            self._interpolation_times,
            self._folded_flux,
            axis=0
        )

        self.integrate_array()

    def build_evaluation_function(self):
        if self._spec == 'bpl':
            def _evaluate(C, index1, index2, break_energy):
                """
                Evaulate this source.
                :param K: the fitted parameter
                :param echan: echan
                :return:
                """

                return self._source_counts

        elif self._spec == 'pl':

            def _evaluate(C, index):
                """
                Evaulate this source.
                :param K: the fitted parameter
                :param echan: echan
                :return:
                """

                return self._source_counts

        return _evaluate

    def __call__(self, echan):

        return self._evaluate(*self.parameter_value)

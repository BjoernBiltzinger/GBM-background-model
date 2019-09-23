import collections
from gbmbkgpy.modeling.parameter import Parameter
import numpy as np
from scipy import integrate
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

        for key, value in self._parameter_dict.iteritems():
            self.__dict__[key] = value


    @property
    def parameter_value(self):
        """
        Returns the current parameter values
        :return:
        """

        return [par.value for par in self._parameter_dict.itervalues()]
    
    def __call__(self, echan):
        """
        Starts the evaluation of the counts per time bin with the current parameters
        :param echan: echan for which the counts should be returned
        :return:
        """

        return self._evaluate(*self.parameter_value, echan = echan)

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
        self._function_array[np.where(~saa_mask)] = 0.

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """

        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - \
                                                         np.min(self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array
        :return:
        """

        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - \
                                                          np.mean(self._function_array[self._function_array != 0],
                                                                  dtype=np.float64)

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        self._integrated_function_array = integrate.cumtrapz(self._function_array, time_bins)

    def _evaluate(self, K, echan=None):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the current parameter value for K
        :param echan: echan,dummy value as this source is only for one echan
        :return:
        """
        int_function_array = self._integrated_function_array[:,0]
        return ne.evaluate("K*int_function_array")


    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan=echan)


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
        self._function_array[:, np.where(~saa_mask)] = 0.

    def remove_vertical_movement(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the minimal value of the array
        :return:
        """

        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - \
                                                         np.min(self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array
        :return:
        """

        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - \
                                                          np.mean(self._function_array[self._function_array != 0],
                                                                  dtype=np.float64)

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is changed during the fit acts as a
        multiplication of a constant on this array. This saves a lot of computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        self._integrated_function_array = []

        for i in range(len(self._function_array)):
            self._integrated_function_array.append(integrate.cumtrapz(self._function_array[i], time_bins))

    def _evaluate(self, K, echan=None):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the fitted parameter
        :param echan: echan
        :return:
        """
        int_function_array_echan = self._integrated_function_array[echan][:, 0]
        return ne.evaluate("K*int_function_array_echan")


    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan=echan)


class GlobalFunctionSpectrumFit(Function):
    """
    A class in which a global constant and spectral parameters can be generated which is the same for all Echans.
    Use this if you want a source with free spectral parameters. Is computational much more expensive than the fixed
    spectrum!
    """
    
    def __init__(self, coefficient_name, spectrum='bpl'):
        """
        Init the parameters of a broken power law
        :param coefficient_name:
        """
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

    def set_response_array(self, response_array):
        """
        effective response sum for all times for which the geometry was calculated (NO INTERPOLATION HERE)
        :param response_array:
        :return:
        """

        self._response_array = response_array

    def set_interpolation_times(self, interpolation_times):
        """
        times for which the geometry was calculated
        :param interpolation_times:
        :return:
        """
        self._interpolation_times = interpolation_times
        
    def set_basis_function_array(self, time_bins):
        """
        Basis array that has the length as the time_bins array with all entries 1
        :param time_bins:
        :return:
        """
        self._time_bins = time_bins
        self._function_array_b = np.ones_like(time_bins)

    def set_saa_zero(self, saa_mask):
        """
        Set the SAA sections in the function array to zero
        :param saa_mask:
        :return:
        """
        self._function_array_b[np.where(~saa_mask)] = 0.
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
        folded_flux_all = self._folded_flux_inter(self._time_bins)
        self._integrated_function_array = []

        # For all echans calculate the count prediction for all time bins
        for i in range(len(folded_flux_all)):
            self._integrated_function_array.append(integrate.cumtrapz(folded_flux_all[i]*self._function_array_b, self._time_bins))

    def _spectrum(self, energy):
        """
        Defines spectrum of source
        :param energy:
        :return:
        """
        if self._spec == 'bpl':

            return self._C / ((energy / self._break_energy) ** self._index1 + (energy / self._break_energy) ** self._index2)

        elif self._spec == 'pl':

            return self._C / energy ** self._index

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

        true_flux = self._integral(self._energy_bins[:-1], self._energy_bins[1:]) 
        folded_flux = np.dot(true_flux, self._response_array)

        self._folded_flux_inter = interpolate.interp1d(self._interpolation_times, folded_flux.T)
        self.integrate_array()

    def build_evaluation_function(self):
        if self._spec=='bpl':
            def _evaluate(C, index1, index2, break_energy, echan=None):
                """
                Evaulate this source.
                :param K: the fitted parameter
                :param echan: echan
                :return:
                """

                return self._integrated_function_array[echan][:, 0]
            
        elif self._spec == 'pl':

            def _evaluate(C, index, echan=None):
                """
                Evaulate this source.
                :param K: the fitted parameter
                :param echan: echan
                :return:
                """

                return self._integrated_function_array[echan][:, 0]

        return _evaluate


    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan=echan)

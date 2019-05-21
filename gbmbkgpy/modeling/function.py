import collections
from gbmbkgpy.modeling.parameter import Parameter
import numpy as np
from scipy import integrate
import numexpr as ne

class Function(object):


    def __init__(self, *parameters):

        parameter_dict = collections.OrderedDict()

        for parameter in parameters:

            parameter_dict[parameter.name] = parameter


        self._parameter_dict = parameter_dict

        for key, value in self._parameter_dict.iteritems():
            self.__dict__[key] = value


    # def __setattr__(self, name, value):
    #     raise Exception("It is read only!")
    #


    @property
    def parameter_value(self):

        return [par.value for par in self._parameter_dict.itervalues()]

    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan = echan)



    #def _evaluate(self):
    #    pass


    @property
    def parameters(self):

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
        Set the temporal interpolation that will be used for the function


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

        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - np.min(self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array
        :return:
        """

        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - np.mean(self._function_array[self._function_array != 0], dtype=np.float64)

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is fit later only acts as a multiplication
        of a constant in the integral. This saves a lot computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        self._integrated_function_array = integrate.cumtrapz(self._function_array, time_bins)

    def _evaluate(self, K, echan=None):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the fitted parameter
        :param echan: echan
        :return:
        """
        int_function_array = self._integrated_function_array[:,0]
        return ne.evaluate("K*int_function_array")


    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan=echan)


class PointSourceFunction(Function):
    def __init__(self, coefficient_name):
        """
        A PointSource function that is parametrized by a constant multiplied by
        a an interpolated function

        :param coefficient_name: the name of the coefficient
        """

        assert isinstance(coefficient_name, str)

        # build the constant

        K = Parameter(coefficient_name, initial_value=1., min_value=0, max_value=None, delta=0.1, normalization=True)

        super(PointSourceFunction, self).__init__(K)

    def set_function_array(self, function_array):
        """
        Set the temporal interpolation that will be used for the function


        :param function_array: a scipy interpolation function
        :return:
        """

        self._function_array = function_array

    def set_earth_zero(self, earth_mask):
        """
        Uses the mask for PS behind earth to set the function array to zero for the timebins for which the mask is 0
        :param earth_mask:
        :return:
        """

        self._function_array[np.where(earth_mask < 0.5)] = 0.

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
        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - np.min(
            self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array
        :return:
        """
        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - np.mean(
            self._function_array[self._function_array != 0], dtype=np.float64)

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is fit later only acts as a multiplication
        of a constant in the integral. This saves a lot computing time!
        :param time_bins: The time bins of the data
        :return:
        """

        self._integrated_function_array = integrate.cumtrapz(self._function_array, time_bins)

    def _evaluate(self, K, echan=None):
        """
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to
        speed up.
        :param K: the fitted parameter
        :param echan: echan
        :return:
        """
        int_function_array = self._integrated_function_array[:, 0]
        return ne.evaluate("K*int_function_array")

    def __call__(self, echan):
        return self._evaluate(*self.parameter_value, echan = echan)


class GlobalFunction(Function):
    """
    A class in which a global constant can be generated which is the same for all Echans
    """
    def __init__(self, coefficient_name):

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

        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - np.min(self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array
        :return:
        """

        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - np.mean(self._function_array[self._function_array != 0], dtype=np.float64)

    def integrate_array(self, time_bins):
        """
        We can precompute the integral over the time bins as the parameter that is fit later only acts as a multiplication
        of a constant in the integral. This saves a lot computing time!
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

class GlobalFunctionEarth(Function):
    """                                                                                                                                                                                                                                                                        
    A class in which a global constant can be generated which is the same for all Echans                                                                                                                                                                                       
    """
    def __init__(self, coefficient_name):

        K = Parameter(coefficient_name, initial_value=1., min_value=0, max_value=None, delta=0.1,
                      normalization=True)
        B = Parameter(coefficient_name, initial_value=0., min_value=0, max_value=0.07, delta=0.01,
                      normalization=True)

        super(GlobalFunctionEarth, self).__init__(K,B)

    def set_function_array(self, function_array):
        
        self._function_array = np.ones_like(function_array)

    def set_base_function_all_times(self, function_array):
        """                                                                                                                                                                                                                                                                    
        Set the temporal interpolation that will be used for the function                                                                                                                                                                                                      
        Here the function_array is a list with as many entries as echans fitted together!                                                                                                                                                                                      
        :param function_array: a scipy interpolation function                                                                                                                                                                                                                  
        :return:                                                                                                                                                                                                                                                               
        """

        self._base_array_all_times = np.array(function_array)
        print("rank {} made it".format(rank))
    def set_angle_of_points_all_times(self, angles_of_all_times):

        self._angles_all = np.array(angles_of_all_times)

    def final_function_array(self, B, echan):
        base_array_echan = self._base_array_all_times[:,:,echan]
        array_echan = ne.evaluate('sum(exp(B*self._angles_all)*base_array_echan,axis=1)')
        return integrate.cumtrapz(self._function_array[i], time_bins)
    
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

        self._function_array[self._function_array > 0] = self._function_array[self._function_array > 0] - np.min(self._function_array[self._function_array > 0])

    def remove_vertical_movement_mean(self):
        """                                                                                                                                                                                                                                                                    
        Remove the vertical movement of the values in the function array by subtracting the mean value of the array                                                                                                                                                            
        :return:                                                                                                                                                                                                                                                               
        """

        self._function_array[self._function_array != 0] = self._function_array[self._function_array != 0] - np.mean(self._function_array[self._function_array != 0], dtype=np.float64)

    def integrate_array(self, time_bins):
        """                                                                                                                                                                                                                                                                    
        We can precompute the integral over the time bins as the parameter that is fit later only acts as a multiplication                                                                                                                                                     
        of a constant in the integral. This saves a lot computing time!                                                                                                                                                                                                        
        :param time_bins: The time bins of the data                                                                                                                                                                                                                            
        :return:                                                                                                                                                                                                                                                               
        """

        self._integrated_function_array = []

        for i in range(len(self._function_array)):
            self._integrated_function_array.append(integrate.cumtrapz(self._function_array[i], time_bins))

    def _evaluate(self, K, B, echan=None):
        """                                                                                                                                                                                                                                                                    
        Evaulate this source. Use the precalculated integrated over the time bins function array and use numexpr to                                                                                                                                                            
        speed up.                                                                                                                                                                                                                                                              
        :param K: the fitted parameter                                                                                                                                                                                                                                        

        :param echan: echan                                                                                                                                                                                                                                                    
        :return:                                                                                                                                                                                                                                                               
        """
        integrated_final_function_array = self._final_function_array(B,echan)[:, 0]
        saa_function_array = self._function_array
        return ne.evaluate("K*final_function_array*saa_function_array")


    def __call__(self, echan):

        return self._evaluate(*self.parameter_value, echan=echan)

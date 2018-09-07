import collections
from gbmbkgpy.modeling.parameter import Parameter
import numpy as np

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
    def parameter_values(self):

        return [par.value for par in self._parameter_dict.itervalues()]

    def __call__(self):

        return self._evaluate(*self.parameter_values)



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

    def _evaluate(self, K):
        return K * self._function_array


    def __call__(self):

        return self._evaluate(*self.parameter_values)


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

    def _evaluate(self, K):
        return K * self._function_array

    def __call__(self):
        return self._evaluate(*self.parameter_values)
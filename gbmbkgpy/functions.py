from gbmbkgpy.modeling.function import Function
from gbmbkgpy.modeling.parameter import Parameter
import numpy as np


class Solar_Flare(Function):

    def __init__(self):

        K = Parameter('K', initial_value=1., min_value=0, max_value= 1, delta= 0.1, normalization = True)
        decay_constant = Parameter('decay_constant', initial_value=1., min_value=0, max_value= 1, delta= 0.1)


        super(Solar_Flare, self).__init__(K, decay_constant)


    def _evaluate(self, x, K, decay_constant):


        return K * np.exp(-x / decay_constant)


class SAA_Decay(Function):

    def __init__(self):

        pass


    def _evaluate(self, x):

        pass


class ContinuumFunction(Function):


    def __init__(self, coefficient_name):

        K =  Parameter(coefficient_name, initial_value=1., min_value=0, max_value= 1, delta= 0.1, normalization = True)

        super(ContinuumFunction, self).__init__(K)


    def set_interpolated_function(self, interpolation):

        self._interpolation = interpolation


    def _evaluate(self, x, K):


         K * self._interpolation(x)


class Cosmic_Gamma_Ray_Background(ContinuumFunction):

    def __init__(self):

        super(Cosmic_Gamma_Ray_Background, self).__init__('a')


class Magnetic_Continuum(Function):
    pass

class Solar_Continuum(Function):
    pass

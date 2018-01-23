from gbmbkgpy.modeling.function import Function, ContinuumFunction, ContinuumFunctionSpecial
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

        A = Parameter('A', initial_value=1., min_value=0, max_value=1, delta=0.1, normalization=True)
        saa_decay_constant = Parameter('saa_decay_constant', initial_value=1., min_value=0, max_value=1, delta=0.1)

        super(SAA_Decay, self).__init__(A, saa_decay_constant)


    def set_saa_exit_time(self, time):

        self._saa_exit_time = time


    def _evaluate(self, time, A, saa_decay_constant):

        out = np.zeros_like(time)
        t0 = self._saa_exit_time
        idx = time < t0

        out[~idx] = A * np.exp(saa_decay_constant * (time[~idx] - t0))

        return out

# The continuums 

class Cosmic_Gamma_Ray_Background(ContinuumFunctionSpecial):
    def __init__(self):
        super(Cosmic_Gamma_Ray_Background, self).__init__('a')

class Magnetic_Continuum(ContinuumFunctionSpecial):
    def __init__(self):
        super(Magnetic_Continuum, self).__init__('b')

class Solar_Continuum(ContinuumFunctionSpecial):
    def __init__(self):
        super(Solar_Continuum, self).__init__('c')

class Point_Source_Continuum(ContinuumFunctionSpecial):
    def __init__(self):
        super(Point_Source_Continuum, self).__init__('d')

class Earth_Albedo_Continuum(ContinuumFunctionSpecial):
    def __init__(self):
        super(Earth_Albedo_Continuum, self).__init__('e')
from gbmbkgpy.modeling.function import Function, ContinuumFunction, PointSourceFunction
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

    def __init__(self, saa_number):

        A = Parameter('A-' + saa_number, initial_value=1., min_value=0, max_value=1, delta=0.1, normalization=True)
        saa_decay_constant = Parameter('saa_decay_constant-' + saa_number, initial_value=-1., min_value=-np.inf, max_value=0, delta=0.1)

        super(SAA_Decay, self).__init__(A, saa_decay_constant)


    def set_saa_exit_time(self, time):

        self._saa_exit_time = time


    def set_time_bins(self, time_bins):

        self._time_bins = time_bins

    def _evaluate(self, A, saa_decay_constant):

        out = np.zeros_like(self._time_bins)
        t0 = self._saa_exit_time
        idx = self._time_bins < t0

        out[~idx] = A * (-saa_decay_constant) * np.exp(saa_decay_constant * (self._time_bins[~idx] - t0))

        return out

# The continuums 

class Cosmic_Gamma_Ray_Background(ContinuumFunction):
    def __init__(self):
        super(Cosmic_Gamma_Ray_Background, self).__init__('a')

class Magnetic_Continuum(ContinuumFunction):
    def __init__(self):
        super(Magnetic_Continuum, self).__init__('b')

class Solar_Continuum(ContinuumFunction):
    def __init__(self):
        super(Solar_Continuum, self).__init__('c')

class Earth_Albedo_Continuum(PointSourceFunction):
    def __init__(self):
        super(Earth_Albedo_Continuum, self).__init__('d')

class Point_Source_Continuum(PointSourceFunction):
    def __init__(self):
        super(Point_Source_Continuum, self).__init__('e')


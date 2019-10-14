from gbmbkgpy.modeling.function import Function, ContinuumFunction, GlobalFunction, GlobalFunctionSpectrumFit
from gbmbkgpy.modeling.parameter import Parameter
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


class Solar_Flare(Function):

    def __init__(self):
        K = Parameter('K', initial_value=1., min_value=0, max_value=None, delta=0.1, normalization=True)
        decay_constant = Parameter('decay_constant', initial_value=-0.01, min_value=-1, max_value=0, delta=0.1)

        super(Solar_Flare, self).__init__(K, decay_constant)

    def _evaluate(self, x, K, decay_constant, echan=None):
        return K * np.exp(-x / decay_constant)


class SAA_Decay(Function):

    def __init__(self, saa_number, echan):
        A = Parameter("A-{} echan-{}".format(saa_number, echan), initial_value=1., min_value=0, max_value=None, delta=0.1, normalization=True, prior='log_uniform')
        saa_decay_constant = Parameter("saa_decay_constant-{} echan-{}".format(saa_number, echan), initial_value=0.01, min_value=0., max_value=1., delta=0.1, prior='log_uniform')

        super(SAA_Decay, self).__init__(A, saa_decay_constant)

    def set_saa_exit_time(self, time):
        self._saa_exit_time = time

    def set_time_bins(self, time_bins):
        self._time_bins = time_bins

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

    def _evaluate(self, A, saa_decay_constant, echan=None):
        """
        Calculates the exponential decay for the SAA exit
        The the values are calculated for the start and stop times of the bins with the analytic solution of the integral
        for a function A*exp(-saa_decay_constant*(t-t0)) which is -A/saa_decay_constant *
        (exp(-saa_decay_constant*(tend_bin-to) - exp(-saa_decay_constant*(tstart_bin-to))
        :param A:
        :param saa_decay_constant:
        :return:
        """

        out = np.zeros_like(self._time_bins[:, 0])

        t0 = self._t0
        tstart = self._tstart
        tstop = self._tstop

        # out[~self._idx_start] = ne.evaluate("-A / saa_decay_constant*(exp((t0-tstop)*saa_decay_constant) - exp((t0 - tstart)*saa_decay_constant))")
        out[~self._idx_start] = ne.evaluate("-A / saa_decay_constant*(exp((t0-tstop)*abs(saa_decay_constant)) - exp((t0 - tstart)*abs(saa_decay_constant)))")

        return out


class GRB(Function):

    def __init__(self):
        super(GRB, self).__init__()

    def set_time_bins(self, time_bins):
        self._time_bins = time_bins

    def set_grb_params(self, A, t_start, t_rise, t_decay):
        self._A = A
        self._t_start = t_start
        self._t_rise = t_rise
        self._t_decay = t_decay

    def _evaluate(self, echan=None):
        """
        Calculates a "typical" GRB pulse with a preset rise and decay time.
        The the values are calculated for the start and stop times of the bins for vectorized integration
        :return:
        """

        out = np.zeros_like(self._time_bins)
        idx_start = self._time_bins[:, 0] < self._t_start
        idx_stop = self._time_bins[:, 1] < self._t_start

        out[:, 0][~idx_start] = self._A * np.exp(2 * (self._t_rise / self._t_decay) ** (1 / 2)) * np.exp(-self._t_rise /
                                                                                                         (self._time_bins[:, 0][~idx_start] - self._t_start) - (self._time_bins[:, 0][~idx_start] - self._t_start) / self._t_decay)

        out[:, 1][~idx_stop] = self._A * np.exp(2 * (self._t_rise / self._t_decay) ** (1 / 2)) * np.exp(-self._t_rise /
                                                                                                        (self._time_bins[:, 1][~idx_stop] - self._t_start) - (self._time_bins[:, 1][~idx_stop] - self._t_start) / self._t_decay)
        return out


# The continuums

class Cosmic_Gamma_Ray_Background(GlobalFunction):
    def __init__(self):
        super(Cosmic_Gamma_Ray_Background, self).__init__('norm_cgb')


class Magnetic_Continuum(ContinuumFunction):
    def __init__(self, echan):
        super(Magnetic_Continuum, self).__init__('norm_magnetic_echan-' + echan)


class Solar_Continuum(ContinuumFunction):
    def __init__(self, echan):
        super(Solar_Continuum, self).__init__('norm_solar_echan-' + echan)


class Earth_Albedo_Continuum(GlobalFunction):
    def __init__(self):
        super(Earth_Albedo_Continuum, self).__init__('norm_earth_albedo')


class Point_Source_Continuum(GlobalFunction):
    def __init__(self, name):
        super(Point_Source_Continuum, self).__init__(name)


class offset(ContinuumFunction):
    def __init__(self, echan):
        super(offset, self).__init__('constant_echan-' + echan)


class Magnetic_Continuum_Global(GlobalFunction):
    def __init__(self):
        super(Magnetic_Continuum_Global, self).__init__('norm_magnetic_global')


class Magnetic_Constant_Global(GlobalFunction):
    def __init__(self):
        super(Magnetic_Constant_Global, self).__init__('constant_magnetic_global')


class Earth_Albedo_Continuum_Fit_Spectrum(GlobalFunctionSpectrumFit):
    def __init__(self):
        super(Earth_Albedo_Continuum_Fit_Spectrum, self).__init__('Earth_Albedo-Spectrum_fitted', spectrum='bpl')


class Cosmic_Gamma_Ray_Background_Fit_Spectrum(GlobalFunctionSpectrumFit):
    def __init__(self):
        super(Cosmic_Gamma_Ray_Background_Fit_Spectrum, self).__init__('CGB-Spectrum_fitted', spectrum='bpl')


class Point_Source_Continuum_Fit_Spectrum(GlobalFunctionSpectrumFit):
    def __init__(self, name, E_norm=1):
        super(Point_Source_Continuum_Fit_Spectrum, self).__init__(name, spectrum='pl', E_norm=E_norm)


class SAA_Decay_Linear(ContinuumFunction):
    def __init__(self, echan):
        super(SAA_Decay_Linear, self).__init__('saa_decay_long_echan-' + echan)


# Testing secondary earth

class Magnetic_Secondary_Continuum(ContinuumFunction):
    def __init__(self, echan):
        super(Magnetic_Secondary_Continuum, self).__init__('secondary_echan-' + echan)


class West_Effect_Continuum(ContinuumFunction):
    def __init__(self, echan):
        super(West_Effect_Continuum, self).__init__('west_effect-' + echan)

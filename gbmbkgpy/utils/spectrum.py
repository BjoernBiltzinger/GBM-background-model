#!/usr/bin/env python3
import numpy as np


try:
    from numba import njit, float64, prange

    has_numba = True
except:
    has_numba = False

################################## NUMBA Implementations ###############################################

if has_numba:

    ############### Spectra definitions ###########################

    # Numba Implementation
    @njit(cache=True)
    def _spectrum_bpl(energy, c, break_energy, index1, index2):
        """
        Calculates the differential spectra
        :param energy: energy where to evaluate bpl
        :param c: C param of bpl
        :param break_energy: Energy where the bpl breaks
        :param index: index of bpl
        :return: differential pl evaluation [1/kev*s]
        """

        return c / (
            (energy / break_energy) ** index1 + (energy / break_energy) ** index2
        )

    @njit(cache=True)
    def _spectrum_pl(energy, c, e_norm, index):
        """
        Calculates the differential spectra
        :param energy: energy where to evaluate pl
        :param c: C param of pl
        :param e_norm: Energy where to norm the pl
        :param index: index of pl
        :return: differential pl evaluation [1/kev*s]
        """
        # norm to crab
        one_crab = 15.0
        dp2 = 2-index
        inv_int_flux = 1.0/(195.0**dp2-14.0**dp2)*dp2
        return c * one_crab * inv_int_flux *(energy)**-index

    @njit(cache=True)
    def _spectrum_bb(energy, c, temp):
        """
        Calculates the differential spectra
        :param energy: energy where to evaluate pl
        :param c: C param of pl
        :param e_norm: Energy where to norm the pl
        :param index: index of pl
        :return: differential pl evaluation [1/kev*s]
        """
        
        return c * energy ** 2 / (np.expm1(energy / temp))

    ##################### Integration of spectra #############################

    @njit(cache=True)
    def _spec_integral_pl(e1, e2, c, e_norm, index):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return: number photons per second in the incoming ebins
        """
        res = np.zeros(len(e1))
        for i in prange(len(e1)):

            res[i] = (
                (e2[i] - e1[i])
                / 6.0
                * (
                    _spectrum_pl(e1[i], c, e_norm, index)
                    + 4 * _spectrum_pl((e1[i] + e2[i]) / 2.0, c, e_norm, index)
                    + _spectrum_pl(e2[i], c, e_norm, index)
                )
            )
        return res

    @njit(cache=True)
    def _spec_integral_bb(e1, e2, c, temp):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return: number photons per second in the incoming ebins
        """
        res = np.zeros(len(e1))
        for i in prange(len(e1)):

            res[i] = (
                (e2[i] - e1[i])
                / 6.0
                * (
                    _spectrum_bb(e1[i], c, temp)
                    + 4 * _spectrum_bb((e1[i] + e2[i]) / 2.0, c, temp)
                    + _spectrum_bb(e2[i], c, temp)
                )
            )
        return res

    @njit(
        cache=True,
    )
    def _spec_integral_bb_pl(e1, e2, c_pl, e_norm, index, c_bb, temp):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return: number photons per second in the incoming ebins
        """
        res = np.zeros(len(e1))
        for i in prange(len(e1)):

            res[i] = (
                (e2[i] - e1[i])
                / 6.0
                * (
                    (
                        _spectrum_pl(e1[i], c_pl, e_norm, index)
                        + _spectrum_bb(e1[i], c_bb, temp)
                    )
                    + 4
                    * (
                        _spectrum_pl((e1[i] + e2[i]) / 2.0, c_pl, e_norm, index)
                        + _spectrum_bb((e1[i] + e2[i]) / 2.0, c_bb, temp)
                    )
                    + (
                        _spectrum_pl(e2[i], c_pl, e_norm, index)
                        + _spectrum_bb(e2[i], c_bb, temp)
                    )
                )
            )
        return res

    @njit(
        cache=True,
    )
    def _spec_integral_bpl(e1, e2, c, break_energy, index1, index2):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return: number photons per second in the incoming ebins
        """
        res = np.zeros(len(e1))
        for i in prange(len(e1)):
            res[i] = (
                (e2[i] - e1[i])
                / 6.0
                * (
                    _spectrum_bpl(e1[i], c, break_energy, index1, index2)
                    + 4
                    * _spectrum_bpl(
                        (e1[i] + e2[i]) / 2.0, c, break_energy, index1, index2
                    )
                    + _spectrum_bpl(e2[i], c, break_energy, index1, index2)
                )
            )
        return res


else:
    ################################## NUMPY Implementations #################################################

    ############################# Vectorized Spectrum Functions###########################

    def _spectrum_bpl(energy, c, break_energy, index1, index2):

        return c / (
            (energy / break_energy) ** index1 + (energy / break_energy) ** index2
        )

    def _spectrum_pl(energy, c, e_norm, index):

        return c / (energy / e_norm) ** index

    def _spectrum_bb(energy, c, temp):

        return c * energy ** 2 / (np.expm1(energy / temp))

    ########################## Integration over energy bins ###############################

    def _spec_integral_pl(e1, e2, c, e_norm, index):

        return (
            (e2 - e1)
            / 6.0
            * (
                _spectrum_pl(e1, c, e_norm, index)
                + 4 * _spectrum_pl((e1 + e2) / 2.0, c, e_norm, index)
                + _spectrum_pl(e2, c, e_norm, index)
            )
        )

    def _spec_integral_bb_pl(e1, e2, c_pl, e_norm, index, c_bb, temp):

        return (
            (e2 - e1)
            / 6.0
            * (
                (_spectrum_pl(e1, c_pl, e_norm, index) + _spectrum_bb(e1, c_bb, temp))
                + 4
                * (
                    _spectrum_pl((e1 + e2) / 2.0, c_pl, e_norm, index)
                    + _spectrum_bb((e1 + e2) / 2.0, c_bb, temp)
                )
                + (_spectrum_pl(e2, c_pl, e_norm, index) + _spectrum_bb(e2, c_bb, temp))
            )
        )

    def _spec_integral_bpl(e1, e2, c, break_energy, index1, index2):

        return (
            (e2 - e1)
            / 6.0
            * (
                _spectrum_bpl(e1, c, break_energy, index1, index2)
                + 4 * _spectrum_bpl((e1 + e2) / 2.0, c, break_energy, index1, index2)
                + _spectrum_bpl(e2, c, break_energy, index1, index2)
            )
        )

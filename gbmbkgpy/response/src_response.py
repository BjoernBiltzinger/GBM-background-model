import numpy as np
from scipy.interpolate import interp1d

from gbmbkgpy.utils.progress_bar import progress_bar


def cart2ang(vec):
    """
    transform normalized vec to ra, dec
    :param vec: norm vector
    :return: ra, dec
    """

    if len(vec.shape) == 1:
        dec = np.arcsin(vec[2])
        ra = np.arctan2(vec[1], vec[0])
    else:
        dec = np.arcsin(vec[:, 2])
        ra = np.arctan2(vec[:, 1], vec[:, 0])
    return np.rad2deg(ra), np.rad2deg(dec)


class PointSourceResponse:

    def __init__(self, response_generator, interp_times, ra, dec):
        """
        :param ra: ra in ICRS
        :param dec: dec in ICRS
        """

        self._rsp_gen = response_generator
        self._times = interp_times
        self._ra = ra
        self._dec = dec

        self._num_ebins_out = self._rsp_gen.num_ebins_out
        self._Ebins_in_edge = self._rsp_gen.Ebins_in_edge

        responses = np.zeros((len(self._times),
                              len(self._rsp_gen._Ebins_in_edge)-1,
                              self._rsp_gen._num_ebins_out
                              ))

        for i, time in enumerate(self._times):
            responses[i] = self._rsp_gen.calc_response_ra_dec(self._ra,
                                                              self._dec,
                                                              time,
                                                              occult=True)

        self._effective_response_interp = interp1d(self._times,
                                                   responses,
                                                   axis=0,
                                                   fill_value='extrapolate')

    def interp_effective_response(self, time):
        return self._effective_response_interp(time)

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def num_ebins_out(self):
        return self._rsp_gen.num_ebins_out

    @property
    def Ebins_in_edge(self):
        return self._rsp_gen.Ebins_in_edge


class ExtendedSourceResponse:

    def __init__(self, times, resp_prec, weights):

        response_grid = resp_prec.response_grid

        self._resp_prec = resp_prec

        self._num_ebins_out = self._resp_prec.drm_gen.num_ebins_out
        self._Ebins_in_edge = self._resp_prec.drm_gen.Ebins_in_edge

        assert weights.shape[1] == response_grid.shape[0],\
            "Shape mismatch"
        assert times.shape[0] == weights.shape[0],\
            "Shape mismatch"

        self._response_grid = response_grid
        self._weights = weights
        self._times = times

        self._num_times = len(times)
        self._area_per_point = 4*np.pi/(len(response_grid))

        self._calc_effective_responses()

        # build interpolation
        self._effective_response_interp = interp1d(self._times,
                                                   self._effective_responses,
                                                   axis=0,
                                                   fill_value='extrapolate')

    def _calc_effective_responses(self):
        eff_responses = np.zeros((self._num_times,
                                  *self._response_grid.shape[1:]))
        for i in range(self._num_times):
            eff_responses[i] = np.dot(self._response_grid.T,
                                      self._weights[i]).T

        self._effective_responses = eff_responses

    @property
    def effective_responses(self):
        return self._effective_responses

    @property
    def num_ebins_out(self):
        return self._resp_prec.drm_gen.num_ebins_out

    @property
    def Ebins_in_edge(self):
        return self._resp_prec.drm_gen.Ebins_in_edge

    def interp_effective_response(self, time):
        return self._effective_response_interp(time)


class EarthCGBResponse(ExtendedSourceResponse):

    def __init__(self, geometry, interp_times, resp_prec, kind="earth albedo"):

        assert kind in ["earth albedo", "cgb"]

        weights = self._construct_weights(geometry,
                                          interp_times,
                                          resp_prec,
                                          kind)

        super().__init__(interp_times, resp_prec,
                         weights)

    def _construct_weights(self, geom, interp_times, resp_prec, kind):

        # weights
        weights = np.zeros((len(interp_times),
                            len(resp_prec._points)
                            ),
                           dtype=bool
                           )

        # normalized response grid points in sat frame from pre_calc
        grid_points_pos_norm_vec = (resp_prec._points /
                                    np.linalg.norm(resp_prec._points,
                                                   axis=1)[:, np.newaxis])

        # get az, el of grid points
        azs, els = cart2ang(grid_points_pos_norm_vec)

        for k, time in enumerate(interp_times):

            ras, decs = geom.satellite_to_icrs(time, azs, els)
            occulted = geom.is_occulted(time, ras, decs)

            if kind == "earth albedo":
                weights[k, occulted] = 1
            else:
                weights[k, ~occulted] = 1

        return weights


class EarthResponse(EarthCGBResponse):

    def __init__(self, geometry, interp_times, resp_prec):

        super().__init__(geometry, interp_times, resp_prec, kind="earth albedo")


class CGBResponse(EarthCGBResponse):

    def __init__(self, geometry, interp_times, resp_prec):

        super().__init__(geometry, interp_times, resp_prec, kind="cgb")


class GalacticCenterResponse(ExtendedSourceResponse):

    def __init__(self, geometry, interp_times, resp_prec):

        weights = self._construct_weights(geometry, interp_times, resp_prec)

        super().__init__(interp_times,
                         resp_prec,
                         weights)

    def _construct_weights(self, geom, interp_times, resp_prec):

        # weights
        weights = np.zeros((len(interp_times),
                            len(resp_prec._points)
                            )
                           )

        # normalized response grid points in sat frame from pre_calc
        grid_points_pos_norm_vec = (resp_prec._points /
                                    np.linalg.norm(resp_prec._points,
                                                   axis=1)[:, np.newaxis])

        # get az, el of grid points
        azs, els = cart2ang(grid_points_pos_norm_vec)

        for k, time in enumerate(interp_times):

            ras, decs = geom.satellite_to_icrs(time, azs, els)
            occulted = geom.is_occulted(time, ras, decs)

            l, b = geom.satellite_to_galactic(time,
                                              azs[~occulted],
                                              els[~occulted])

            weights[k, ~occulted] = self._lorentzian(l, b)

        return weights

    def _lorentzian(self, l, b):
        # define lorentzian functions for GC according
        # to M. Türler in Integral paper
        # FWHM_l is 21 degrees
        #
        l = np.deg2rad(l)
        b = np.deg2rad(b)
        #
        FWHM_l = np.deg2rad(21.)
        l_0 = 0
        gamma_l = FWHM_l / 2

        # FWHM_b is 1.2 degrees
        FWHM_b = np.deg2rad(1.2)
        # b_0 is -0.15 degrees
        b_0 = np.deg2rad(-0.15)
        gamma_b = FWHM_b / 2

        # Normalization so that the integral of the resulting
        # function over the unit sphere equals 1
        c = 1 / 0.0180403

        return (c *
                (gamma_l**2 / ((l - l_0)**2 + gamma_l**2)) *
                (gamma_b**2 / ((b - b_0)**2 + gamma_b**2)))

from collections import Iterable
import numpy as np


class ResponseGenerator:

    def __init__(self, geometry, Ebins_in_edge, num_ebins_out):
        self._geometry = geometry
        self._Ebins_in_edge = Ebins_in_edge
        self._num_ebins_out = num_ebins_out

    def calc_response_az_zen(self, az, zen):
        raise NotImplementedError("Has to be implenemented in subclass")

    def calc_response_ra_dec(self, ra, dec, time, occult):

        if occult:
            if self._geometry.is_occulted(time, ra, dec):
                return np.zeros(
                    (
                        len(self._Ebins_in_edge)-1,
                        self._num_ebins_out)
                )

        az, zen = self._geometry.icrs_to_satellite(time, ra, dec)
        if not isinstance(az, Iterable):
            return self.calc_response_az_zen(az, zen)

        res = [self.calc_response_az_zen(a, z) for (a, z) in zip(az, zen)]

        if len(res) == 1:
            return res[0]

        return res

    def calc_response_xyz(self, x, y, z):
        """
        calc response matrix for a given position in detector frame
        defined by x,y and z
        :returns: response matrix
        """
        vec = np.array([x, y, z])
        norm = np.linalg.norm(vec)
        norm_vec = vec/norm

        zen = np.rad2deg(np.arcsin(norm_vec[2]))
        az = np.rad2deg(np.arctan2(norm_vec[1], norm_vec[0]))

        return self.calc_response_az_zen(az, zen)

    @property
    def Ebins_in_edge(self):
        return self._Ebins_in_edge

    @property
    def num_ebins_out(self):
        return self._num_ebins_out

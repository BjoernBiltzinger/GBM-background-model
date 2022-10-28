import numpy as np
from gbm_drm_gen.drmgen import DRMGen

from gbmbkgpy.response.response import ResponseGenerator

valid_det_names = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]

class GBMResponseGenerator(ResponseGenerator):

    def __init__(self, geometry, det, Ebins_in_edge, data):

        self._echans_mask = data.echans_mask
        Ebins_out_edge = data.ebin_out_edges

        super().__init__(geometry, Ebins_in_edge, self._echans_mask.shape[0])

        # detector name <-> number convention for GBM

        assert det in valid_det_names

        det_num = np.argwhere(np.array(valid_det_names) == det)[0, 0]

        self._drm_gen_no_occult = DRMGen(
            self._geometry._position_interpolator,
            det_num,
            Ebins_in_edge,
            mat_type=0,
            ebin_edge_out=Ebins_out_edge,
            occult=False,
            time=geometry._position_interpolator.time[1]
            )

    def calc_response_az_zen(self, az, zen):
        """
        calc response matrix for a given position in detector frame
        defined by az and zen
        :returns: response matrix
        """

        rsp = self._drm_gen_no_occult

        mat = rsp.to_3ML_response_direct_sat_coord(az, zen).matrix.T

        # sum the responses needed
        response_final = np.zeros((mat.shape[0], self._echans_mask.shape[0]))

        for i, echan_mask in enumerate(self._echans_mask):
            for j, entry in enumerate(echan_mask):
                if entry:
                    response_final[:, i] += mat[:, j]

        return response_final

from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.utils.spectrum import _spec_integral_bb, _spec_integral_pl
from scipy.interpolate import interpolate
from gbmgeometry import PositionInterpolator
import astropy.coordinates as coord
from gbm_drm_gen.drmgen import DRMGen
import numpy as np

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

# TODO: Change Sun object for new layout with multi detectors
class Sun(object):
    def __init__(self, det_responses, geometry_object, echans, spec):
        """
        Initalize the Sun as source with a free powerlaw and precalcule the response for all times
        for which the geometry was calculated.
        :params response_object: response_precalculation object
        :params geometry_object: geomerty precalculatation object
        :params echan_list: which echan
        """
        self._geom = geometry_object
        self._rsp = det_responses.responses
        self._detectors = list(det_responses.responses.keys())
        self._data_type = self._rsp[self._detectors[0]].data_type
        self._echans = echans

        if self._data_type == "ctime" or self._data_type == "trigdat":
            echans_mask = []

            for e in echans:
                bounds = e.split("-")
                mask = np.zeros(8, dtype=bool)
                if len(bounds) == 1:
                    # Only one echan given
                    index = int(bounds[0])
                    mask[index] = True
                else:
                    # Echan start and stop given
                    index_start = int(bounds[0])
                    index_stop = int(bounds[1])
                    mask[index_start : index_stop + 1] = np.ones(
                        1 + index_stop - index_start, dtype=bool
                    )
                echans_mask.append(mask)

        elif self._data_type == "cspec":
            echans_mask = []

            for e in echans:
                bounds = e.split("-")
                mask = np.zeros(128, dtype=bool)
                if len(bounds) == 1:
                    # Only one echan given
                    index = int(bounds[0])
                    mask[index] = True
                else:
                    # Echan start and stop given
                    index_start = int(bounds[0])
                    index_stop = int(bounds[1])
                    mask[index_start : index_stop + 1] = np.ones(
                        1 + index_stop - index_start, dtype=bool
                    )
                echans_mask.append(mask)

        self._echans_mask = echans_mask
        self._piv = 10
        self._calc_det_responses()
        # Read the spec dict to figure out which spec and the spectral params
        self._read_spec(spec)

        # Calculate the rates for the ps with the normalization set to 1
        self._calc_sun_rates()

        # Interpolate between the times, for which the geometry was calculated
        self._interpolate_sun_rates()

    @property
    def sun_response_array(self):
        """
        Returns an array with the poit source response for the times for which the geometry
        was calculated.
        """
        return self._sun_response

    @property
    def geometry_times(self):

        return self._geom.time

    @property
    def Ebin_in_edge(self):
        """
        Returns the Ebin_in edges as defined in the response object
        """
        return self._rsp.Ebin_in_edge

    def _calc_det_responses(self):

        sun_response = {}

        for det_idx, det in enumerate(self._detectors):

            sun_response[det] = self._response_one_det(det_response=self._rsp[det])

        self._sun_response = sun_response

    
    def _response_one_det(self, det_response):
        """
        Funtion that imports and precalculate everything that is needed to get the point source array
        for all echans
        :return:
        """

        response_matrix = []

        sun_positions = self._geom.sun_positions
        
        pos_inter = PositionInterpolator(quats=np.array([self._geom.quaternion[0], self._geom.quaternion[0]]), 
                                               sc_pos=np.array([self._geom.sc_pos[0],self._geom.sc_pos[0]]) ,
                                               time=np.array([-1,1]), trigtime=0)

        d = DRMGen(
            pos_inter,
            det_response.det,
            det_response.Ebin_in_edge,
            mat_type=0,
            ebin_edge_out=det_response.Ebin_out_edge,
            )
        for j in range(len(self._geom.quaternion)):
            d._quaternions = self._geom.quaternion[j]
            d._sc_pos = self._geom.sc_pos[j]
            d._compute_spacecraft_coordinates()
            
            all_response_step = (d
                .to_3ML_response_direct_sat_coord(sun_positions[j].lon.deg, sun_positions[j].lat.deg)
                .matrix
            )

            # sum the responses needed
            response_step = np.zeros(
                (len(self._echans_mask), len(all_response_step[0]))
            )
            for i, echan_mask in enumerate(self._echans_mask):
                for j, entry in enumerate(echan_mask):
                    if entry:
                        response_step[i] += all_response_step[j]

            response_matrix.append(response_step.T)

        response_matrix = np.array(response_matrix)

        return response_matrix

    def _read_spec(self, spec):
        """
        Read the spec dict to figure out which spectral type and which params should be used
        :param spec: Dict with spectral type and parameters
        """

        # Read spectral type
        self._spec_type = spec["spectrum_type"]

        # Check if spectral type is valid
        assert (
            self._spec_type == "bb" or self._spec_type == "pl"
        ), "Spectral Type must be bb (Blackbody) or pl (Powerlaw)"

        # Read spectral params
        if self._spec_type == "bb":
            self._bb_temp = spec["blackbody_temp"]
        if self._spec_type == "pl":
            self._pl_index = spec["powerlaw_index"]

    def get_sun_rates(self, met):
        """
        Returns an array with the predicted count rates for the times for which the geometry
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later)
        and the fixed spectral index defined in the init of the object.
        :param met:
        """

        # Get the rates for all times
        sun_rates = self._interp_rate_sun(met)

        # The interpolated flux has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        sun_rates = np.swapaxes(sun_rates, 1, 2)
        sun_rates = np.swapaxes(sun_rates, 2, 3)

        return sun_rates

    def _calc_sun_rates(self):
        """
        Calaculates the rate in all energy channels for all times for which the geometry was calculated.
        Uses the responses calculated in _response_array.
        :param pl_index: Index of powerlaw
        """

        folded_flux_sun = np.zeros(
            (
                len(self._geom.geometry_times),
                len(self._detectors),
                len(self._echans),
            )
        )

        for det_idx, det in enumerate(self._detectors):
            true_flux_sun = self._integral_sun(
                e1=self._rsp[det].Ebin_in_edge[:-1], e2=self._rsp[det].Ebin_in_edge[1:]
            )

            sun_response_det = self._sun_response[det]

            folded_flux_sun[:, det_idx, :] = np.dot(true_flux_sun, sun_response_det)

        self._folded_flux_sun = folded_flux_sun

    def _interpolate_sun_rates(self):

        self._interp_rate_sun = interpolate.interp1d(
            self._geom.geometry_times, self._folded_flux_sun, axis=0
        )

    def _integral_sun(self, e1, e2):
        """
        Method to integrate the diff. flux over the Ebins of the incoming photons
        :params e1: lower bound of Ebin_in
        :params e2: upper bound of Ebin_in
        :params index: Index of spectrum
        :return: flux in the Ebin_in
        """
        if self._spec_type == "bb":
            return _spec_integral_bb(e1, e2, 1, self._bb_temp)

        if self._spec_type == "pl":
            return _spec_integral_pl(e1, e2, 1, self._piv, self._pl_index)

    @property
    def spec_type(self):
        return self._spec_type

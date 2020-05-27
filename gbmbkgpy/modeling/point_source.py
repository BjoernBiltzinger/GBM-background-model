import astropy.coordinates as coord
import astropy.units as u
from gbmgeometry.gbm_frame import GBMFrame
from gbm_drm_gen.drmgen import DRMGen
from scipy.interpolate import interpolate

from gbmbkgpy.utils.progress_bar import progress_bar

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


class PointSrc_free(object):
    def __init__(self, name, ra, dec, det_responses, geometry, echans):
        """
        Initialize a PS and precalculates the response for all the times for which the geomerty was
        calculated. Needed for a spectral fit of the point source.

        :params name: Name of PS
        :params ra: ra position of PS (J2000)
        :params dec: dec position of PS (J2000)
        :params response_object: response_precalculation object
        :params geometry_object: geomerty precalculatation object
        :params index: Powerlaw index of PS spectrum
        """
        self._name = name

        self._ra = ra
        self._dec = dec

        self._detectors = list(det_responses.responses.keys())

        self._rsp = det_responses.responses
        self._geom = geometry

        self._data_type = self._rsp[self._detectors[0]].data_type

        self._echans = echans

        if self._data_type == "ctime":

            self._echan_mask = np.zeros(8, dtype=bool)

            for e in echans:

                self._echan_mask[e] = True

        elif self._data_type == "cspec":

            self._echan_mask = np.zeros(128, dtype=bool)

            for e in echans:

                self._echan_mask[e] = True

        self._calc_det_responses()

    @property
    def responses(self):
        return self._rsp

    @property
    def ps_effective_response(self):
        """
        Returns an array with the poit source response for the times for which the geometry
        was calculated.
        """
        return self._ps_response

    def _calc_det_responses(self):

        ps_response = {}

        for det_idx, det in enumerate(self._detectors):

            ps_response[det] = self._response_one_det(
                det_response=self._rsp[det]
            )

        self._ps_response = ps_response

    def _response_one_det(self, det_response):

        response_matrix = []

        for j in range(len(self._geom.quaternion)):
            response_step = (
                DRMGen(
                    self._geom.quaternion[j],
                    self._geom.sc_pos[j],
                    det_response.det,
                    det_response.Ebin_in_edge,
                    mat_type=0,
                    ebin_edge_out=det_response.Ebin_out_edge,
                )
                .to_3ML_response(self._ra, self._dec)
                .matrix[self._echan_mask]
            )

            response_matrix.append(response_step.T)

        response_matrix = np.array(response_matrix)

        return response_matrix

    @property
    def name(self):

        return self._name


class PointSrc_fixed(PointSrc_free):
    def __init__(
            self, name, ra, dec, det_responses, geometry, echans, spec#spectral_index=2.114
    ):
        super(PointSrc_fixed, self).__init__(name, ra, dec, det_responses, geometry, echans)
        """
        Initialize a PS and precalculates the rates for all the times for which the geomerty was
        calculated.

        :params name: Name of PS
        :params ra: ra position of PS (J2000)
        :params dec: dec position of PS (J2000)
        :params response_object: response_precalculation object
        :params geometry_object: geomerty precalculatation object
        :params spec: Which spectrum type? pl or bb? And spectral params.
        #:params index: Powerlaw index of PS spectrum
        """
        # Read the spec dict to figure out which spec and the spectral params
        self._read_spec(spec)

        # Calculate the rates for the ps with the normalization set to 1
        self._calc_ps_rates()

        # Interpolate between the times, for which the geometry was calculated
        self._interpolate_ps_rates()

    def _read_spec(self, spec):
        """
        Read the spec dict to figure out which spectral type and which params should be used
        :param spec: Dict with spectral type and parameters
        """

        # Read spectral type
        self._spec_type = spec["spectrum_type"]

        # Check if spectral type is valid
        assert (self._spec_type == 'bb' or self._spec_type == 'pl'), "Spectral Type must be bb (Blackbody) or pl (Powerlaw)"

        # Read spectral params
        if self._spec_type == 'bb':
            self._bb_temp = spec['blackbody_temp']
        if self._spec_type == 'pl':
            self._pl_index = spec['powerlaw_index']

    def get_ps_rates(self, met):
        """
        Returns an array with the predicted count rates for the times for which the geometry
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later)
        and the fixed spectral index defined in the init of the object.
        :param met:
        """

        # Get the rates for all times
        ps_rates = self._interp_rate_ps(met)

        # The interpolated flux has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        ps_rates = np.swapaxes(ps_rates, 1, 2)
        ps_rates = np.swapaxes(ps_rates, 2, 3)

        return ps_rates

    def _calc_ps_rates(self):
        """
        Calaculates the rate in all energy channels for all times for which the geometry was calculated.
        Uses the responses calculated in _response_array.
        :param pl_index: Index of powerlaw
        """

        folded_flux_ps = np.zeros(
            (
                len(self._geom.geometry_times),
                len(self._detectors),
                len(self._echans),
            )
        )

        for det_idx, det in enumerate(self._detectors):
            true_flux_ps = self._integral_ps(
                e1=self._rsp[det].Ebin_in_edge[:-1],
                e2=self._rsp[det].Ebin_in_edge[1:]
            )

            ps_response_det = self._ps_response[det]

            folded_flux_ps[:, det_idx, :] = np.dot(true_flux_ps, ps_response_det)

        self._folded_flux_ps = folded_flux_ps

    def _interpolate_ps_rates(self):

        self._interp_rate_ps = interpolate.interp1d(
            self._geom.geometry_times, self._folded_flux_ps, axis=0
        )

    def _spectrum_ps(self, energy):
        """
        Define the function of a spectral form. Needed for PS spectrum
        :params energy: Energy where to evaluate the diffenrential spectrum
        :return: Differential flux
        """
        if self._spec_type == 'bb':

            return 1*energy**2/(np.expm1(energy/self._bb_temp))

        if self._spec_type == 'pl':
            e_norm = 1.0

            return 1 / (energy / e_norm) ** self._pl_index

    def _integral_ps(self, e1, e2):
        """
        Method to integrate the diff. flux over the Ebins of the incoming photons
        :params e1: lower bound of Ebin_in
        :params e2: upper bound of Ebin_in
        :params index: Index of spectrum
        :return: flux in the Ebin_in
        """
        return (
            (e2 - e1)
            / 6.0
            * (
                self._spectrum_ps(e1)
                + 4 * self._spectrum_ps((e1 + e2) / 2.0)
                + self._spectrum_ps(e2)
            )
        )

    @property
    def spec_type(self):
        return self._spec_type

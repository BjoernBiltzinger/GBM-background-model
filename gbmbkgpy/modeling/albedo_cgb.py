import numpy as np
from scipy.interpolate import interpolate

from gbmbkgpy.utils.progress_bar import progress_bar

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


class Albedo_CGB_free(object):
    """
    Class that precalulated the response arrays for the Earth Albedo and CGB for all times for which the geometry
    was calculated. Use this if you want the spectra to be free in the fit (not only normalization) 
    """

    def __init__(self, det_responses, geometry):

        self._detectors = list(det_responses.responses.keys())
        self._echans = det_responses.echans

        self._rsp = det_responses.responses
        self._geom = geometry

        self._calc_earth_cgb_responses()

    @property
    def cgb_effective_response(self):
        """
        Returns the precalulated effective response for the CGB for all times for which the geometry
        was calculated
        """
        return self._cgb_response_sums

    @property
    def earth_effective_response(self):
        """
        Returns the precalulated effective response for the Earth for all times for which the geometry
        was calculated
        """
        return self._earth_response_sums

    @property
    def responses(self):
        return self._rsp

    @property
    def interp_times(self):
        return self._geom.geometry_times

    def _calc_earth_cgb_responses(self):
        # Calculate the true flux for the Earth for the assumed spectral parameters (Normalization=1).
        # This true flux is binned in energy bins as defined in the response object

        cgb_response_sums = {}
        earth_response_sums = {}

        for det_idx, det in enumerate(self._detectors):
            (
                cgb_response_sums[det],
                earth_response_sums[det],
            ) = self._get_effective_response_albedo_cgb(det_response=self._rsp[det])

        self._cgb_response_sums = cgb_response_sums
        self._earth_response_sums = earth_response_sums

    def _get_effective_response_albedo_cgb(self, det_response):
        """
        Calculate the effective response sum for all interpolation times for which the geometry was
        calculated. This supports MPI to reduce the calculation time.
        To calculate the responses created on a grid in the response_object are used. All points
        that are not occulted by the earth are added
        """
        # define the opening angle of the earth in degree
        earth_radius = 6371.0
        fermi_radius = np.sqrt(np.sum(self._geom.sc_pos ** 2, axis=1))
        horizon_angle = 90 - np.rad2deg(np.arccos(earth_radius / fermi_radius))

        min_vis = np.deg2rad(horizon_angle)

        resp_grid_points = det_response.points

        sr_points = 4 * np.pi / len(resp_grid_points)

        # Calculate the normalization of the spacecraft position vectors
        earth_position_cart_norm = np.sqrt(
            np.sum(
                self._geom.earth_position_cart * self._geom.earth_position_cart, axis=1
            )
        ).reshape((len(self._geom.earth_position_cart), 1))

        # Calculate the normalization of the grid points of the response precalculation
        resp_grid_points_norm = np.sqrt(
            np.sum(resp_grid_points * resp_grid_points, axis=1)
        ).reshape((len(resp_grid_points), 1))

        tmp = np.clip(
            np.dot(
                self._geom.earth_position_cart / earth_position_cart_norm,
                resp_grid_points.T / resp_grid_points_norm.T,
            ),
            -1,
            1,
        )

        # Calculate separation angle between
        # spacecraft and earth horizon
        ang_sep = np.arccos(tmp)

        # Create a mask with True when the Earth is in the FOV
        # and False when its CGB
        earth_occultion_idx = np.less(ang_sep.T, min_vis).T

        # Sum up the responses that are occulted by earth in earth_effective_response
        # and the others in cgb_effective_response
        # then mulitiply by the sr_points factor which is the area
        # of the unit sphere covered by every point

        # TODO: Check why the earth occultation mask has to be inverted for the earth
        # and not the other way around!
        effective_response_earth = (
            np.tensordot(
                ~earth_occultion_idx, det_response.response_array, [(1,), (0,)]
            )
            * sr_points
        )

        effective_response_cgb = (
            np.tensordot(earth_occultion_idx, det_response.response_array, [(1,), (0,)])
            * sr_points
        )

        return effective_response_earth, effective_response_cgb


class Albedo_CGB_fixed(Albedo_CGB_free):
    """
    Class that precalulated the rates arrays for the Earth Albedo and CGB for all times for which the geometry
    was calculated for a normalization 1. Use this if you want that only the normalization of the spectra
    is a free fit parameter.
    """

    def __init__(self, det_responses, geometry, earth_dict=None, cgb_dict=None):
        super(Albedo_CGB_fixed, self).__init__(det_responses, geometry)

        # Set spectral parameters to literature values (Earth and CGB from Ajello)

        if cgb_dict:
            self._index1_cgb = cgb_dict["spectrum"]["alpha"]
            self._index2_cgb = cgb_dict["spectrum"]["beta"]
            self._break_energy_cgb = cgb_dict["spectrum"]["Eb"]
            self._norm_cgb = cgb_dict["spectrum"]["norm"]
        else:
            # cgb spectrum
            self._index1_cgb = 1.32
            self._index2_cgb = 2.88
            self._break_energy_cgb = 29.99
            self._norm_cgb = 0.1

        if earth_dict:
            # earth spectrum
            self._index1_earth = earth_dict["spectrum"]["alpha"]
            self._index2_earth = earth_dict["spectrum"]["beta"]
            self._break_energy_earth = earth_dict["spectrum"]["Eb"]
            self._norm_earth = earth_dict["spectrum"]["norm"]
        else:
            # earth spectrum
            self._index1_earth = -5
            self._index2_earth = 1.72
            self._break_energy_earth = 33.7
            self._norm_earth = 0.015
        self._calc_earth_cgb_rates()

        self._interpolate_earth_cgb_rates()

    def get_earth_rates(self, met):
        """
        Returns an array with the predicted count rates for the times for which the geometry 
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later) 
        and the fixed spectral parameters defined above.
        """
        earth_rates = self._interp_rate_earth(met)

        # The interpolated rate has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        earth_rates = np.swapaxes(earth_rates, 1, 2)
        earth_rates = np.swapaxes(earth_rates, 2, 3)

        return earth_rates

    def get_cgb_rates(self, met):
        """
        Returns an array with the predicted count rates for the times for which the geometry 
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later) 
        and the fixed spectral parameters defined above.
        """

        cgb_rates = self._interp_rate_cgb(met)

        # The interpolated rate has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we net to swap axes
        # The 2 is the start stop in the time_bins

        cgb_rates = np.swapaxes(cgb_rates, 1, 2)
        cgb_rates = np.swapaxes(cgb_rates, 2, 3)

        return cgb_rates

    def _calc_earth_cgb_rates(self):
        # Calculate the true flux for the Earth for the assumed spectral parameters (Normalization=1).
        # This true flux is binned in energy bins as defined in the response object

        folded_flux_cgb = np.zeros(
            (len(self._geom.geometry_times), len(self._detectors), len(self._echans),)
        )

        folded_flux_earth = np.zeros(
            (len(self._geom.geometry_times), len(self._detectors), len(self._echans),)
        )

        for det_idx, det in enumerate(self._detectors):
            true_flux_cgb = self._integral_cgb(
                self._rsp[det].Ebin_in_edge[:-1], self._rsp[det].Ebin_in_edge[1:]
            )

            true_flux_earth = self._integral_earth(
                self._rsp[det].Ebin_in_edge[:-1], self._rsp[det].Ebin_in_edge[1:]
            )

            folded_flux_cgb[:, det_idx, :] = np.dot(
                true_flux_cgb, self._cgb_response_sums[det]
            )

            folded_flux_earth[:, det_idx, :] = np.dot(
                true_flux_earth, self._earth_response_sums[det]
            )

        self._folded_flux_cgb = self._norm_cgb*folded_flux_cgb
        self._folded_flux_earth = self._norm_earth*folded_flux_earth

    @property
    def responses(self):
        return self._rsp

    def _interpolate_earth_cgb_rates(self):
        # interp_rate_earth = {}
        # interp_rate_cgb = {}

        self._interp_rate_cgb = interpolate.interp1d(
            self._geom.geometry_times, self._folded_flux_cgb, axis=0
        )

        self._interp_rate_earth = interpolate.interp1d(
            self._geom.geometry_times, self._folded_flux_earth, axis=0
        )

    def _spectrum_bpl(self, energy, C, index1, index2, break_energy):
        """
        Define the function of a broken power law. Needed for earth and cgb spectrum
        :param energy:
        :param C:
        :param index1:
        :param index2:
        :param break_energy:
        :return:
        """
        return C / (
            (energy / break_energy) ** index1 + (energy / break_energy) ** index2
        )

    def _differential_flux_earth(self, e):
        """
        Calculate the diff. flux with the constants defined for the Earth
        :param e: Energy of incoming photon
        :return: differential flux
        """
        C = 1  # set the constant=1 will be fitted later to fit the data best
        return self._spectrum_bpl(
            e, C, self._index1_earth, self._index2_earth, self._break_energy_earth
        )

    def _integral_earth(self, e1, e2):
        """
        Method to integrate the diff. flux over the Ebins of the incoming photons
        :param e1: lower bound of Ebin_in
        :param e2: upper bound of Ebin_in
        :return: flux in the Ebin_in
        """
        return (
            (e2 - e1)
            / 6.0
            * (
                self._differential_flux_earth(e1)
                + 4 * self._differential_flux_earth((e1 + e2) / 2.0)
                + self._differential_flux_earth(e2)
            )
        )

    def _differential_flux_cgb(self, e):
        """
        Same as for Earth, just other constants
        :param e: Energy
        :return: differential flux
        """
        C = 1  # set the constant=1 will be fitted later to fit the data best
        return self._spectrum_bpl(
            e, C, self._index1_cgb, self._index2_cgb, self._break_energy_cgb
        )

    def _integral_cgb(self, e1, e2):
        """
        same as for earth
        :param e1: lower bound of Ebin_in
        :param e2: upper bound of Ebin_in
        :return: flux in the Ebin_in
        """
        return (
            (e2 - e1)
            / 6.0
            * (
                self._differential_flux_cgb(e1)
                + 4 * self._differential_flux_cgb((e1 + e2) / 2.0)
                + self._differential_flux_cgb(e2)
            )
        )

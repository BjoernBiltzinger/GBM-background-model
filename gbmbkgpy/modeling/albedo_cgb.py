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

    def __init__(self, det_responses, det_geometries):
        assert det_responses.responses.keys() == det_geometries.geometries.keys(), \
            'The detectors in the response object do not match the detectors in the geometry object'

        self._detectors = list(det_responses.responses.keys())
        self._echans = det_responses.echans

        self._rsp = det_responses.responses
        self._geom = det_geometries.geometries

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

    def _calc_earth_cgb_responses(self):
        # Calculate the true flux for the Earth for the assumed spectral parameters (Normalization=1).
        # This true flux is binned in energy bins as defined in the response object

        cgb_response_sums = {}
        earth_response_sums = {}

        for det_idx, det in enumerate(self._detectors):
            cgb_response_sums[det], earth_response_sums[det] = self._response_sum_one_det(
                det_response=self._rsp[det],
                det_geometry=self._geom[det]
            )

        self._cgb_response_sums = cgb_respons_sums
        self._earth_response_sums = earth_response_sums

    def _response_sum_one_det(self, det_response, det_geometry):
        """
        Calculate the effective response sum for all interpolation times for which the geometry was 
        calculated. This supports MPI to reduce the calculation time.
        To calculate the responses created on a grid in the response_object are used. All points 
        that are not occulted by the earth are added to the cgb and all others to the earth.
        """

        # Get the precalulated points and responses on the unit sphere
        points = det_response.points
        responses = det_response.responses

        # Factor to multiply the responses with. Needed as the later spectra are given in units of
        # 1/sr. The sr_points gives the area of the sphere occulted by one point
        sr_points = 4 * np.pi / len(points)

        # Get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []

        # If MPI available it is used to speed up the calculation
        if using_mpi:
            num_times = len(self._geom.earth_zen)
            times_per_rank = float(num_times) / float(size)
            times_lower_bound_index = int(np.floor(rank * times_per_rank))
            times_upper_bound_index = int(np.floor((rank + 1) * times_per_rank))

            if rank == 0:
                with progress_bar(times_upper_bound_index - times_lower_bound_index,
                                  title='Calculating earth position in sat frame for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for i in range(times_lower_bound_index, times_upper_bound_index):
                        earth_pos_inter_times.append(
                            np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                      * np.cos(det_geometry.earth_az[i] * (np.pi / 180)),
                                      np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                      * np.sin(det_geometry.earth_az[i] * (np.pi / 180)),
                                      np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
                        p.increase()
            else:
                for i in range(times_lower_bound_index, times_upper_bound_index):
                    earth_pos_inter_times.append(
                        np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                  * np.cos(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                  * np.sin(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
            earth_pos_inter_times = np.array(earth_pos_inter_times)

            # Define the opening angle of the earth in degree
            opening_angle_earth = 67

            # Initalize the lists for the summed effective responses
            array_cgb_response_sum = []
            array_earth_response_sum = []

            # Add the responses that are occulted by earth to earth effective response
            # and the others to CGB effective response. Do this for all times for which the
            # geometry was calculated
            if rank == 0:
                with progress_bar(len(earth_pos_inter_times),
                                  title='Calculating the effective response for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for pos in earth_pos_inter_times:
                        angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                        cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                        earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                        array_cgb_response_sum.append(cgb_response_time)
                        array_earth_response_sum.append(earth_response_time)
                        p.increase()
            else:
                for pos in earth_pos_inter_times:
                    angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                    cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                    earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                    array_cgb_response_sum.append(cgb_response_time)
                    array_earth_response_sum.append(earth_response_time)
                    p.increase()

            # Collect all results in rank=0 and broadcast it to all ranks
            # in the end
            array_cgb_response_sum = np.array(array_cgb_response_sum)
            array_earth_response_sum = np.array(array_earth_response_sum)
            array_cgb_response_sum_g = comm.gather(array_cgb_response_sum, root=0)
            array_earth_response_sum_g = comm.gather(array_earth_response_sum, root=0)
            if rank == 0:
                array_cgb_response_sum_g = np.concatenate(array_cgb_response_sum_g)
                array_earth_response_sum_g = np.concatenate(array_earth_response_sum_g)
            array_cgb_response_sum = comm.bcast(array_cgb_response_sum_g, root=0)
            array_earth_response_sum = comm.bcast(array_earth_response_sum_g, root=0)
        else:

            # The same as above just for single core calculation without MPI
            with progress_bar(len(det_geometry.earth_zen),
                              title='Calculating earth position in sat frame for several times') as p:

                for i in range(0, len(det_geometry.earth_zen)):
                    earth_pos_inter_times.append(
                        np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180)) * np.cos(
                            det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.cos(det_geometry.earth_zen[i] * (np.pi / 180)) * np.sin(
                                      det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
                    p.increase()
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)

            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []

            with progress_bar(len(earth_pos_inter_times),
                              title='Calculating the effective response for several times.') as p:
                for pos in self._earth_pos_inter_times:
                    angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                    cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                    earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                    array_cgb_response_sum.append(cgb_response_time)
                    array_earth_response_sum.append(earth_response_time)
                    p.increase()

        # Mulitiply by the sr_points factor which is the area of the unit sphere covered by every point
        array_cgb_response_sum = np.array(array_cgb_response_sum) * sr_points
        array_earth_response_sum = np.array(array_earth_response_sum) * sr_points
        return array_cgb_response_sum, array_earth_response_sum

class Albedo_CGB_fixed(object):
    """
    Class that precalulated the rates arrays for the Earth Albedo and CGB for all times for which the geometry
    was calculated for a normalization 1. Use this if you want that only the normalization of the spectra
    is a free fit parameter.
    """

    def __init__(self, det_responses, det_geometries):

        assert det_responses.responses.keys() == det_geometries.geometries.keys(), \
            'The detectors in the response object do not match the detectors in the geometry object'

        self._detectors = list(det_responses.responses.keys())
        self._echans = det_responses.echans

        self._rsp = det_responses.responses
        self._geom = det_geometries.geometries

        # Set spectral parameters to literature values (Earth and CGB from Ajello)

        # cgb spectrum
        self._index1_cgb = 1.32
        self._index2_cgb = 2.88
        self._break_energy_cgb = 29.99

        # earth spectrum
        self._index1_earth = -5
        self._index2_earth = 1.72
        self._break_energy_earth = 33.7

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

        # cgb_rates = np.zeros((
        #     len(met),
        #     len(self._detectors),
        #     len(self._echans),
        #     2
        # ))

        # for det_idx, det in enumerate(self._detectors):
        #     interpolated_cgb_rate = self._interp_rate_cgb[det](met)

        #     # The interpolated rate has the dimensions nr_echans, nr_time_bins, 2
        #     # So we swap zeroth and first axes to get nr_time_bins, nr_echans, 2

        #     cgb_rates[:, det_idx, :, :] = np.swapaxes(interpolated_cgb_rate, 0, 1)

        return cgb_rates

    def _calc_earth_cgb_rates(self):
        # Calculate the true flux for the Earth for the assumed spectral parameters (Normalization=1).
        # This true flux is binned in energy bins as defined in the response object

        folded_flux_cgb = np.zeros((
            len(self._geom[self._detectors[0]].time),
            len(self._detectors),
            len(self._echans),
        ))

        folded_flux_earth = np.zeros((
            len(self._geom[self._detectors[0]].time),
            len(self._detectors),
            len(self._echans),
        ))

        for det_idx, det in enumerate(self._detectors):
            true_flux_cgb = self._integral_cgb(
                self._rsp[det].Ebin_in_edge[:-1],
                self._rsp[det].Ebin_in_edge[1:]
            )

            true_flux_earth = self._integral_earth(
                self._rsp[det].Ebin_in_edge[:-1],
                self._rsp[det].Ebin_in_edge[1:]
            )

            cgb_response_sum, earth_response_sum = self._response_sum_one_det(
                det_response=self._rsp[det],
                det_geometry=self._geom[det]
            )


            folded_flux_cgb[:, det_idx, :] = np.dot(true_flux_cgb, cgb_response_sum)
            folded_flux_earth[:, det_idx, :] = np.dot(true_flux_earth, earth_response_sum)

        self._folded_flux_cgb = folded_flux_cgb
        self._folded_flux_earth = folded_flux_earth

    @property
    def geometry_time(self):
        return self._geom[self._detectors[0]].time

    def _interpolate_earth_cgb_rates(self):
        # interp_rate_earth = {}
        # interp_rate_cgb = {}

        self._interp_rate_cgb = interpolate.interp1d(
            self.geometry_time,
            self._folded_flux_cgb,
            axis=0
        )

        self._interp_rate_earth = interpolate.interp1d(
            self.geometry_time,
            self._folded_flux_earth,
            axis=0
        )


        # for det_idx, det in enumerate(self._detectors):

        #     interp_rate_cgb[det] = interpolate.interp1d(
        #         self._geom[det].time,
        #         self._folded_flux_cgb[:, det_idx, :].T
        #     )

        #     interp_rate_earth[det] = interpolate.interp1d(
        #         self._geom[det].time,
        #         self._folded_flux_earth[:, Det_idx, :].T
        #     )

        # self._interp_rate_cgb = interp_rate_cgb
        # self._interp_rate_earth = interp_rate_earth


    def _response_sum_one_det(self, det_response, det_geometry):
        """
        Calculate the effective response sum for all interpolation times for which the geometry was 
        calculated. This supports MPI to reduce the calculation time.
        To calculate the responses created on a grid in the response_object are used. All points 
        that are not occulted by the earth are added
        """

        # Get the precalulated points and responses on the unit sphere
        points = det_response.points
        responses = det_response.response_array

        # Factor to multiply the responses with. Needed as the later spectra are given in units of
        # 1/sr. The sr_points gives the area of the sphere occulted by one point
        sr_points = 4 * np.pi / len(points)

        # Get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []

        # If MPI available it is used to speed up the calculation
        if using_mpi:
            num_times = len(det_geometry.earth_zen)
            times_per_rank = float(num_times) / float(size)
            times_lower_bound_index = int(np.floor(rank * times_per_rank))
            times_upper_bound_index = int(np.floor((rank + 1) * times_per_rank))

            if rank == 0:
                with progress_bar(times_upper_bound_index - times_lower_bound_index,
                                  title='Calculating earth position in sat frame for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for i in range(times_lower_bound_index, times_upper_bound_index):
                        earth_pos_inter_times.append(
                            np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                      * np.cos(det_geometry.earth_az[i] * (np.pi / 180)),
                                      np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                      * np.sin(det_geometry.earth_az[i] * (np.pi / 180)),
                                      np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
                        p.increase()
            else:
                for i in range(times_lower_bound_index, times_upper_bound_index):
                    earth_pos_inter_times.append(
                        np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                  * np.cos(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.cos(det_geometry.earth_zen[i] * (np.pi / 180))
                                  * np.sin(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
            earth_pos_inter_times = np.array(earth_pos_inter_times)

            # Define the opening angle of the earth in degree
            opening_angle_earth = 67

            # Initalize the lists for the summed effective responses
            array_cgb_response_sum = []
            array_earth_response_sum = []

            # Add the responses that are occulted by earth to earth effective response
            # and the others to CGB effective response. Do this for all times for which the
            # geometry was calculated
            if rank == 0:
                with progress_bar(len(earth_pos_inter_times),
                                  title='Calculating the effective response for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for pos in earth_pos_inter_times:
                        angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                        cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                        earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                        array_cgb_response_sum.append(cgb_response_time)
                        array_earth_response_sum.append(earth_response_time)
                        p.increase()
            else:
                for pos in earth_pos_inter_times:
                    angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                    cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                    earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                    array_cgb_response_sum.append(cgb_response_time)
                    array_earth_response_sum.append(earth_response_time)

            # Collect all results in rank=0 and broadcast it to all ranks
            # in the end
            array_cgb_response_sum = np.array(array_cgb_response_sum)
            array_earth_response_sum = np.array(array_earth_response_sum)
            array_cgb_response_sum_g = comm.gather(array_cgb_response_sum, root=0)
            array_earth_response_sum_g = comm.gather(array_earth_response_sum, root=0)
            if rank == 0:
                array_cgb_response_sum_g = np.concatenate(array_cgb_response_sum_g)
                array_earth_response_sum_g = np.concatenate(array_earth_response_sum_g)
            array_cgb_response_sum = comm.bcast(array_cgb_response_sum_g, root=0)
            array_earth_response_sum = comm.bcast(array_earth_response_sum_g, root=0)
        else:

            # The same as above just for single core calculation without MPI
            with progress_bar(len(det_geometry.earth_zen),
                              title='Calculating earth position in sat frame for several times.') as p:
                for i in range(0, len(det_geometry.earth_zen)):
                    earth_pos_inter_times.append(
                        np.array([np.cos(det_geometry.earth_zen[i] * (np.pi / 180)) * np.cos(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.cos(det_geometry.earth_zen[i] * (np.pi / 180)) * np.sin(det_geometry.earth_az[i] * (np.pi / 180)),
                                  np.sin(det_geometry.earth_zen[i] * (np.pi / 180))]))
                    p.increase()
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)

            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []

            with progress_bar(len(earth_pos_inter_times),
                              title='Calculating the effective response for several times.') as p:
                for pos in earth_pos_inter_times:
                    angle_earth = np.arccos(np.dot(pos, points.T)) * (180 / np.pi)

                    cgb_response_time = np.sum(responses[np.where(angle_earth > opening_angle_earth)], axis=0)

                    earth_response_time = np.sum(responses[np.where(angle_earth < opening_angle_earth)], axis=0)

                    array_cgb_response_sum.append(cgb_response_time)
                    array_earth_response_sum.append(earth_response_time)
                    p.increase()

        # Mulitiply by the sr_points factor which is the area of the unit sphere covered by every point
        array_cgb_response_sum = np.array(array_cgb_response_sum) * sr_points
        array_earth_response_sum = np.array(array_earth_response_sum) * sr_points

        return array_cgb_response_sum, array_earth_response_sum

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
        return C / ((energy / break_energy) ** index1 + (energy / break_energy) ** index2)

    def _differential_flux_earth(self, e):
        """
        Calculate the diff. flux with the constants defined for the Earth
        :param e: Energy of incoming photon
        :return: differential flux
        """
        C = 1  # set the constant=1 will be fitted later to fit the data best
        return self._spectrum_bpl(e, C, self._index1_earth, self._index2_earth, self._break_energy_earth)

    def _integral_earth(self, e1, e2):
        """
        Method to integrate the diff. flux over the Ebins of the incoming photons
        :param e1: lower bound of Ebin_in
        :param e2: upper bound of Ebin_in
        :return: flux in the Ebin_in
        """
        return (e2 - e1) / 6.0 * (
                self._differential_flux_earth(e1) + 4 * self._differential_flux_earth((e1 + e2) / 2.0) +
                self._differential_flux_earth(e2))

    def _differential_flux_cgb(self, e):
        """
        Same as for Earth, just other constants
        :param e: Energy
        :return: differential flux
        """
        C = 1  # set the constant=1 will be fitted later to fit the data best
        return self._spectrum_bpl(e, C, self._index1_cgb, self._index2_cgb, self._break_energy_cgb)

    def _integral_cgb(self, e1, e2):
        """
        same as for earth
        :param e1: lower bound of Ebin_in
        :param e2: upper bound of Ebin_in
        :return: flux in the Ebin_in
        """
        return (e2 - e1) / 6.0 * (
                self._differential_flux_cgb(e1) + 4 * self._differential_flux_cgb((e1 + e2) / 2.0) +
                self._differential_flux_cgb(e2))

from gbmbkgpy.utils.progress_bar import progress_bar
import astropy.coordinates as coord
from gbm_drm_gen.drmgen import DRMGen
import numpy as np

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False 

class Sun(object):

    def __init__(self, response_object, geometry_object, echan_list):
        """
        Initalize the Sun as source with a free powerlaw and precalcule the response for all times
        for which the geometry was calculated.
        :params response_object: response_precalculation object
        :params geometry_object: geomerty precalculatation object
        :params echan_list: which echan
        """
        self._geom = geometry_object
        self._rsp = response_object
        self._data_type = self._rsp.data_type

        if self._data_type == 'ctime' or self._data_type == 'trigdat':
            self._echan_mask = np.zeros(8, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        elif self._data_type == 'cspec':
            self._echan_mask = np.zeros(128, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        self._response_array()

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

    def _response_array(self):
        """
        Funtion that imports and precalculate everything that is needed to get the point source array
        for all echans
        :return:
        """

        # Import the quaternion, sc_pos and earth_position (as SkyCoord object) from the geometry_object
        sun_positions = self._geom.sun_positions
        earth_positions = self._geom.earth_position

        # Import the points of the grid around the detector from the response_object
        Ebin_in_edge = self._rsp.Ebin_in_edge
        Ebin_out_edge = self._rsp.Ebin_out_edge
        det = self._rsp.det

        # Use Mpi when it is available
        if using_mpi:
            num_times = len(self._geom.earth_zen)
            times_per_rank = float(num_times) / float(size)
            times_lower_bound_index = int(np.floor(rank * times_per_rank))
            times_upper_bound_index = int(np.floor((rank + 1) * times_per_rank))
            sun_pos_sat_list = []
            if rank == 0:
                with progress_bar(times_upper_bound_index - times_lower_bound_index,
                                  title='Calculate sun azimuth and zenith angle in satelite frame.') as p:
                    for i in range(times_lower_bound_index, times_upper_bound_index):
                        az = sun_positions[i].lon.deg
                        zen = sun_positions[i].lat.deg
                        sun_pos_sat_list.append([np.cos(zen * (np.pi / 180)) *
                                                np.cos(az * (np.pi / 180)),
                                                np.cos(zen * (np.pi / 180)) *
                                                np.sin(az * (np.pi / 180)),
                                                np.sin(zen * (np.pi / 180))])
                        p.increase()

            else:
                for i in range(times_lower_bound_index, times_upper_bound_index):
                    az = sun_positions[i].lon.deg
                    zen = sun_positions[i].lat.deg
                    sun_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                            np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                            np.sin(zen * (np.pi / 180))])

            sun_pos_sat_list = np.array(sun_pos_sat_list)

            # Calcutate the response for the different ps locations

            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            sun_response = []
            if rank == 0:
                with progress_bar(len(sun_pos_sat_list),
                                  title='Calculating the response for all Sun positions in sat frame.This shows the progress of rank 0. All other should be about the same.') as p:
                    for point in sun_pos_sat_list:
                        matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                        sun_response.append(matrix.T)
                    p.increase()
            else:
                for point in sun_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    sun_response.append(matrix.T)

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for i, earth_position in enumerate(earth_positions):
                separation.append(coord.SkyCoord.separation(sun_positions[i], earth_position).value)
            separation = np.array(separation)

            # define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than sun is behind earth)
            if rank==0:
                with progress_bar(len(sun_pos_sat_list),
                                  title='Checking earth occultation for sun.') as p:
                    for i in range(len(sun_response)):
                        # Check if not occulted by earth
                        if separation[i] < earth_opening_angle:
                            # If occulted by earth set response to zero
                            sun_response[i] = sun_response[i] * 0
            else:
                for i in range(len(sun_response)):
                    # Check if not occulted by earth
                    if separation[i] < earth_opening_angle:
                        # If occulted by earth set response to zero
                        sun_response[i] = sun_response[i] * 0

            # Gather all results in rank=0 and broadcast the final result to all ranks
            sun_response = np.array(sun_response)
            sun_response_g = comm.gather(sun_response, root=0)


            separation = np.array(separation)
            separation_g = comm.gather(separation, root=0)

            if rank == 0:
                sun_response_g = np.concatenate(sun_response_g)
                separation_g = np.concatenate(separation_g)
            sun_response = comm.bcast(sun_response_g, root=0)
            separation = comm.bcast(separation_g, root=0)

        # Singlecore calculation
        else:

            # Get the postion of the PS in the sat frame for every timestep
            sun_pos_sat_list = []
            with progress_bar(len(sun_positions),
                              title='Calculating Sun azimuth and zenith angles') as p:
                for i in range(0, len(sun_positions)):
                    az = sun_positions[i].lon.deg
                    zen = sun_positions[i].lat.deg
                    sun_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                            np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                            np.sin(zen * (np.pi / 180))])
                    p.increase()

            sun_pos_sat_list = np.array(sun_pos_sat_list)
            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            sun_response = []
            with progress_bar(len(sun_pos_sat_list),
                              title='Calculating the response for all Sun positions in sat frame.This shows the progress of rank 0. All other should be about the same.') as p:
                for point in sun_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    sun_response.append(matrix.T)
                p.increase()

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for i, earth_position in enumerate(earth_positions):
                separation.append(coord.SkyCoord.separation(sun_positions[i], earth_position).value)
            separation = np.array(separation)

            # Define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than ps is behind earth)
            for i in range(len(sun_response)):
                # Check if not occulted by earth
                if separation[i] < earth_opening_angle:
                    # If occulted by earth set response to zero
                    sun_response[i] = sun_response[i] * 0
        self._separation = separation
        self._sun_response = np.array(sun_response)

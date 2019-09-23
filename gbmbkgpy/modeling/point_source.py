import astropy.coordinates as coord
import astropy.units as u
from gbmgeometry.gbm_frame import GBMFrame
from gbm_drm_gen.drmgen import DRMGen
from gbmbkgpy.utils.progress_bar import progress_bar


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

class PointSrc_fixed(object):

    def __init__(self, name, ra, dec, response_object, geometry_object, echan_list, index=2.114):
        """
        Initialize a PS and precalculates the rates for all the times for which the geomerty was
        calculated.

        :params name: Name of PS
        :params ra: ra position of PS (J2000)
        :params dec: dec position of PS (J2000)
        :params response_object: response_precalculation object
        :params geometry_object: geomerty precalculatation object
        :params index: Powerlaw index of PS spectrum
        """
        self._name = name

        # Build a SkyCoord object of the PS 
        self._ps_skycoord = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

        self._rsp = response_object
        self._geom = geometry_object
        self._data_type = self._rsp.data_type

        if self._data_type == 'ctime':
            self._echan_mask = np.zeros(8, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        elif self._data_type == 'cspec':
            self._echan_mask = np.zeros(128, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        
        self._response_array()
        self._rate_array(index=index)

    @property
    def skycoord(self):
        """
        Returns the SkyCoord object of the PS
        """
        return self._ps_skycoord

    @property
    def ps_rate_array(self):
        """
        Returns an array with the predicted count rates for the times for which the geometry
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later)
        and the fixed spectral index defined in the init of the object.
        """
        return self._folded_flux_ps

    @property
    def geometry_times(self):

        return self._geom.time

    @property
    def Ebin_in_edge(self):
        """
        Returns the Ebin_in edges as defined in the response object
        """
        return self._rsp.Ebin_in_edge

    def _rate_array(self, index=2):
        """
        Calaculates the rate in all energy channels for all times for which the geometry was calculated.
        Uses the responses calculated in _response_array.
        :param index: Index of powerlaw
        """
        true_flux_ps = self._integral_ps(self._rsp.Ebin_in_edge[:-1], self._rsp.Ebin_in_edge[1:], index)
        self._folded_flux_ps = np.dot(true_flux_ps, self._ps_response)
        
            
    def _response_array(self):
        """
        Funtion that imports and precalculate everything that is needed to get the point source array 
        for all echans
        :return:
        """

        # Import the quaternion, sc_pos and earth_position (as SkyCoord object) from the geometry_object
        quaternion = self._geom.quaternion
        sc_pos = self._geom.sc_pos
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
            # Calcutate the GBMFrame for all the times for which the geomerty was calcutated
            GBMFrame_list = []
            if rank == 0:
                
                with progress_bar(times_upper_bound_index-times_lower_bound_index,
                                  title='Calculating GBM frame for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for i in range(times_lower_bound_index, times_upper_bound_index):
                        q1, q2, q3, q4 = quaternion[i]
                        scx, scy, scz = sc_pos[i]
                        GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                      quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                      sc_pos_Z=scz))

                        p.increase()
            else:
                for i in range(times_lower_bound_index, times_upper_bound_index):
                    q1, q2, q3, q4 = quaternion[i]
                    scx, scy, scz = sc_pos[i]
                    GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                  quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                  sc_pos_Z=scz))
            GBMFrame_list = np.array(GBMFrame_list)

            # Get the postion of the PS in the satellite frame (saved as vector and as SkyCoord object)
            ps_pos_sat_list = []
            ps_pos_sat_objects = []
            if rank == 0:
                with progress_bar(len(GBMFrame_list),
                                  title='Calculating PS position in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(self._name)) as p:
                    for i in range(0, len(GBMFrame_list)):
                        ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                        ps_pos_sat_objects.append(ps_pos_sat)
                        az = ps_pos_sat.lon.deg
                        zen = ps_pos_sat.lat.deg
                        ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) *
                                                np.cos(az * (np.pi / 180)),
                                                np.cos(zen * (np.pi / 180)) *
                                                np.sin(az * (np.pi / 180)),
                                                np.sin(zen * (np.pi / 180))])
                        p.increase()

            else:
                for i in range(0, len(GBMFrame_list)):
                    ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                    ps_pos_sat_objects.append(ps_pos_sat)
                    az = ps_pos_sat.lon.deg
                    zen = ps_pos_sat.lat.deg
                    ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                                  np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                                  np.sin(zen * (np.pi / 180))])
            ps_pos_sat_list = np.array(ps_pos_sat_list)
            ps_pos_sat_objects = np.array(ps_pos_sat_objects)

            # Calcutate the response for the different ps locations

            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            ps_response = []
            if rank == 0:
                with progress_bar(len(ps_pos_sat_list),
                                  title='Calculating the response for all PS positions in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(self._name)) as p:
                    for point in ps_pos_sat_list:
                        matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                        ps_response.append(matrix.T)
                    p.increase()
            else:
                for point in ps_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    ps_response.append(matrix.T)

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            separation = np.array(separation)
            
            # define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than ps is behind earth)
            for i in range(len(ps_response)):
                # Check if not occulted by earth
                if separation[i] < earth_opening_angle:
                    # If occulted by earth set response to zero
                    ps_response[i] = ps_response[i]*0

            # Gather all results in rank=0 and broadcast the final result to all ranks
            ps_response=np.array(ps_response)
            ps_response_g = comm.gather(ps_response, root=0)

            ps_pos_sat_objects = np.array(ps_pos_sat_objects)
            ps_pos_sat_objects_g = comm.gather(ps_pos_sat_objects, root=0)

            separation = np.array(separation)
            separation_g = comm.gather(separation, root=0)

            if rank == 0:
                ps_response_g = np.concatenate(ps_response_g)
                separation_g = np.concatenate(separation_g)
                ps_pos_sat_objects_g = np.concatenate(ps_pos_sat_objects_g)
            ps_response = comm.bcast(ps_response_g, root=0)
            separation = comm.bcast(separation_g, root=0)
            self._ps_pos_sat_objects = comm.bcast(ps_pos_sat_objects_g, root=0)

        # Singlecore calculation
        else:
            
            # Calcutate the GBMFrame for all these times
            GBMFrame_list = []
            with progress_bar(len(self._geom.earth_zen),
                              title='Calculating GBM frame for several times. '
                              'This shows the progress of rank 0. All other should be about the same.') as p:
                for i in range(0, len(quaternion)):
                    q1, q2, q3, q4 = quaternion[i]
                    scx, scy, scz = sc_pos[i]
                    GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                  quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                  sc_pos_Z=scz))
                p.increase()
            GBMFrame_list = np.array(GBMFrame_list)

            # Get the postion of the PS in the sat frame for every timestep
            ps_pos_sat_list = []
            with progress_bar(len(GBMFrame_list),
                              title='Calculating PS position in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                  self._name)) as p:
                for i in range(0, len(GBMFrame_list)):
                    ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                    az = ps_pos_sat.lon.deg
                    zen = ps_pos_sat.lat.deg
                    ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                                  np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                                  np.sin(zen * (np.pi / 180))])
                    p.increase()

            ps_pos_sat_list = np.array(ps_pos_sat_list)
            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            ps_response = []
            with progress_bar(len(ps_pos_sat_list),
                              title='Calculating the response for all PS positions in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                  self._name)) as p:
                for point in ps_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    ps_response.append(matrix.T)
                p.increase()

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            separation = np.array(separation)

            # Define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than ps is behind earth)
            for i in range(len(ps_response)):
                # Check if not occulted by earth
                if separation[i] < earth_opening_angle:
                    # If occulted by earth set response to zero
                    ps_response[i] = ps_response[i]*0
        self._separation = separation
        self._ps_response = np.array(ps_response)
        

    def _spectrum_ps(self, energy, C, index):
        """
        Define the function of a power law. Needed for PS spectrum
        :params energy:
        :params C:
        :params index:
        :return:
        """
        return C / energy**index
    
    def _differential_flux_ps(self, e, index):
        """
        Calculate the diff. flux with the constants defined for the earth
        :params e: Energy of incoming photon
        :params index: Index of spectrum 
        :return: differential flux
        """
        C = 1  # Set the Constant=1. It will be fitted later to fit the data best
        return self._spectrum_ps(e, C, index)

    def _integral_ps(self, e1, e2, index):
        """
        Method to integrate the diff. flux over the Ebins of the incoming photons
        :params e1: lower bound of Ebin_in
        :params e2: upper bound of Ebin_in
        :params index: Index of spectrum
        :return: flux in the Ebin_in
        """
        return (e2 - e1) / 6.0 * (self._differential_flux_ps(e1, index) +
                                  4 * self._differential_flux_ps((e1 + e2) / 2.0, index) +
                                  self._differential_flux_ps(e2, index))    

    def _calc_src_occ(self):
        """

        :return:
        """

        src_occ_ang = []
        earth_positions = self._data.earth_position #type: ContinuousData.earth_position

        with progress_bar(len(self._interpolation_time) - 1, title='Calculating earth occultation of point source') as p:
            for earth_position in earth_positions:
                src_occ_ang.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)

                p.increase()

        self._src_occ_ang = np.array(src_occ_ang)

        self._occulted_time = np.array(self._interpolation_time)

        self._occulted_time[np.where(self._src_occ_ang != 0)] = 0

        self._point_source_earth_angle_interpolator = interpolate.interp1d(self._interpolation_time, src_occ_ang)


        del src_occ_ang, earth_positions


    def _zero(self):
        print( "Numpy where condition true")
        return 0


    def _set_relative_location(self):

        """
        look at continous data sun stuff (setup_geometry)

        coordinate is _pointing of the detectors
        calculate seperation of detectors and point source
        store in arrays for time bins / sub sample time

        interpolate and create a function

        """

        sep_angle = []
        pointing = self._data.pointing #type: ContinuousData.pointing

        # go through a subset of times and calculate the sun angle with GBM geometry

        with progress_bar(len(self._interpolation_time)-1,title='Calculating point source seperation angles') as p:

            for point in pointing:
                sep_angle.append(coord.SkyCoord.separation(self._ps_skycoord,point).value)

                p.increase()

        # interpolate it
        self._point_source_interpolator = interpolate.interp1d(self._interpolation_time, sep_angle)

        del sep_angle, pointing


    def calc_occ_array(self, time_bins):
        """

        :param time_bins:
        :return:
        """
        self._src_ang_bin = np.array(self._point_source_interpolator(time_bins))
        self._src_ang_bin[np.where(self._src_occ_ang == 0)] = 0.

        return self._src_ang_bin

    def earth_occ_of_ps(self,mean_time): #mask for ps behind earth
        """
        Calculates a mask that is 0 for all time_bins in which the PS is behind the earth and 1 if not
        :param mean_time:
        :return:
        """
        # define the size of the earth
        earth_radius = 6371000.8  # geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
        atmosphere = 12000.  # the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere
        sat_dist = 6912000.
        earth_opening_angle = math.asin(r / sat_dist) * 360. / (2. * math.pi)  # earth-cone
        mask = np.zeros_like(mean_time)
        mask[np.where(np.array(self._point_source_earth_angle_interpolator(mean_time)) > earth_opening_angle)] = 1
        return mask

    def separation_angle(self, met):
        """
        call interpolated function and return separation for met (mid eval time)
        """
        return self._point_source_interpolator(met)

    def _cleanup(self):
        del self._interpolation_time, self._src_occ_ang

    @property
    def location(self):

        return self._ps_skycoord

    @property
    def src_ang_bin(self):

        return self._src_ang_bin


    @property
    def name(self):

        return self._name

    @property
    def separation(self):

        return self._separation

    @property
    def ps_pos_sat_objects(self):

        return self._ps_pos_sat_objects


class PointSrc_free(object):

    def __init__(self, name, ra, dec, response_object, geometry_object, echan_list):
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

        # Build a SkyCoord object of the PS
        self._ps_skycoord = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

        self._rsp = response_object
        self._geom = geometry_object
        self._data_type = self._rsp.data_type

        if self._data_type == 'ctime':
            self._echan_mask = np.zeros(8, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True
        elif self._data_type == 'cspec':
            self._echan_mask = np.zeros(128, dtype=bool)
            for e in echan_list:
                self._echan_mask[e] = True

        self._response_array()


    @property
    def skycoord(self):
        """
        Returns the SkyCoord object of the PS
        """
        return self._ps_skycoord

    @property
    def ps_response_array(self):
        """
        Returns an array with the poit source response for the times for which the geometry
        was calculated.
        """
        return self._ps_response

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
        quaternion = self._geom.quaternion
        sc_pos = self._geom.sc_pos
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
            # Calcutate the GBMFrame for all the times for which the geomerty was calcutated
            GBMFrame_list = []
            if rank == 0:

                with progress_bar(times_upper_bound_index - times_lower_bound_index,
                                  title='Calculating GBM frame for several times. '
                                        'This shows the progress of rank 0. All other should be about the same.') as p:
                    for i in range(times_lower_bound_index, times_upper_bound_index):
                        q1, q2, q3, q4 = quaternion[i]
                        scx, scy, scz = sc_pos[i]
                        GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                      quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                      sc_pos_Z=scz))

                        p.increase()
            else:
                for i in range(times_lower_bound_index, times_upper_bound_index):
                    q1, q2, q3, q4 = quaternion[i]
                    scx, scy, scz = sc_pos[i]
                    GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                  quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                  sc_pos_Z=scz))
            GBMFrame_list = np.array(GBMFrame_list)

            # Get the postion of the PS in the satellite frame (saved as vector and as SkyCoord object)
            ps_pos_sat_list = []
            ps_pos_sat_objects = []
            if rank == 0:
                with progress_bar(len(GBMFrame_list),
                                  title='Calculating PS position in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                      self._name)) as p:
                    for i in range(0, len(GBMFrame_list)):
                        ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                        ps_pos_sat_objects.append(ps_pos_sat)
                        az = ps_pos_sat.lon.deg
                        zen = ps_pos_sat.lat.deg
                        ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) *
                                                np.cos(az * (np.pi / 180)),
                                                np.cos(zen * (np.pi / 180)) *
                                                np.sin(az * (np.pi / 180)),
                                                np.sin(zen * (np.pi / 180))])
                        p.increase()

            else:
                for i in range(0, len(GBMFrame_list)):
                    ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                    ps_pos_sat_objects.append(ps_pos_sat)
                    az = ps_pos_sat.lon.deg
                    zen = ps_pos_sat.lat.deg
                    ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                            np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                            np.sin(zen * (np.pi / 180))])
            ps_pos_sat_list = np.array(ps_pos_sat_list)
            ps_pos_sat_objects = np.array(ps_pos_sat_objects)

            # Calcutate the response for the different ps locations

            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            ps_response = []
            if rank == 0:
                with progress_bar(len(ps_pos_sat_list),
                                  title='Calculating the response for all PS positions in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                      self._name)) as p:
                    for point in ps_pos_sat_list:
                        matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                        ps_response.append(matrix.T)
                    p.increase()
            else:
                for point in ps_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    ps_response.append(matrix.T)

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            separation = np.array(separation)

            # define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than ps is behind earth)
            for i in range(len(ps_response)):
                # Check if not occulted by earth
                if separation[i] < earth_opening_angle:
                    # If occulted by earth set response to zero
                    ps_response[i] = ps_response[i] * 0

            # Gather all results in rank=0 and broadcast the final result to all ranks
            ps_response = np.array(ps_response)
            ps_response_g = comm.gather(ps_response, root=0)

            ps_pos_sat_objects = np.array(ps_pos_sat_objects)
            ps_pos_sat_objects_g = comm.gather(ps_pos_sat_objects, root=0)

            separation = np.array(separation)
            separation_g = comm.gather(separation, root=0)

            if rank == 0:
                ps_response_g = np.concatenate(ps_response_g)
                separation_g = np.concatenate(separation_g)
                ps_pos_sat_objects_g = np.concatenate(ps_pos_sat_objects_g)
            ps_response = comm.bcast(ps_response_g, root=0)
            separation = comm.bcast(separation_g, root=0)
            self._ps_pos_sat_objects = comm.bcast(ps_pos_sat_objects_g, root=0)

        # Singlecore calculation
        else:

            # Calcutate the GBMFrame for all these times
            GBMFrame_list = []
            with progress_bar(len(self._geom.earth_zen),
                              title='Calculating GBM frame for several times. '
                                    'This shows the progress of rank 0. All other should be about the same.') as p:
                for i in range(0, len(quaternion)):
                    q1, q2, q3, q4 = quaternion[i]
                    scx, scy, scz = sc_pos[i]
                    GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3,
                                                  quaternion_4=q4, sc_pos_X=scx, sc_pos_Y=scy,
                                                  sc_pos_Z=scz))
                p.increase()
            GBMFrame_list = np.array(GBMFrame_list)

            # Get the postion of the PS in the sat frame for every timestep
            ps_pos_sat_list = []
            with progress_bar(len(GBMFrame_list),
                              title='Calculating PS position in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                  self._name)) as p:
                for i in range(0, len(GBMFrame_list)):
                    ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                    az = ps_pos_sat.lon.deg
                    zen = ps_pos_sat.lat.deg
                    ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                            np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                            np.sin(zen * (np.pi / 180))])
                    p.increase()

            ps_pos_sat_list = np.array(ps_pos_sat_list)
            # DRM object with dummy quaternion and sc_pos values (all in sat frame,
            # therefore not important)
            DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                         np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), det,
                         Ebin_in_edge, mat_type=0, ebin_edge_out=Ebin_out_edge)
            # Calcutate the response matrix for the different ps locations
            ps_response = []
            with progress_bar(len(ps_pos_sat_list),
                              title='Calculating the response for all PS positions in sat frame for {}.This shows the progress of rank 0. All other should be about the same.'.format(
                                  self._name)) as p:
                for point in ps_pos_sat_list:
                    matrix = self._rsp._response(point[0], point[1], point[2], DRM).matrix[self._echan_mask]
                    ps_response.append(matrix.T)
                p.increase()

            # Calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            separation = np.array(separation)

            # Define the earth opening angle
            earth_opening_angle = 67

            # Set response 0 when separation is <67 grad (than ps is behind earth)
            for i in range(len(ps_response)):
                # Check if not occulted by earth
                if separation[i] < earth_opening_angle:
                    # If occulted by earth set response to zero
                    ps_response[i] = ps_response[i] * 0
        self._separation = separation
        self._ps_response = np.array(ps_response)

    def _calc_src_occ(self):
        """

        :return:
        """

        src_occ_ang = []
        earth_positions = self._data.earth_position  # type: ContinuousData.earth_position

        with progress_bar(len(self._interpolation_time) - 1,
                          title='Calculating earth occultation of point source') as p:
            for earth_position in earth_positions:
                src_occ_ang.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)

                p.increase()

        self._src_occ_ang = np.array(src_occ_ang)

        self._occulted_time = np.array(self._interpolation_time)

        self._occulted_time[np.where(self._src_occ_ang != 0)] = 0

        self._point_source_earth_angle_interpolator = interpolate.interp1d(self._interpolation_time, src_occ_ang)

        del src_occ_ang, earth_positions

    def _zero(self):
        print("Numpy where condition true")
        return 0

    def _set_relative_location(self):

        """
        look at continous data sun stuff (setup_geometry)

        coordinate is _pointing of the detectors
        calculate seperation of detectors and point source
        store in arrays for time bins / sub sample time

        interpolate and create a function

        """

        sep_angle = []
        pointing = self._data.pointing  # type: ContinuousData.pointing

        # go through a subset of times and calculate the sun angle with GBM geometry

        with progress_bar(len(self._interpolation_time) - 1, title='Calculating point source seperation angles') as p:
            for point in pointing:
                sep_angle.append(coord.SkyCoord.separation(self._ps_skycoord, point).value)

                p.increase()

        # interpolate it
        self._point_source_interpolator = interpolate.interp1d(self._interpolation_time, sep_angle)

        del sep_angle, pointing

    def calc_occ_array(self, time_bins):
        """

        :param time_bins:
        :return:
        """
        self._src_ang_bin = np.array(self._point_source_interpolator(time_bins))
        self._src_ang_bin[np.where(self._src_occ_ang == 0)] = 0.

        return self._src_ang_bin

    def earth_occ_of_ps(self, mean_time):  # mask for ps behind earth
        """
        Calculates a mask that is 0 for all time_bins in which the PS is behind the earth and 1 if not
        :param mean_time:
        :return:
        """
        # define the size of the earth
        earth_radius = 6371000.8  # geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
        atmosphere = 12000.  # the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere
        sat_dist = 6912000.
        earth_opening_angle = math.asin(r / sat_dist) * 360. / (2. * math.pi)  # earth-cone
        mask = np.zeros_like(mean_time)
        mask[np.where(np.array(self._point_source_earth_angle_interpolator(mean_time)) > earth_opening_angle)] = 1
        return mask

    def separation_angle(self, met):
        """
        call interpolated function and return separation for met (mid eval time)
        """
        return self._point_source_interpolator(met)

    def _cleanup(self):
        del self._interpolation_time, self._src_occ_ang

    @property
    def location(self):

        return self._ps_skycoord

    @property
    def src_ang_bin(self):

        return self._src_ang_bin

    @property
    def name(self):

        return self._name

    @property
    def separation(self):

        return self._separation

    @property
    def ps_pos_sat_objects(self):

        return self._ps_pos_sat_objects

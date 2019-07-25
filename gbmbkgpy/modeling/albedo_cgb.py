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


def Albedo_CGB_free(object):
    """
    Class that precalulated the response arrays for the Earth Albedo and CGB for all times for which the geometry
    was calculated. Use this if you want the spectra to be free in the fit (not only normalization) 
    """
    def __init__(self, response_object, geometry_object):
        self._rsp = response_object
        self._geom = geometry_object
        
        self._response_sum()

    @property
    def cgb_effective_response(self):
        """
        Returns the precalulated effective response for the CGB for all times for which the geometry
        was calculated
        """
        return self._array_cgb_response_sum

    @property
    def earth_effective_response(self):
        """
        Returns the precalulated effective response for the Earth for all times for which the geometry
        was calculated
        """
        return self._array_earth_response_sum

    @property
    def Ebin_in_edge(self):
        """
        Returns the Ebin_in edges as defined in the response object
        """
        return self._rsp.Ebin_in_edge
    
    def _response_sum(self):
        """
        Calculate the effective response sum for all interpolation times for which the geometry was 
        calculated. This supports MPI to reduce the calculation time.
        To calculate the responses created on a grid in the response_object are used. All points 
        that are not occulted by the earth are added
        """

        # Get the precalulated points and responses on the unit sphere
        points = self._rsp.points
        responses = self._rsp.responses

        # Factor to multiply the responses with. Needed as the later spectra are given in units of
        # 1/sr. The sr_points gives the area of the sphere occulted by one point
        sr_points = 4 * np.pi / len(points)
        
        # Get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []

        #If MPI available it is used to speed up the calculation
        if using_mpi:
            
            # Last rank has to cover one more index. Caused by the calculation of the
            # Geometry for the last time bin of the day
            if rank == size - 1:
                upper_index = self._geom.times_upper_bound_index + 1
            else:
                upper_index = self._geom.times_upper_bound_index

            for i in range(self._geom.times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._geom.earth_zen[i] * (np.pi / 180))
                              * np.cos(self._geom.earth_az[i]* (np.pi / 180)),
                              np.cos(self._geom.earth_zen[i] * (np.pi / 180))
                              * np.sin(self._geom.earth_az[i]* (np.pi / 180)),
                              np.sin(self._geom.earth_zen[i] * (np.pi / 180))]))
            earth_pos_inter_times = np.array(earth_pos_inter_times)
            
            # Define the opening angle of the earth in degree
            opening_angle_earth = 67

            # Initalize the lists for the summed effective responses
            array_cgb_response_sum = []
            array_earth_response_sum = []

            # Add the responses that are occulted by earth to earth effective response
            # and the others to CGB effective response. Do this for all times for which the
            # geometry was calculated
            for pos in earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
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
            
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []
            for pos in self._earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
                array_cgb_response_sum.append(cgb_response_time)
                array_earth_response_sum.append(earth_response_time)

        # Mulitiply by the sr_points factor which is the area of the unit sphere covered by every point
        self._array_cgb_response_sum = np.array(array_cgb_response_sum)*sr_points
        self._array_earth_response_sum = np.array(array_earth_response_sum)*sr_points


def Albedo_CGB_fixed(object):
    """
    Class that precalulated the rates arrays for the Earth Albedo and CGB for all times for which the geometry
    was calculated for a normalization 1. Use this if you want that inly the normalization of the spectra 
    is a free fit parameter.
    """
    def __init__(self, response_object, geometry_object):

        self._rsp = response_object
        self._geom = geometry_object

        # Set spectral parameters to literature values (Earth and CGB from Ajello)

        #cgb spectrum
        self._index1_cgb = 1.32
        self._index2_cgb = 2.88
        self._break_energy_cgb = 29.99

        #earth spectrum
        self._index1_earth = -5
        self._index2_earth = 1.72
        self._break_energy_earth = 33.7

        self._response_sum()
        self._rates_array()

    @property
    def earth_rate_array(self):
        """
        Returns an array with the predicted count rates for the times for which the geometry 
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later) 
        and the fixed spectral parameters defined above.
        """
        return self._folded_flux_earth

    @property
    def cgb_rate_array(self):
        """
        Returns an array with the predicted count rates for the times for which the geometry 
        was calculated for all energy channels. Assumed an normalization=1 (will be fitted later) 
        and the fixed spectral parameters defined above.
        """
        return self._folded_flux_cgb

    
    
    def _rates_array(self):
        # Calculate the true flux for the Earth for the assumed spectral parameters (Normalization=1).
        # This true flux is binned in energy bins as defined in the response object
        true_flux_earth = self._integral_earth(self._rsp.Ebin_in_edge[:-1], self._rsp.Ebin_in_edge[1:]) 
        self._folded_flux_earth = np.dot(true_flux_earth, self._array_earth_response_sum)

        true_flux_cgb= self._integral_cgb(self._rsp.Ebin_in_edge[:-1], self._rsp.Ebin_in_edge[1:])
        self._folded_flux_cgb = np.dot(true_flux_cgb, self._array_cgb_response_sum)
        


    def _response_sum(self):
        """
        Calculate the effective response sum for all interpolation times for which the geometry was 
        calculated. This supports MPI to reduce the calculation time.
        To calculate the responses created on a grid in the response_object are used. All points 
        that are not occulted by the earth are added
        """

        # Get the precalulated points and responses on the unit sphere
        points = self._rsp.points
        responses = self._rsp.responses

        # Factor to multiply the responses with. Needed as the later spectra are given in units of
        # 1/sr. The sr_points gives the area of the sphere occulted by one point
        sr_points = 4 * np.pi / len(points)
        
        # Get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []

        #If MPI available it is used to speed up the calculation
        if using_mpi:
            
            # Last rank has to cover one more index. Caused by the calculation of the
            # Geometry for the last time bin of the day
            if rank == size - 1:
                upper_index = self._geom.times_upper_bound_index + 1
            else:
                upper_index = self._geom.times_upper_bound_index

            for i in range(self._geom.times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._geom.earth_zen[i] * (np.pi / 180))
                              * np.cos(self._geom.earth_az[i]* (np.pi / 180)),
                              np.cos(self._geom.earth_zen[i] * (np.pi / 180))
                              * np.sin(self._geom.earth_az[i]* (np.pi / 180)),
                              np.sin(self._geom.earth_zen[i] * (np.pi / 180))]))
            earth_pos_inter_times = np.array(earth_pos_inter_times)
            
            # Define the opening angle of the earth in degree
            opening_angle_earth = 67

            # Initalize the lists for the summed effective responses
            array_cgb_response_sum = []
            array_earth_response_sum = []

            # Add the responses that are occulted by earth to earth effective response
            # and the others to CGB effective response. Do this for all times for which the
            # geometry was calculated
            for pos in earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
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
            
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []
            for pos in self._earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
                array_cgb_response_sum.append(cgb_response_time)
                array_earth_response_sum.append(earth_response_time)

        # Mulitiply by the sr_points factor which is the area of the unit sphere covered by every point
        self._array_cgb_response_sum = np.array(array_cgb_response_sum)*sr_points
        self._array_earth_response_sum = np.array(array_earth_response_sum)*sr_points
    
    
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
        return self._spectrum(e, C, self._index1_earth, self._index2_earth, self._break_energy_earth)

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
        return self._spectrum(e, C, self._index1_cgb, self._index2_cgb, self._break_energy_cgb)

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
    

    def _earth_rate_array(self):
        """
        Calculate the earth_rate_array for all interpolation times for which the geometry 
        was calculated. This supports MPI to reduce the calculation time. To calculate the 
        earth_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM 
        for the earth albedo.
        :return:
        """
        points = self._rate_generator_DRM.points
        earth_rates = self._rate_generator_DRM.earth_rate
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
                print(upper_index)
            else:
                upper_index = self._times_upper_bound_index
            
            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            earth_pos = np.array(earth_pos_inter_times) #
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []

            det_earth_angle = []#
            #point_earth_angle_all_inter = [] #                                                                                                                                                                                                                                
            #point_base_rate_all_inter = [] # 
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])

                det = np.array([1,0,0])#
                det_earth_angle.append(np.arccos(np.dot(pos, det))*180/np.pi)#
                #point_earth_angle = [] #                                                                                                                                                                                                                                      
                #point_base_rate = [] #
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    #point_earth_angle.append(angle_earth)#
                    if angle_earth < opening_angle_earth:
                        B=0
                        earth_rate += earth_rates[i]*np.exp(B*angle_earth)#TODO RING EFFECT
                        #point_base_rate.append(earth_rates[i])# 
                    #else:#                                                                                                                                                                                                                                                    
                        #point_base_rate.append(np.zeros_like(earth_rates[i]))#
                array_earth_rate.append(earth_rate)
                #point_base_rate_all_inter.append(point_base_rate)#
                #point_earth_angle_all_inter.append(point_earth_angle)#
                
            array_earth_rate = np.array(array_earth_rate)
            det_earth_angle = np.array(det_earth_angle)#
            #point_earth_angle = np.array(point_earth_angle_all_inter)#                                                                                                                                                                                                  
            #point_base_rate = np.array(point_base_rate_all_inter)#
            #del point_earth_angle_all_inter, point_base_rate_all_inter, earth_pos_inter_times
            array_earth_rate_g = comm.gather(array_earth_rate, root=0)
            det_earth_angle_g = comm.gather(det_earth_angle, root=0)#
            earth_pos_g = comm.gather(earth_pos, root=0)#
            #point_earth_angle_g = comm.gather(point_earth_angle, root=0)
            #point_base_rate_g = comm.gather(point_base_rate, root=0)
            if rank == 0:
                array_earth_rate_g = np.concatenate(array_earth_rate_g)
                det_earth_angle_g = np.concatenate(det_earth_angle_g)
                earth_pos_g = np.concatenate(earth_pos_g)
                #point_earth_angle_g = np.concatenate(point_earth_angle_g)
                #point_base_rate_g = np.concatenate(point_base_rate_g)
            array_earth_rate = comm.bcast(array_earth_rate_g, root=0)
            det_earth_angle = comm.bcast(det_earth_angle_g, root=0)
            earth_pos = comm.bcast(earth_pos_g,root=0)#
            #point_earth_angle = comm.bcast(point_earth_angle_g, root=0)
            #point_base_rate = comm.bcast(point_base_rate_g, root=0)
            #del array_earth_rate_g, point_earth_angle_g, point_base_rate_g
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []
            #point_earth_angle_all_inter = [] #
            #point_base_rate_all_inter = [] #
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])
                #point_earth_angle = [] #
                #point_base_rate = [] #
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    #point_earth_angle.append(angle_earth)#
                    if angle_earth < opening_angle_earth:
                        #point_base_rate.append(earth_rates[i])#
                        earth_rate += earth_rates[i]
                    #else:#
                        #point_base_rate.append(np.zeros_like(earth_rates[i]))#
                array_earth_rate.append(earth_rate)
                #point_base_rate_all_inter.append(point_base_rate)#
                #point_earth_angle_all_inter.append(point_earth_angle)#
            #point_base_rate = point_base_rate_all_inter
            #point_earth_angle = point_earth_angle_all_inter
        array_earth_rate = np.array(array_earth_rate).T
        #point_earth_angle = np.array(point_earth_angle)#
        #if rank==0:
            #print('Earth pos')
            #print(earth_pos[:10])
            #print('earth_rate')
            #print(array_earth_rate[4][:10])
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #surf = ax.scatter(earth_pos[:,0],earth_pos[:,1],earth_pos[:,2], s=0.4, c=array_earth_rate[4], cmap='plasma')
            #ax.scatter(1,0,0,s=10,c='red')
            #fig.colorbar(surf)
            #fig.savefig('testing_B_{}.pdf'.format(B))
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #surf = ax.scatter(points[:,0],points[:,1],points[:,2], s=0.4, c=earth_rates[:,4], cmap='plasma')
            #fig.colorbar(surf)
            #fig.savefig('testing_2.pdf')
        #point_base_rate = np.array(point_base_rate)#
        #self._point_earth_angle_interpolator = interpolate.interp1d(self._sun_time, point_earth_angle, axis=0)#
        #self._point_base_rate_interpolator = interpolate.interp1d(self._sun_time, point_base_rate, axis=0)#
        self._earth_rate_interpolator = interpolate.interp1d(self._sun_time, array_earth_rate)
        #del point_base_rate, point_earth_angle, array_earth_rate
   

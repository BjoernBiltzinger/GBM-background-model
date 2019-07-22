import numpy as np
from gbmgeometry import PositionInterpolator, gbm_detector_list
import astropy.time as astro_time

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.utils.progress_bar import progress_bar

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True ###################33

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False




valid_det_names = ['n0','n1' ,'n2' ,'n3' ,'n4' ,'n5' ,'n6' ,'n7' ,'n8' ,'n9' ,'na' ,'nb'] 

class Geometry(object):
    def __init__(self, time_bins_mean, det, pos_hist_path, n_bins_to_calculate):
        """
        Initalize the geometry precalculation. This calculates several quantities (e.g. Earth
        position in the satellite frame for n_bins_to_calculate times during the day
        """

        # Test if all the input is valid
        assert type(self.mean_time)==np.ndarray, 'Invalid type for mean_time. Must be an array but is {}.'.format(type(self.mean_time))
        assert det in valid_det_names, 'Invalid det name. Must be one of these {} but is {}.'.format(valid_det_names, det)
        assert file_existing_and_readable(pos_hist_path), '{} does not exist'.format(pos_hist_path)
        assert type(n_bins_to_calculate)==int, 'Type of n_bins_to_calculate has to be int but is {}'.format(type(n_bins_to_calculate))


        # Save everything 
        self.mean_time = time_bins_mean
        self._det = det
        self._pos_hist = pos_hist_path
        self._n_bins_to_calculate = n_bins_to_calculate

        # Calculate Geometry. With or without Mpi support.
        if using_mpi:
            self._setup_geometery_mpi()
        else:
            self._setup_geometery_no_mpi()

    # All properties of the class.
    # Returns the calculated values of the quantities for all the n_bins_to_calculate times
    # Of the day used in setup_geometry
    @property
    def time(self):
        """
        Returns the times of the time bins for which the geometry was calculated
        """

        return self._time

    @property
    def sun_angle(self):
        """
        Returns the angle between the sun and the line of sight for all times for which the 
        geometry was calculated
        """
        
        return self._sun_angle

    @property
    def earth_az(self):
        """
        Returns the azimuth angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_az

    @property
    def earth_zen(self):
        """
        Returns the zenith angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_zen

    @property
    def quaternion(self):
        """
        Returns the quaternions, defining the rotation of the satellite, for all times for which the 
        geometry was calculated
        """

        return self._quaternion
    
    @property
    def sc_pos(self):
        """
        Returns the spacecraft position, in ECI coordinates, for all times for which the 
        geometry was calculated
        """

        return self._sc_pos


    def _setup_geometery_mpi(self):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you have MPI
        and are running this on several cores.
        """

        assert using_mpi, 'You need MPI to use this function, please use _setup_geometery_no_mpi if you do not have MPI'


        # Create the PositionInterpolator object with the infos from the poshist file
        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # Number of bins to skip, to equally distribute the n_bins_to_calculate times over the day 
        n_skip = int(np.ceil(self._n_time_bins / self._n_bins_to_calculate))

        # Init all lists
        sun_angle = []
        time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = [] #earth pos in icrs frame (skycoord)

        #additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos =[]

        # Split up the times at which the geometry is calculated between the used cores
        list_times_to_calculate = self.mean_time[::n_skip]
        self._times_per_rank = float(len(list_times_to_calculate))/float(size)
        self._times_lower_bound_index = int(np.floor(rank*self._times_per_rank))
        self._times_upper_bound_index = int(np.floor((rank+1)*self._times_per_rank))

        # Only rank==0 gives some output how much of the geometry is already calculated (progress_bar)
        if rank==0:
            with progress_bar(len(list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]),
                              title='Calculating geomerty. This shows the progress of rank 0. All other should be about the same.') as p:

                # Calculate the geometry for all times associated with this rank 
                for mean_time in list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]:
                    quaternion_step = self._position_interpolator.quaternion(mean_time)
                    sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                    sun_angle.append(det.sun_angle.value)
                    time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)

                    pointing_vector.append(det.pointing_vector_earth_frame)
                    p.increase()
        else:
            # Calculate the geometry for all times associated with this rank (for rank!=0).
            # No output here.
            for mean_time in list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]:
                quaternion_step = self._position_interpolator.quaternion(mean_time)
                sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)

        # Get the last data point with the last rank
        # We have to do this to get the iterpolation working for all time bins of the day
        # Otherwise the interpolations won't work for the last time bins
        if rank == size - 1:

            mean_time = self.mean_time[-2]
            quaternion_step = self._position_interpolator.quaternion(mean_time)
            sc_pos_step = self._position_interpolator.sc_pos(mean_time)

            det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                               sc_pos=sc_pos_step,
                                               time=astro_time.Time(self._position_interpolator.utc(mean_time)))

            sun_angle.append(det.sun_angle.value)
            time.append(mean_time)
            az, zen = det.earth_az_zen_sat
            earth_az.append(az)
            earth_zen.append(zen)
            earth_position.append(det.earth_position)

            quaternion.append(quaternion_step)
            sc_pos.append(sc_pos_step)


        #make the list numpy arrays
        sun_angle = np.array(sun_angle)
        time = np.array(time)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        quaternion = np.array(quaternion)
        sc_pos = np.array(sc_pos)

        #gather all results in rank=0
        sun_angle_gather = comm.gather(sun_angle, root=0)
        time_gather = comm.gather(time, root=0)
        earth_az_gather = comm.gather(earth_az, root=0)
        earth_zen_gather = comm.gather(earth_zen, root=0)
        earth_position_gather = comm.gather(earth_position, root=0)


        quaternion_gather = comm.gather(quaternion, root=0)
        sc_pos_gather = comm.gather(sc_pos, root=0)

        #make one list out of this
        if rank == 0:
            sun_angle_gather = np.concatenate(sun_angle_gather)
            time_gather = np.concatenate(time_gather)
            earth_az_gather = np.concatenate(earth_az_gather)
            earth_zen_gather = np.concatenate(earth_zen_gather)
            earth_position_gather = np.concatenate(earth_position_gather)

            quaternion_gather=np.concatenate(quaternion_gather)
            sc_pos_gather = np.concatenate(sc_pos_gather)

        #broadcast the final arrays again to all ranks
        sun_angle = comm.bcast(sun_angle_gather, root=0)
        time = comm.bcast(time_gather, root=0)
        earth_az = comm.bcast(earth_az_gather, root=0)
        earth_zen = comm.bcast(earth_zen_gather, root=0)
        earth_position = comm.bcast(earth_position_gather, root=0)

        quaternion = comm.bcast(quaternion_gather, root=0)
        sc_pos = comm.bcast(sc_pos_gather, root=0)

        # Final save of everything 

        self._sun_angle = sun_angle
        self._time = time
        self._earth_az = earth_az
        self._earth_zen = earth_zen
        self._earth_position = earth_position

        self._quaternion = quaternion
        self._sc_pos = sc_pos
    
    def _setup_geometery_no_mpi(self):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you do not use MPI
        """
        assert not using_mpi, 'This function is only available if you are not using mpi!'

        
        # Create the PositionInterpolator object with the infos from the poshist file
        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # Number of bins to skip, to equally distribute the n_bins_to_calculate times over the day
        n_skip = int(np.ceil(self._n_time_bins / self._n_bins_to_calculate))

        # Init all lists
        sun_angle = []
        time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = [] #earth pos in icrs frame (skycoord)

        #additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos =[]

        # Give some output how much of the geometry is already calculated (progress_bar)
        with progress_bar(self._n_bins_to_calculate, title='Calculating sun and earth position') as p:
            # Calculate the geometry for all times
            for mean_time in self.mean_time[::n_skip]:
                quaternion_step = self._position_interpolator.quaternion(mean_time)
                sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)

                p.increase()

        # Get the geometry for the last time bin
        # We have to do this to get the iterpolation working for all time bins of the day
        # Otherwise the interpolations won't work for the last time bins 

        mean_time = self.mean_time[-2]

        quaternion_step = self._position_interpolator.quaternion(mean_time)
        sc_pos_step = self._position_interpolator.sc_pos(mean_time)
        det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                           sc_pos=sc_pos_step,
                                           time=astro_time.Time(self._position_interpolator.utc(mean_time)))

        sun_angle.append(det.sun_angle.value)
        time.append(mean_time)
        az, zen = det.earth_az_zen_sat
        earth_az.append(az)
        earth_zen.append(zen)
        earth_position.append(det.earth_position)

        quaternion.append(quaternion_step)
        sc_pos.append(sc_pos_step)

        # Make the list numpy arrays
        sun_angle = np.array(sun_angle)
        time = np.array(time)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        quaternion = np.array(quaternion)
        sc_pos = np.array(sc_pos)


        # Final save of everything
        
        self._sun_angle = sun_angle
        self._time = time
        self._earth_az = earth_az
        self._earth_zen = earth_zen
        self._earth_position = earth_position

        self._quaternion = quaternion
        self._sc_pos = sc_pos



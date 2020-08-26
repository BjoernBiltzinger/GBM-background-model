import numpy as np
import os
from gbmgeometry import PositionInterpolator, gbm_detector_list
import astropy.time as astro_time

from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

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

if using_mpi:
    using_multiprocessing = False
else:
    try:
        from pathos.pools import ProcessPool as Pool
        using_multiprocessing = True
    except:
        using_multiprocessing = False


class Geometry(object):
    def __init__(self, data, dates, n_bins_to_calculate_per_day):
        """
        Initalize the geometry precalculation. This calculates several quantities (e.g. Earth
        position in the satellite frame for n_bins_to_calculate times during the day
        """

        # Test if all the input is valid
        assert (
            type(data.mean_time) == np.ndarray
        ), "Invalid type for mean_time. Must be an array but is {}.".format(
            type(data.mean_time)
        )

        assert (
            type(n_bins_to_calculate_per_day) == int
        ), "Type of n_bins_to_calculate has to be int but is {}".format(
            type(n_bins_to_calculate_per_day)
        )

        # Save everything
        self._data = data
        self._mean_time = data.mean_time
        self._n_bins_to_calculate_per_day = n_bins_to_calculate_per_day
        self._day_start_times = data.day_start_times
        self._day_stop_times = data.day_stop_times
        self._day_list = sorted(dates)  # map(str, sorted(map(int, day_list)))

        # Number of bins to skip, to equally distribute the n_bins_to_calculate times over the day
        n_skip = int(
            np.ceil(
                len(self._mean_time) / (self._n_bins_to_calculate_per_day * len(dates))
            )
        )

        # Create the lists of the times where to calculate the geometry
        list_times_to_calculate = self._mean_time[::n_skip]

        # Add start and stop time of days to times for which the geometry should be calculated (to ensure a valid
        # interpolation for all used times
        # self._list_times_to_calculate = self._add_start_stop(list_times_to_calculate, self._day_start_times,
        #                                                      self._day_stop_times)

        if self._data.data_type == "trigdat":
            self._pos_hist = np.array([self._data.trigdata_path])

            # Dirty fix because the position data in trigdat only interpolates up to the beginning of the last time bin
            list_times_to_calculate[-2] = self._data.time_bins[-2, 1]
            list_times_to_calculate[-1] = self._data.time_bins[-1, 0]

            # GBM Geometry handles trigdat times in reference to the trigger time
            # we have to account for this before and after the position interpolation
            list_times_to_calculate = list_times_to_calculate - self._data.trigtime
            self._day_start_times = self._day_start_times - self._data.trigtime
            self._day_stop_times = self._day_stop_times - self._data.trigtime

        else:
            # Check if poshist file exists, if not download it and save the paths for all days in an array
            self._pos_hist = np.array([])
            for day in dates:
                poshistfile_name = "glg_{0}_all_{1}_v00.fit".format("poshist", day)
                poshistfile_path = os.path.join(
                    get_path_of_external_data_dir(), "poshist", poshistfile_name
                )

                # If using MPI only rank=0 downloads the data, all other have to wait
                if using_mpi:
                    if rank == 0:
                        if not file_existing_and_readable(poshistfile_path):
                            download_data_file(day, "poshist")
                    comm.Barrier()
                else:
                    if not file_existing_and_readable(poshistfile_path):
                        download_data_file(day, "poshist")

                # Save poshistfile_path for later usage
                self._pos_hist = np.append(self._pos_hist, poshistfile_path)
            for pos in self._pos_hist:
                assert file_existing_and_readable(pos), "{} does not exist".format(pos)

            # Add start and stop time of days to times for which the geometry should be calculated (to ensure a valid
            # interpolation for all used times
            list_times_to_calculate = self._add_start_stop(
                list_times_to_calculate, self._day_start_times, self._day_stop_times
            )

            # Add the boundaries of the position interpolator to ensure valid interpolation of all time_bins
            list_times_to_calculate = self._add_interpolation_boundaries(
                list_times_to_calculate
            )

        # Remove possible dublicates, these would break the numba interpolation
        self._list_times_to_calculate = np.unique(list_times_to_calculate)

        # Create a mask with all entries False to later un-mask the valid times
        self._interpolation_mask = np.zeros(
            len(self._list_times_to_calculate), dtype=bool
        )

        # Calculate Geometry. With or without Mpi support.
        for day_number, day in enumerate(dates):
            if using_mpi:
                (
                    time,
                    sc_pos,
                    quaternion,
                    sc_altitude,
                    earth_positions,
                    earth_az_zen,
                    earth_cartesian,
                    sun_positions,
                    sun_az_zen,
                    sun_cartesian,
                    times_lower_bound_index,
                    times_upper_bound_index,
                ) = self._one_day_setup_geometery_mpi(day_number)
            else:
                (
                    time,
                    sc_pos,
                    quaternion,
                    sc_altitude,
                    earth_positions,
                    earth_az_zen,
                    earth_cartesian,
                    sun_positions,
                    sun_az_zen,
                    sun_cartesian,
                ) = self._one_day_setup_geometery_no_mpi(day_number)
            if day_number == 0:
                self._time = [time]
                self._sc_pos = [sc_pos]
                self._quaternion = [quaternion]
                self._sc_altitude = [sc_altitude]
                self._earth_positions = [earth_positions]
                self._earth_az_zen = [earth_az_zen]
                self._earth_cartesian = [earth_cartesian]
                self._sun_positions = [sun_positions]
                self._sun_az_zen = [sun_az_zen]
                self._sun_cartesian = [sun_cartesian]
                if using_mpi:
                    self._times_lower_bound_index = np.array([times_lower_bound_index])
                    self._times_upper_bound_index = np.array([times_upper_bound_index])
            else:
                self._time.append(time)
                self._sc_pos.append(sc_pos)
                self._quaternion.append(quaternion)
                self._sc_altitude.append(sc_altitude)
                self._earth_positions.append(earth_positions)
                self._earth_az_zen.append(earth_az_zen)
                self._earth_cartesian.append(earth_cartesian)
                self._sun_positions.append(sun_positions)
                self._sun_az_zen.append(sun_az_zen)
                self._sun_cartesian.append(sun_cartesian)
                if using_mpi:
                    self._times_lower_bound_index = np.append(
                        self._times_lower_bound_index, times_lower_bound_index
                    )
                    self._times_upper_bound_index = np.append(
                        self._times_upper_bound_index, times_upper_bound_index
                    )
        self._time = np.concatenate(self._time, axis=0)
        self._sc_pos = np.concatenate(self._sc_pos, axis=0)
        self._quaternion = np.concatenate(self._quaternion, axis=0)
        self._sc_altitude = np.concatenate(self._sc_altitude, axis=0)
        self._earth_positions = np.concatenate(self._earth_positions, axis=0)
        self._earth_az_zen = np.concatenate(self._earth_az_zen, axis=0)
        self._earth_cartesian = np.concatenate(self._earth_cartesian, axis=0)
        self._sun_positions = np.concatenate(self._sun_positions, axis=0)
        self._sun_az_zen = np.concatenate(self._sun_az_zen, axis=0)
        self._sun_cartesian = np.concatenate(self._sun_cartesian, axis=0)

        # Here we add the trigger time to build the model in MET
        if self._data.data_type == "trigdat":
            self._time = self._time + self._data.trigtime
            self._list_times_to_calculate = (
                self._list_times_to_calculate + self._data.trigtime
            )

    # All properties of the class.
    # Returns the calculated values of the quantities for all the n_bins_to_calculate times
    # Of the day used in setup_geometry
    @property
    def geometry_times(self):
        """
        Returns the times of the time bins for which the geometry was calculated
        """

        return self._list_times_to_calculate[self._interpolation_mask]

    @property
    def time_days(self):
        """
        Returns the times of the time bins for which the geometry was calculated for all days separately as arrays in
        one big array
        """

        return self._time

    @property
    def sun_positions(self):
        """
        :return: sun positions as skycoord object in sat frame for all times for which the geometry was calculated
        """

        return self._sun_positions

    @property
    def earth_az(self):
        """
        Returns the azimuth angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_az_zen[:, 0]

    @property
    def earth_zen(self):
        """
        Returns the zenith angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_az_zen[:, 1]

    @property
    def earth_position(self):
        """
        Returns the Earth position as SkyCoord object for all times for which the geometry was 
        calculated
        """
        return self._earth_positions

    @property
    def earth_position_cart(self):
        """
        Returns the Earth position in cartesian coordinates for all times for which the geometry was
        calculated
        """
        return self._earth_cartesian

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

    @property
    def times_upper_bound_index(self):
        """
        Returns the upper time bound of the geometries calculated by this rank
        """
        return self._times_upper_bound_index

    @property
    def times_lower_bound_index(self):
        """
        Returns the lower time bound of the geometries calculated by this rank
        """
        return self._times_lower_bound_index

    def _one_day_setup_geometery_mpi(self, day_number):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you have MPI
        and are running this on several cores.
        """

        assert (
            using_mpi
        ), "You need MPI to use this function, please use _setup_geometery_no_mpi if you do not have MPI"

        if self._data.data_type == "trigdat":
            # Create the PositionInterpolator object with the infos from the trigdat file
            position_interpolator = PositionInterpolator.from_trigdat(
                trigdat_file=self._pos_hist[day_number]
            )
        else:
            # Create the PositionInterpolator object with the infos from the poshist file
            position_interpolator = PositionInterpolator.from_poshist(
                poshist_file=self._pos_hist[day_number]
            )

        # Get the times for which the geometry should be calculated for this day (Build a mask that masks all time bins
        # outside the start and stop day of this time bin

        day_idx = np.logical_and(
            (self._list_times_to_calculate >= self._day_start_times),
            (self._list_times_to_calculate <= self._day_stop_times),
        )

        # Mask the bins that are outside of the interpolation times
        interp_range_mask = np.logical_and(
            (self._list_times_to_calculate >= min(position_interpolator.time)),
            (self._list_times_to_calculate <= max(position_interpolator.time)),
        )
        masktot = np.logical_and(day_idx, interp_range_mask)

        self._interpolation_mask[masktot] = True

        list_times_to_calculate = self._list_times_to_calculate[masktot]

        times_per_rank = float(len(list_times_to_calculate)) / float(size)
        times_lower_bound_index = int(np.floor(rank * times_per_rank))
        times_upper_bound_index = int(np.floor((rank + 1) * times_per_rank))

        # Get spacecraft position and quaternions for all geomety times
        sc_positions = position_interpolator.sc_pos(list_times_to_calculate)
        quaternions = position_interpolator.quaternion(list_times_to_calculate)

        # Calculate spacecraft altitude
        earth_radius = 6371.0
        fermi_radius = np.sqrt(np.sum(sc_positions ** 2, axis=0))
        sc_altitude = fermi_radius - earth_radius

        # Init all lists
        sun_positions = []
        sun_az_zen = []
        earth_az_zen = []  # azimuth and zenith angle of earth in sat. frame
        earth_positions = []  # earth pos in icrs frame (skycoord)

        # Only rank==0 gives some output how much of the geometry is already calculated (progress_bar)
        hidden = False if rank == 0 else True

        with progress_bar(
            len(
                list_times_to_calculate[times_lower_bound_index:times_upper_bound_index]
            ),
            title="Calculating geomerty for day {}. This shows the progress of rank 0. "
            "All other should be about the same.".format(self._day_list[day_number]),
            hidden=hidden,
        ) as p:

            # Calculate the geometry for all times associated with this rank
            for i, mean_time in enumerate(
                list_times_to_calculate[times_lower_bound_index:times_upper_bound_index]
            ):

                step_idx = i + times_lower_bound_index

                # The detector used is only a dummy as we are not using
                # any detector specific geometry calculations
                det = gbm_detector_list["n0"](
                    quaternion=quaternions[step_idx],
                    sc_pos=sc_positions[step_idx],
                    time=astro_time.Time(position_interpolator.utc(mean_time)),
                )

                sun_positions.append(det.sun_position)
                sun_az_zen.append([det.sun_position.lon.deg, det.sun_position.lat.deg])

                earth_positions.append(det.earth_position)
                earth_az_zen.append(det.earth_az_zen_sat)

                p.increase()

        # Make the list numpy arrays
        sun_positions = np.array(sun_positions)
        sun_az_zen = np.array(sun_az_zen)
        earth_az_zen = np.array(earth_az_zen)
        earth_positions = np.array(earth_positions)

        # gather all results in rank=0
        sun_positions_gather = comm.gather(sun_positions, root=0)
        sun_az_zen_gather = comm.gather(sun_az_zen, root=0)
        earth_az_zen_gather = comm.gather(earth_az_zen, root=0)
        earth_positions_gather = comm.gather(earth_positions, root=0)

        # make one list out of this
        if rank == 0:
            sun_positions_gather = np.concatenate(sun_positions_gather)
            sun_az_zen_gather = np.concatenate(sun_az_zen_gather)
            earth_az_zen_gather = np.concatenate(earth_az_zen_gather)
            earth_positions_gather = np.concatenate(earth_positions_gather)

        # broadcast the final arrays again to all ranks
        sun_positions = comm.bcast(sun_positions_gather, root=0)
        sun_az_zen = comm.bcast(sun_az_zen_gather, root=0)
        earth_az_zen = comm.bcast(earth_az_zen_gather, root=0)
        earth_positions = comm.bcast(earth_positions_gather, root=0)

        # Calculate the earth position in cartesian coordinates
        earth_rad = np.deg2rad(earth_az_zen)

        earth_cartesian = np.zeros((len(earth_positions), 3))
        earth_cartesian[:, 0] = np.cos(earth_rad[:, 1]) * np.cos(earth_rad[:, 0])
        earth_cartesian[:, 1] = np.cos(earth_rad[:, 1]) * np.sin(earth_rad[:, 0])
        earth_cartesian[:, 2] = np.sin(earth_rad[:, 1])

        # Calculate the sun position in cartesian coordinates
        sun_rad = np.deg2rad(sun_az_zen)

        sun_cartesian = np.zeros((len(sun_az_zen), 3))
        sun_cartesian[:, 0] = np.cos(sun_rad[:, 1]) * np.cos(sun_rad[:, 0])
        sun_cartesian[:, 1] = np.cos(sun_rad[:, 1]) * np.sin(sun_rad[:, 0])
        sun_cartesian[:, 2] = np.sin(sun_rad[:, 1])

        # Return everything
        return (
            list_times_to_calculate,
            sc_positions,
            quaternions,
            sc_altitude,
            earth_positions,
            earth_az_zen,
            earth_cartesian,
            sun_positions,
            sun_az_zen,
            sun_cartesian,
            times_lower_bound_index,
            times_upper_bound_index,
        )

    def _one_day_setup_geometery_no_mpi(self, day_number):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you do not use MPI
        """
        assert (
            not using_mpi
        ), "This function is only available if you are not using mpi!"

        if self._data.data_type == "trigdat":
            # Create the PositionInterpolator object with the infos from the trigdat file
            position_interpolator = PositionInterpolator.from_trigdat(
                trigdat_file=self._pos_hist[day_number]
            )
        else:
            # Create the PositionInterpolator object with the infos from the poshist file
            position_interpolator = PositionInterpolator.from_poshist(
                poshist_file=self._pos_hist[day_number]
            )

        # Get the times for which the geometry should be calculated for this day (Build a mask that masks all time bins
        # outside the start and stop day of this time bin

        day_idx = np.logical_and(
            (self._list_times_to_calculate >= self._day_start_times),
            (self._list_times_to_calculate <= self._day_stop_times),
        )
        # Mask the bins that are outside of the interpolation times
        interp_range_mask = np.logical_and(
            (self._list_times_to_calculate >= min(position_interpolator.time)),
            (self._list_times_to_calculate <= max(position_interpolator.time)),
        )
        masktot = np.logical_and(day_idx, interp_range_mask)

        self._interpolation_mask[masktot] = True

        list_times_to_calculate = self._list_times_to_calculate[masktot]

        # Get spacecraft position and quaternions for all geomety times
        sc_positions = position_interpolator.sc_pos(list_times_to_calculate)
        quaternions = position_interpolator.quaternion(list_times_to_calculate)

        # Calculate spacecraft altitude
        earth_radius = 6371.0
        fermi_radius = np.sqrt(np.sum(sc_positions ** 2, axis=0))
        sc_altitude = fermi_radius - earth_radius

        # Init all lists
        sun_positions = []
        sun_az_zen = []
        earth_az_zen = []  # azimuth and zenith angle of earth in sat. frame
        earth_positions = []  # earth pos in icrs frame (skycoord)

        if using_multiprocessing:
            print("Calculation sun and earth position with multiprocessing.")

            def calc_geo_mp(step_idx):
                det = gbm_detector_list["n0"](
                    quaternion=quaternions[step_idx],
                    sc_pos=sc_positions[step_idx],
                    time=astro_time.Time(
                        position_interpolator.utc(
                            list_times_to_calculate[step_idx]
                        )
                    ),
                )

                return (
                    det.earth_position,
                    det.earth_az_zen_sat,
                    det.sun_position,
                    [det.sun_position.lon.deg, det.sun_position.lat.deg]
                )

            with Pool() as pool:
                geo_steps = pool.map(calc_geo_mp, range(len(list_times_to_calculate)))

            for geo_step in geo_steps:
                earth_positions.append(geo_step[0])
                earth_az_zen.append(geo_step[1])
                sun_positions.append(geo_step[2])
                sun_az_zen.append(geo_step[3])

        else:

            # Give some output how much of the geometry is already calculated (progress_bar)
            with progress_bar(
                len(list_times_to_calculate), title="Calculating sun and earth position"
            ) as p:
                # Calculate the geometry for all times
                for step_idx, mean_time in enumerate(list_times_to_calculate):

                    # The detector used is only a dummy as we are not using
                    # any detector specific geometry calculations
                    det = gbm_detector_list["n0"](
                        quaternion=quaternions[step_idx],
                        sc_pos=sc_positions[step_idx],
                        time=astro_time.Time(position_interpolator.utc(mean_time)),
                    )

                    sun_positions.append(det.sun_position)
                    sun_az_zen.append([det.sun_position.lon.deg, det.sun_position.lat.deg])

                    earth_positions.append(det.earth_position)
                    earth_az_zen.append(det.earth_az_zen_sat)

                    p.increase()

        # Make the list numpy arrays
        sun_positions = np.array(sun_positions)
        sun_az_zen = np.array(sun_az_zen)
        earth_az_zen = np.array(earth_az_zen)
        earth_positions = np.array(earth_positions)

        # Calculate the earth position in cartesian coordinates
        earth_rad = np.deg2rad(earth_az_zen)

        earth_cartesian = np.zeros((len(earth_az_zen), 3))
        earth_cartesian[:, 0] = np.cos(earth_rad[:, 1]) * np.cos(earth_rad[:, 0])
        earth_cartesian[:, 1] = np.cos(earth_rad[:, 1]) * np.sin(earth_rad[:, 0])
        earth_cartesian[:, 2] = np.sin(earth_rad[:, 1])

        # Calculate the sun position in cartesian coordinates
        sun_rad = np.deg2rad(sun_az_zen)

        sun_cartesian = np.zeros((len(sun_az_zen), 3))
        sun_cartesian[:, 0] = np.cos(sun_rad[:, 1]) * np.cos(sun_rad[:, 0])
        sun_cartesian[:, 1] = np.cos(sun_rad[:, 1]) * np.sin(sun_rad[:, 0])
        sun_cartesian[:, 2] = np.sin(sun_rad[:, 1])

        # Return everything
        return (
            list_times_to_calculate,
            sc_positions,
            quaternions,
            sc_altitude,
            earth_positions,
            earth_az_zen,
            earth_cartesian,
            sun_positions,
            sun_az_zen,
            sun_cartesian,
        )

    def _add_start_stop(self, timelist, start_add, stop_add):
        """
        Function that adds the times in start_add and stop_add to timelist if they are not already in the list
        :param timelist: list of times
        :param start_add: start of all days
        :param stop_add: stop of all days
        :return: timelist with start_add and stop_add times added
        """
        for start in start_add:
            if start not in timelist:
                timelist = np.append(timelist, start)
        for stop in stop_add:
            if stop not in timelist:
                timelist = np.append(timelist, stop)
        timelist.sort()
        return timelist

    def _add_interpolation_boundaries(self, timelist):

        for day in self._day_list:
            poshistfile_name = "glg_{0}_all_{1}_v00.fit".format("poshist", day)
            poshistfile_path = os.path.join(
                get_path_of_external_data_dir(), "poshist", poshistfile_name
            )

            # Create the PositionInterpolator object with the infos from the poshist file
            position_interpolator = PositionInterpolator.from_poshist(
                poshist_file=poshistfile_path
            )

            if min(position_interpolator.time) not in timelist:
                timelist = np.append(timelist, min(position_interpolator.time))

            if max(position_interpolator.time) not in timelist:
                timelist = np.append(timelist, min(position_interpolator.time))

        timelist.sort()
        return timelist

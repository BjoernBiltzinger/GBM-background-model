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
]


class Geometry(object):
    def __init__(self, data, detectors, dates, n_bins_to_calculate_per_day):
        geometries = {}

        geometry_times = None

        for det in detectors:
            geometries[det] = Det_Geometry(
                data, det, dates, n_bins_to_calculate_per_day
            )

            # Assert that the times where the geometry is calculated is the same for all detectors
            # this will allow us the speed up the following calculations
            if geometry_times is None:
                geometry_times = geometries[det].time
            else:
                assert np.array_equal(geometry_times, geometries[det].time)

        self._geometries = geometries
        self._geometry_times = geometry_times

    @property
    def geometries(self):
        return self._geometries

    @property
    def geometry_times(self):
        return self._geometry_times


class Det_Geometry(object):
    def __init__(self, data, det, dates, n_bins_to_calculate_per_day):
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
            det in valid_det_names
        ), "Invalid det name. Must be one of these {} but is {}.".format(
            valid_det_names, det
        )
        assert (
            type(n_bins_to_calculate_per_day) == int
        ), "Type of n_bins_to_calculate has to be int but is {}".format(
            type(n_bins_to_calculate_per_day)
        )

        # Save everything
        self._data = data
        self._mean_time = data.mean_time
        self._det = det
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
            list_times_to_calculate = (
                list_times_to_calculate - self._data.trigtime
            )
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
                    sun_positions,
                    time,
                    earth_az,
                    earth_zen,
                    earth_position,
                    earth_cartesian,
                    quaternion,
                    sc_pos,
                    sc_altitude,
                    times_lower_bound_index,
                    times_upper_bound_index,
                ) = self._one_day_setup_geometery_mpi(day_number)
            else:
                (
                    sun_positions,
                    time,
                    earth_az,
                    earth_zen,
                    earth_position,
                    earth_cartesian,
                    quaternion,
                    sc_pos,
                    sc_altitude,
                ) = self._one_day_setup_geometery_no_mpi(day_number)
            if day_number == 0:
                self._sun_positions = [sun_positions]
                self._time = [time]
                self._earth_az = [earth_az]
                self._earth_zen = [earth_zen]
                self._earth_position = [earth_position]
                self._earth_cartesian = [earth_cartesian]
                self._quaternion = [quaternion]
                self._sc_pos = [sc_pos]
                self._sc_altitude = [sc_altitude]
                if using_mpi:
                    self._times_lower_bound_index = np.array([times_lower_bound_index])
                    self._times_upper_bound_index = np.array([times_upper_bound_index])
            else:
                self._sun_positions.append(sun_positions)
                self._time.append(time)
                self._earth_az.append(earth_az)
                self._earth_zen.append(earth_zen)
                self._earth_position.append(earth_position)
                self._earth_cartesian.append(earth_cartesian)
                self._quaternion.append(quaternion)
                self._sc_pos.append(sc_pos)
                self._sc_altitude.append(sc_altitude)
                if using_mpi:
                    self._times_lower_bound_index = np.append(
                        self._times_lower_bound_index, times_lower_bound_index
                    )
                    self._times_upper_bound_index = np.append(
                        self._times_upper_bound_index, times_upper_bound_index
                    )
        self._time = np.concatenate(self._time, axis=0)
        self._sun_positions = np.concatenate(self._sun_positions, axis=0)
        self._earth_az = np.concatenate(self._earth_az, axis=0)
        self._earth_zen = np.concatenate(self._earth_zen, axis=0)
        self._earth_position = np.concatenate(self._earth_position, axis=0)
        self._earth_cartesian = np.concatenate(self._earth_cartesian, axis=0)
        self._quaternion = np.concatenate(self._quaternion, axis=0)
        self._sc_pos = np.concatenate(self._sc_pos, axis=0)
        self._sc_altitude = np.concatenate(self._sc_altitude, axis=0)

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
    def time(self):
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

        return self._earth_az

    @property
    def earth_zen(self):
        """
        Returns the zenith angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_zen

    @property
    def earth_position(self):
        """
        Returns the Earth position as SkyCoord object for all times for which the geometry was 
        calculated
        """
        return self._earth_position

    @property
    def earth_cartesian(self):
        """
        Returns the Earth position as SkyCoord object for all times for which the geometry was
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
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = []  # earth pos in icrs frame (skycoord)

        # Only rank==0 gives some output how much of the geometry is already calculated (progress_bar)
        if rank == 0:
            with progress_bar(
                len(
                    list_times_to_calculate[
                        times_lower_bound_index:times_upper_bound_index
                    ]
                ),
                title="Calculating geomerty for day {}. This shows the progress of rank 0. "
                "All other should be about the same.".format(
                    self._day_list[day_number]
                ),
            ) as p:

                # Calculate the geometry for all times associated with this rank
                for i, mean_time in enumerate(list_times_to_calculate[
                    times_lower_bound_index:times_upper_bound_index
                ]):

                    step_idx = i + times_lower_bound_index

                    det = gbm_detector_list[self._det](
                        quaternion=quaternions[step_idx],
                        sc_pos=sc_positions[step_idx],
                        time=astro_time.Time(position_interpolator.utc(mean_time)),
                    )

                    sun_positions.append(det.sun_position)

                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    p.increase()
        else:
            # Calculate the geometry for all times associated with this rank (for rank!=0).
            # No output here.
            for i, mean_time in enumerate(list_times_to_calculate[
                times_lower_bound_index:times_upper_bound_index
            ]):

                step_idx = i + times_lower_bound_index

                det = gbm_detector_list[self._det](
                    quaternion=quaternions[step_idx],
                    sc_pos=sc_positions[step_idx],
                    time=astro_time.Time(position_interpolator.utc(mean_time)),
                )

                sun_positions.append(det.sun_position)

                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

        # make the list numpy arrays
        sun_positions = np.array(sun_positions)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        # gather all results in rank=0
        sun_positions_gather = comm.gather(sun_positions, root=0)
        earth_az_gather = comm.gather(earth_az, root=0)
        earth_zen_gather = comm.gather(earth_zen, root=0)
        earth_position_gather = comm.gather(earth_position, root=0)

        # make one list out of this
        if rank == 0:
            sun_positions_gather = np.concatenate(sun_positions_gather)
            earth_az_gather = np.concatenate(earth_az_gather)
            earth_zen_gather = np.concatenate(earth_zen_gather)
            earth_position_gather = np.concatenate(earth_position_gather)

        # broadcast the final arrays again to all ranks
        sun_positions = comm.bcast(sun_positions_gather, root=0)
        earth_az = comm.bcast(earth_az_gather, root=0)
        earth_zen = comm.bcast(earth_zen_gather, root=0)
        earth_position = comm.bcast(earth_position_gather, root=0)

        # Calculate the earth position in cartesian coordinates
        earth_pos = np.dstack((earth_zen, earth_az))[0]
        earth_rad = np.deg2rad(earth_pos)

        earth_cartesian = np.zeros((len(earth_pos), 3))
        earth_cartesian[:, 0] = np.cos(earth_rad[:, 0]) * np.cos(earth_rad[:, 1])
        earth_cartesian[:, 1] = np.cos(earth_rad[:, 0]) * np.sin(earth_rad[:, 1])
        earth_cartesian[:, 2] = np.sin(earth_rad[:, 0])

        # Return everything
        return (
            sun_positions,
            list_times_to_calculate,
            earth_az,
            earth_zen,
            earth_position,
            earth_cartesian,
            quaternions,
            sc_positions,
            sc_altitude,
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
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = []  # earth pos in icrs frame (skycoord)

        # Give some output how much of the geometry is already calculated (progress_bar)
        with progress_bar(
            len(list_times_to_calculate), title="Calculating sun and earth position"
        ) as p:
            # Calculate the geometry for all times
            for step_idx, mean_time in enumerate(list_times_to_calculate):
                det = gbm_detector_list[self._det](
                    quaternion=quaternions[step_idx],
                    sc_pos=sc_positions[step_idx],
                    time=astro_time.Time(position_interpolator.utc(mean_time)),
                )

                sun_positions.append(det.sun_position)

                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                p.increase()

        # Make the list numpy arrays
        sun_positions = np.array(sun_positions)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        # Calculate the earth position in cartesian coordinates
        earth_pos = np.dstack((earth_zen, earth_az))[0]
        earth_rad = np.deg2rad(earth_pos)

        earth_cartesian = np.zeros((len(earth_pos), 3))
        earth_cartesian[:, 0] = np.cos(earth_rad[:, 0]) * np.cos(earth_rad[:, 1])
        earth_cartesian[:, 1] = np.cos(earth_rad[:, 0]) * np.sin(earth_rad[:, 1])
        earth_cartesian[:, 2] = np.sin(earth_rad[:, 0])

        # Return everything
        return (
            sun_positions,
            list_times_to_calculate,
            earth_az,
            earth_zen,
            earth_position,
            earth_cartesian,
            quaternions,
            sc_positions,
            sc_altitude
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

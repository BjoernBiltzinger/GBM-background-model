import astropy.coordinates as coord
import astropy.units as u
from gbmgeometry.gbm_frame import GBMFrame
import astropy.time as astro_time
from numpy.linalg import norm


import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt
import math

from gbmgeometry import PositionInterpolator, gbm_detector_list


import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable
#from gbmbkgpy.utils.continuous_data import ContinuousData

from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file
from gbmbkgpy.utils.progress_bar import progress_bar

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

class PointSrc(object):

    def __init__(self, name, ra, dec, data):
        self._name = name
        self._data = data #type: ContinuousData
        self._ps_skycoord = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        self._precalculations()
        #self._calc_src_occ()
        #self._set_relative_location()
        #self._cleanup()
    def _precalculations(self):
        """
        funtion that imports and precalculate everything that is needed to get the point source array for all echans
        :return:
        """
        # import interpolation times, quaternions, sc_pos and the earth position at these times from cont. data
        self._interpolation_time = self._data.interpolation_time  # type: ContinuousData.interpolation_time
        self._quaternion = self._data.quaternion
        self._sc_pos = self._data.sc_pos
        self._earth_positions = self._data.earth_position
        # import the points of the grid around the detector and the rates for a ps spectrum
        self._rate_generator_DRM = self._data.rate_generator_DRM
        self._Ngrid = self._rate_generator_DRM.Ngrid
        self._points_grid = self._rate_generator_DRM.points
        self._rates_points = self._rate_generator_DRM.ps_rate  # for all echans
        if using_mpi:
            # again need to add one to the upper index of the last rank because of the calculation of the Gemoetry for
            # the last time bin
            if rank == size-1:
                upper_index = self._data.times_upper_bound_index + 1
            else:
                upper_index = self._data.times_upper_bound_index

            lower_index = self._data.times_lower_bound_index
            # calcutate the GBMFrame for all the times covered by this rank
            GBMFrame_list = []
            for i in range(lower_index, upper_index):
                q1, q2, q3, q4 = self._quaternion[i]
                scx, scy, scz = self._sc_pos[i]
                GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                                              sc_pos_X=scx, sc_pos_Y=scy, sc_pos_Z=scz))
            self._GBMFrame_list = np.array(GBMFrame_list)
            # get the postion of the PS in the sat frame for every timestep
            self._ps_pos_sat_list = []
            for i in range(0, len(GBMFrame_list)):
                ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                az = ps_pos_sat.lon.deg
                zen = ps_pos_sat.lat.deg
                self._ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                              np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                              np.sin(zen * (np.pi / 180))])
            self._ps_pos_sat_list = np.array(self._ps_pos_sat_list)

            # get the point of the grid closet to the ps pointing (save the corresponding index)
            best_grid_point_index = []
            for i in range(len(self._ps_pos_sat_list)):
                res_vector_norm = norm(self._points_grid - self._ps_pos_sat_list[i], axis=1)
                best_grid_point_index.append(np.argmax(res_vector_norm))
            self._best_grid_point_index = np.array(best_grid_point_index)

            # calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in self._earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            self._separation = np.array(separation)
            # define the earth opening angle
            earth_opening_angle = 67

            # get a list with the rates of the rates of the closest points
            ps_rate_list = []
            for i in range(len(self._best_grid_point_index)):
                # check if not occulted by earth
                if self._separation[i] > earth_opening_angle:
                    ps_rate_list.append(self._rates_points[self._best_grid_point_index[i]])
                else:
                    ps_rate_list.append(np.zeros_like(self._rates_points[self._best_grid_point_index[i]]))
            ps_rate_list=np.array(ps_rate_list)
            ps_rate_list_g = comm.gather(ps_rate_list, root=0)
            if rank == 0:
                ps_rate_list_g = np.concatenate(ps_rate_list_g)
            ps_rate_list = comm.bcast(ps_rate_list_g, root=0)
        else:
            #calcutate the GBMFrame for all these times
            GBMFrame_list = []
            for i in range(0, len(self._quaternion)):
                q1, q2, q3, q4 = self._quaternion[i]
                scx, scy, scz = self._sc_pos[i]
                GBMFrame_list.append(GBMFrame(quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                                              sc_pos_X=scx, sc_pos_Y=scy, sc_pos_Z=scz))
            self._GBMFrame_list = np.array(GBMFrame_list)

            #get the postion of the PS in the sat frame for every timestep
            self._ps_pos_sat_list = []
            for i in range(0, len(GBMFrame_list)):
                ps_pos_sat = self._ps_skycoord.transform_to(GBMFrame_list[i])
                az = ps_pos_sat.lon.deg
                zen = ps_pos_sat.lat.deg
                self._ps_pos_sat_list.append([np.cos(zen * (np.pi / 180)) * np.cos(az * (np.pi / 180)),
                                              np.cos(zen * (np.pi / 180)) * np.sin(az * (np.pi / 180)),
                                              np.sin(zen * (np.pi / 180))])
            self._ps_pos_sat_list = np.array(self._ps_pos_sat_list)

            # get the point of the grid closet to the ps pointing (save the corresponding index)
            best_grid_point_index = []
            for i in range(len(self._ps_pos_sat_list)):
                res_vector_norm = norm(self._points_grid - self._ps_pos_sat_list[i], axis=1)
                best_grid_point_index.append(np.argmax(res_vector_norm))
            self._best_grid_point_index = np.array(best_grid_point_index)

            # calculate the separation of the earth and the ps for every time step
            separation = []
            for earth_position in self._earth_positions:
                separation.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)
            self._separation = np.array(separation)
            # define the earth opening angle
            earth_opening_angle = 67

            # get a list with the rates of the rates of the closest points
            ps_rate_list = []
            for i in range(len(self._best_grid_point_index)):
                # check if not occulted by earth
                if self._separation[i] > earth_opening_angle:
                    ps_rate_list.append(self._rates_points[self._best_grid_point_index[i]])
                else:
                    ps_rate_list.append(np.zeros_like(self._rates_points[self._best_grid_point_index[i]]))
        self._ps_rate_list = np.array(ps_rate_list).T #for all echans
        #interpolate this
        self._earth_rate_interpolator = interpolate.interp1d(self._interpolation_time, self._ps_rate_list)

    def ps_rate_array(self, met, echan):

        return self._earth_rate_interpolator(met)[echan]

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

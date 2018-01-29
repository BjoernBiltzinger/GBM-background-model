import astropy.coordinates as coord
import astropy.units as u
import astropy.time as astro_time


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


class PointSrc(object):

    def __init__(self, name, ra, dec, data):
        self._name = name
        self._data = data #type: ContinuousData
        self._ps_skycoord = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        self._interpolation_time = self._data.interpolation_time #type: ContinuousData.interpolation_time
        self._calc_src_occ()
        self._set_relative_location()
        #self._cleanup()

    def _calc_src_occ(self):
        """

        :return:
        """

        src_occ_ang = []
        earth_positions = self._data.earth_position #type: ContinuousData.earth_position

        # define the size of the earth
        earth_radius = 6371000.8  # geometrically one doesn't need the earth radius at the satellite's position. Instead one needs the radius at the visible horizon. Because this is too much effort to calculate, if one considers the earth as an ellipsoid, it was decided to use the mean earth radius.
        atmosphere = 12000.  # the troposphere is considered as part of the atmosphere that still does considerable absorption of gamma-rays
        r = earth_radius + atmosphere  # the full radius of the occulting earth-sphere
        sat_dist = 6912000.
        earth_opening_angle = math.asin(r / sat_dist) * 360. / (2. * math.pi)  # earth-cone

        with progress_bar(len(self._interpolation_time) - 1, title='Calculating earth occultation of point source') as p:
            for earth_position in earth_positions:
                src_occ_ang.append(coord.SkyCoord.separation(self._ps_skycoord, earth_position).value)

                p.increase()

        src_occ_ang[src_occ_ang < earth_opening_angle] = 0.

        self._src_occ_ang = np.array(src_occ_ang)

        self._occulted_time = np.array(self._interpolation_time)

        self._occulted_time[np.where(self._src_occ_ang != 0)] = 0

        del src_occ_ang, earth_positions


    def _zero(self):
        print "Numpy where condition true"
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
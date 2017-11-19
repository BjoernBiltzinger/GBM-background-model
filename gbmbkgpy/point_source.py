import astropy.coordinates as coord
import astropy.units as u
import astropy.time as astro_time


import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list


import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.utils.continuous_data import ContinuousData

from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file
from gbmbkgpy.utils.progress_bar import progress_bar


class PointSource(object):

    def __init__(self, name, ra, dec, data_in):
        self._name = name
        self._data_in = data_in
        self._ps_skycoord = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        self._set_relative_location()


    def _set_relative_location(self):

        """
        look at continous data sun stuff (setup_geometry)

        coordinate is _pointing of the detectors
        calculate seperation of detectors and point source
        store in arrays for time bins / sub sample time

        interpolate and create a function

        """

        sep_angle = []
        interpolation_time = ContinuousData.interpolation_time.fget(self._data_in)
        pointing = ContinuousData.pointing.fget(self._data_in)

        # go through a subset of times and calculate the sun angle with GBM geometry

        with progress_bar(len(interpolation_time),title='Calculating point source seperation angles') as p:

            for point in pointing:
                sep_angle.append(coord.SkyCoord.separation(self._ps_skycoord,point))

                p.increase()

        # interpolate it

        self._point_source_interpolator = interpolate.interp1d(interpolation_time, sep_angle)


    def separation_angle(self, met):
        """
        call interpolated function and return separation for met (mid eval time)
        """
        return self._point_source_interpolator(met)



    @property
    def location(self):

        return self._ps_skycoord


    @property
    def name(self):

        return self._name
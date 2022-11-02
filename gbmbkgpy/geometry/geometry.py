import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord


class Geometry:

    def galactic_to_satellite(self, time, l, b):
        coord = SkyCoord(l=l*u.degree,
                         b=b*u.degree,
                         frame='galactic')
        coord_icrs = coord.transform_to("icrs")

        ra = coord_icrs.ra.deg
        ra[l > 180] -= 360

        dec = coord_icrs.dec.deg

        return self.icrs_to_satellite(time,
                                      ra,
                                      dec)

    def satellite_to_galactic(self, time, az, el):
        """
        sat to galactic coord transformation
        """

        ra_icrs, dec_icrs = self.satellite_to_icrs(time,
                                                   az,
                                                   el)

        coord = SkyCoord(ra=ra_icrs*u.degree,
                         dec=dec_icrs*u.degree,
                         frame='icrs')
        coord_gal = coord.transform_to("galactic")

        l = coord_gal.l.deg
        l[l > 180] -= 360

        b = coord_gal.b.deg

        return l, b

    def cr_tracer(self, time):
        """
        Returns CR tracer for given times
        :param time: times of interest (array or float)
        """
        raise NotImplementedError("Must be implented in sub-class")

    def icrs_to_satellite(self, time, ra, dec):
        """
        Transform icrs coords to satellite coords
        :param time: time of interest
        :param ra: ra in icrs (degree) (array or float)
        :param dec: dec in icrs (degree) (array or float)
        :returns: az, el in sat frame (degree)
        """
        raise RuntimeError("Has to be implemented in sub-class")

    def satellite_to_icrs(self, time, az, el):
        """
        Transform satellite coords to icrs coords
        :param time: time of interest
        :param az: az in sat frame (degree) (array or float)
        :param el: el in sat frame (degree) (array or float)
        :returns: ra, dec in icrs frame (degree)
        """
        raise RuntimeError("Has to be implemented in sub-class")

    def is_occulted(self, time, ra, dec):
        """
        Check if a position defined by ra and dec (in ICRS) is occulted at
        the given time
        :param time: time of interest (float)
        :param ra: ra of source (array or float)
        :param dec: dec of source (array or float)
        :returns: bool
        """

        raise RuntimeError("Has to be implemented in sub-class")

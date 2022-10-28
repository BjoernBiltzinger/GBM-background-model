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

    @property
    def cr_tracer_diff(self, time):
        """
        Returns the McIlwain L-parameter difference for the satellite position
        for a given time and the minumum mcl value
        """
        raise NotImplementedError("Must be implented in sub-class")

    def icrs_to_satellite(self, time, ra, dec):
        """
        Transform icrs coords to satellite coords
        """
        raise RuntimeError("Has to be implemented in sub-class")

    def satellite_to_icrs(self, time, ra, dec):
        """
        Transform icrs coords to satellite coords
        """
        raise RuntimeError("Has to be implemented in sub-class")

    def is_occulted(self, time, ra, dec):
        """
        Check if a position defined by ra and dec (in ICRS) is occulted at
        the given time
        """
        raise RuntimeError("Has to be implemented in sub-class")

import astropy.coordinates as coord
import astropy.units as u


class PointSource(object):

    def __init__(self, name, ra, dec):
        self._name = name
        self._ps_skycoord = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')


    def set_relative_location(self, time, coordinate):

        # get the separation

        #interpolate
        pass



    def separation_angle(self, met):
        pass



    @property
    def location(self):

        return self._ps_skycoord


    @property
    def name(self):

        return self._name
import astropy.io.fits as fits
import numpy as np
from gbmbkgpy.external_prop import ExternalProps, writefile

class ModelBackground(object):

    def __init__(self):
        self._set_data()

    def __call__(self):
        self._set_data()
        self._build_model()
        self._plot_data()

    @property
    def set_data(self):
        return self._set_data()

    @property
    def build_model(self):
        return self._build_model()

    @property
    def plot_data(self):
        return self._plot_data()


    def _set_data(self):
        #get data from external_prop
        #get continuous data
        self._data

    def _build_model(self):
        return _model

    def _plot_data(self):
        return _plots

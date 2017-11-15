import astropy.io.fits as fits
import numpy as np
from gbmbkgpy.external_prop import ExternalProps, writefile

class ModelBackground(object):

    def __init__(self):
        self._a = None
        self._b = None

    def __call__(self, x):
        return self._a * x + self._b

    def set_fit_parameters(self, a, b):
        self._a = a
        self._b = b

    def set_data(self):
        #get data from external_prop
        #get continuous data
        pass

    def _build_model(self, x):
        self._a = "calculating parameter"
        self._b = "calculating parameter"

    def plot_model(self, model):
        pass

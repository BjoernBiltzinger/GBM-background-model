---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Apply to other satellites


This code can also be used for experiments other than Fermi/GBM. All you need is a response generator, that can generate the response for any given position in the satellite frame and some information about the data and geometry (orbit, pointing,...). In this notebook we see how to use gbmbkgpy for a new mock satellite: All we have to do is write a new Data, Geometry and ResponseGenerator class that inherit from the base classes of gbmbkgpy. Let's start with a few imports first:


## Imports

```python
from gbmbkgpy.data.data import Data
from gbmbkgpy.geometry.geometry import Geometry
from gbmbkgpy.response.response import ResponseGenerator

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="notebook", fscale=1, grid=False)

from astromodels import Powerlaw, Exponential_cutoff, Constant, Band
from astromodels.utils.configuration import astromodels_config
astromodels_config.modeling.ignore_parameter_bounds = True

from gbmbkgpy.modeling.new_astromodels import SBPL

from gbmbkgpy.response.response_precalculation import ResponsePrecalculation
from gbmbkgpy.response.src_response import EarthResponse, CGBResponse, GalacticCenterResponse, PointSourceResponse

from gbmbkgpy.modeling.source import PhotonSourceFixed, SAASource, NormOnlySource
from gbmbkgpy.modeling.functions import AstromodelFunctionVector

from gbmbkgpy.modeling.model import ModelDet

from gbmbkgpy.io.plotting.plot import plot_lightcurve
```

## Define the Classes we need for our newly developed satellite

<!-- #region tags=[] -->
For simplicity assume we already know the time bins and data and do not have to read them from a file
<!-- #endregion -->

```python
class NewSatData(Data):
    
    def __init__(self, name, time_bins, counts):
        super().__init__(name, time_bins, counts)
        
```

The Geometry of our new satellite is very simple. Its coordinate system is always aligned with the icrs coordinate system and points with dec<-30 degree are occulted by the earth. Also we have found a new exciting tracer for CR here in our mock universe, which is just the square of a sinus.

```python
class NewSatGeometry(Geometry):
    
    def cr_tracer(self, time):
        return np.sin(time/100.)**2
    
    def icrs_to_satellite(self, time, ra, dec):
        return ra, dec
    
    def satellite_to_icrs(self, time, ra, dec):
        return ra, dec
    
    def is_occulted(self, time, ra, dec):
        try:
            res = np.zeros(len(ra), dtype=bool)
            res[dec<-30] = True
        
            return res
        except TypeError:
            return dec<-30
    
```

The response of out newly developed satellite is also interesting, because it randomly change every time we start this analysis also the response scales linear with np.abs((zen/90)).

```python
class NewSatResponseGenerator(ResponseGenerator):
    
    def __init__(self, geometry, Ebins_in_edge, data):
        self._num_echan = data.num_echan
        
        self._mat = 1*np.random.rand(len(Ebins_in_edge)-1, self._num_echan)
        
        super().__init__(geometry, Ebins_in_edge, self._num_echan)
        
    def calc_response_az_zen(self, az, zen):
        return np.abs((zen/90))*self._mat
```

## Run General Code

The following code is the same like in the example for GBM data. Therefore we can skip directly to the result plots next.

```python
time_bin_edges = np.linspace(0,1000,101)
time_bins = np.array([time_bin_edges[:-1], time_bin_edges[1:]]).T
counts = np.random.randint(0,10, size=(100,5))
data = NewSatData("dummy", time_bins, counts)
```

```python
# init geometry object for Fermi/GBM at the given date
geom = NewSatGeometry()
```

```python
# General response Generator
drm_gen = NewSatResponseGenerator(geometry=geom, Ebins_in_edge=np.geomspace(10,2000, 101), data=data)
```

```python
# Response precalculation for extended sources
rsp_pre = ResponsePrecalculation(drm_gen, Ngrid=10000)
```

```python
# Time where to calculate the effective responses - linear interpolation in between
interp_time = np.linspace(data.time_bins[0,0], data.time_bins[-1,-1], 80)

# Galactic Center
gc_rsp = GalacticCenterResponse(geometry=geom, interp_times=interp_time, resp_prec=rsp_pre)

# Earth Albedo
earth_rsp = EarthResponse(geometry=geom, interp_times=interp_time, resp_prec=rsp_pre)

# CGB
cgb_rsp = CGBResponse(geometry=geom, interp_times=interp_time, resp_prec=rsp_pre)
```

```python
# Response for Crab
crab_rsp = PointSourceResponse(response_generator=drm_gen, interp_times=interp_time, ra=83.633, dec=22.015)
```

```python
# A point source

# Define spectrum
pl_crab = Powerlaw()
pl_crab.K.value = 39.7
pl_crab.index.value = -2.1

# define source
crab = PhotonSourceFixed("Crab", pl_crab, crab_rsp)


# extended sources

#EARTH
earth_spec = SBPL()
earth_spec.K.value = 0.015
earth_spec.alpha.value = -5
earth_spec.beta.value = 1.72
earth_spec.xb.value = 33.7

earth = PhotonSourceFixed("Earth", earth_spec, earth_rsp)

#CGB
cgb_spec = SBPL()
cgb_spec.K.value = 0.11
cgb_spec.alpha.value = 1.32
cgb_spec.beta.value = 2.88
cgb_spec.xb.value = 30.0
cgb = PhotonSourceFixed("CGB", cgb_spec, cgb_rsp)

#GC
pl1_gc = Powerlaw()
pl1_gc.K.value = 0.08737610581967094
pl1_gc.index.value = -1.45

pl2_gc = Powerlaw()
pl2_gc.K.value = 252.3829377920772
pl2_gc.index.value = -2.9

exp_gc = Exponential_cutoff()
exp_gc.K.value = 0.1036025649336684
exp_gc.xc.value = 8

total = pl1_gc+pl2_gc+exp_gc

gc = PhotonSourceFixed("GC", total, gc_rsp)
```

```python
exp_decay = Exponential_cutoff()
exp_decay.K.value = 50
exp_decay.xc.value = 5
afv_saa = AstromodelFunctionVector(data.num_echan, base_function=exp_decay)

exit_time_saa = 100

saa = SAASource("SAA", exit_time_saa, afv_saa)
```

```python
c = Constant()
c.k.value = 1
afv_cr = AstromodelFunctionVector(data.num_echan, base_function=c)

cr = NormOnlySource("CR", geom.cr_tracer, afv_cr)
```

```python
model = ModelDet(data)

model.add_source(gc)
model.add_source(earth)
model.add_source(cgb)
model.add_source(crab)

model.add_source(saa)
model.add_source(cr)
```

## Plot


Sum of all model components

```python
ax = plot_lightcurve(model, eff_echan=0, show_data=False)
ax.legend();
```

A few individual model components

```python
ax = plot_lightcurve(model, eff_echan=0, show_data=False, show_total_model=False, model_component_list=["CGB", "SAA", "CR"], model_component_colors=["navy", "purple", "red"])
ax.legend();
```

Generate new data from our model

```python
plt.scatter(data.mean_time, model.generate_counts()[:,0])
plt.xlabel("Time [s]")
plt.ylabel("Counts")
```

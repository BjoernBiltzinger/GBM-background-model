# MultiNest Fit routine:

# First we will import the necessary libraries
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.modeling.background_like import BackgroundLike
from gbmbkgpy.modeling.setup_sources import setup_sources
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit

import numpy as np
from mpi4py import MPI

mpi = MPI.COMM_WORLD
rank = mpi.Get_rank()

# Load the Data
date = '150126'
det = 'n5'
file_type = 'ctime'
echan = 4
number_of_SAAs = 10

cd = ContinuousData(date, det, file_type)
ep = ExternalProps(date)

# Build the Model
Source_list = setup_sources(cd, ep, echan)[:(3 + number_of_SAAs)]
model = Model(*Source_list)
background_like = BackgroundLike(cd, model, echan)
model.set_initial_SAA_amplitudes(np.array(cd.saa_initial_values(echan))*100)
model.set_initial_continuum_amplitudes([200,200,200,200])

parameter_bounds = [
  (0, 10**3),
  (0, 10**3),
  (-10**3, 10**4)
]

for i in range(number_of_SAAs):
  parameter_bounds.append((10**-1, 10**7))
  parameter_bounds.append((10**-5, 10**-2))

# Add bounds to the parameters for multinest
model.set_parameter_bounds(parameter_bounds)

#Instantiate Multinest Fit
mn_fit = MultiNestFit(background_like, model.parameters)

# Fit with multinest
mn_fit.minimize()

# Plot Marginals
mn_fit.plot_marginals()

# Multinest Output dir
output_dir = mn_fit.output_dir

# Plot Residuals and posteriour
bin_width = 10
residual_plot = background_like.display_model(min_bin_width=bin_width, show_residuals=True, show_data=True,
                                              plot_sources=True, show_grb_trigger=True)

residual_plot.savefig(output_dir + 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.pdf'.format(date, det, echan, bin_width), dpi=300)

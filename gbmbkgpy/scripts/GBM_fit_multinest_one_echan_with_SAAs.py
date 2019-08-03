import matplotlib
matplotlib.use('Agg')
#imports
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.modeling.background_like import BackgroundLike
from gbmbkgpy.modeling.setup_sources import setup_sources, setup_sources_golbal
from gbmbkgpy.modeling.rate_generator_DRM import Rate_Generator_DRM
from gbmbkgpy.io.downloading import download_files
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit

import numpy as np

from datetime import datetime


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()  

#setup paramters
date='160310'
grb_name='GRB 160310A'
trigger_time='18:42:00.000'
detector = 'nb'
data_type = 'ctime'

#download files with rank=0; all other ranks have to wait!
if rank==0:
    download_files(data_type, detector, date)
    wait=True
else:
    wait=None
wait = comm.bcast(wait, root=0)

# create a Rate_Generator_DRM object that is needed to calculate the rates caused by the earth, the cgb and the Pointsources.
# These calculations use the full DRM's and thus include sat. scattering and partial loss of energy by the photons
rate_gen=Rate_Generator_DRM(detector,date, Ngrid=40000, data_type=data_type)
print('Done with Rate_Gernerator Precalculation')

cd = ContinuousData(date, detector, data_type, rate_gen, use_SAA=True)
print('Done with ContinuousData Precalculation')

ep = ExternalProps(date)
print('Done with ExternalProps Precalculation')

#build model
source_list = []
echan_list=[2] #has to be List! One entry is also possible!
#create the sources which are seperatly for all echans, in this case we only use one echan!
for echan in echan_list:
    source_list += setup_sources(cd,ep,echan,point_source_list=['CRAB'])

#create the global sources
source_list_global = setup_sources_golbal(cd)

#add them together
source_list_total = source_list + source_list_global

print('Done with Source Precalculation - Model is ready')

#Set the model with this
model = Model(*source_list_total)


background_like = BackgroundLike(cd,model,echan_list)

#set the initial amplitudes for the global and not global sources!
model.set_initial_global_amplitudes([0.015, 0.1])
initial_continuum_amplitudes = []
for echan in echan_list:
    initial_continuum_amplitudes.append(50)
    initial_continuum_amplitudes.append(50)
model.set_initial_continuum_amplitudes(initial_continuum_amplitudes)

#set the prior bounds
parameter_bounds= [(10**-3,100),(10**-3,100),(10**-3,100)]
#for the SAA's
num_SAAs = (len(model.parameters)-5)/2
for i in range(num_SAAs):
    parameter_bounds.append((10**-1,10**7)) # Norm of SAA
    parameter_bounds.append((10**-9,10**0)) # decay of SAA
parameter_bounds.append((10**-5,1)) #earth
parameter_bounds.append((10**-5,1)) #cgb
# Add bounds to the parameters for multinest
model.set_parameter_bounds(parameter_bounds)
#Instantiate Multinest Fit
mn_fit = MultiNestFit(background_like, model.parameters)

# Fit with multinest and define the number of live points one wants to use
mn_fit.minimize(n_live_points = 600)

# Plot Marginals
mn_fit.plot_marginals()

# Multinest Output dir
output_dir = mn_fit.output_dir

# Plot Residuals and fit with rank=0 
bin_width = 1
if rank==0:
    for echan in echan_list:
        residual_plot = background_like.display_model(echan, min_bin_width=bin_width, show_residuals=True, show_data=True,
                                                      plot_sources=True, show_grb_trigger=True)

        residual_plot.savefig(output_dir + 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.pdf'.format(date, detector, echan, bin_width), dpi=300)
    print('Whole calculation took: {}'.format(datetime.now() - start))


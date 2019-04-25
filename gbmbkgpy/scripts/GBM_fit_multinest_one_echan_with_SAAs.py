import matplotlib
matplotlib.use('Agg')
#imports
from gbmbkgpy.utils.continuous_data import ContinuousData
from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.modeling.background_like import BackgroundLike
from gbmbkgpy.minimizer.minimizer import Minimizer
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.modeling.setup_sources import setup_sources, setup_sources_golbal
from gbmbkgpy.modeling.rate_generator_DRM import Rate_Generator_DRM
from gbmbkgpy.io.downloading import download_files
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()  

#setup paramters
date='180101'
grb_name='SWIFT_MAXI_trigger'
trigger_time='13:54:09'
#grb_name_2 = 'TGF trigger 181201277'
#trigger_time_2 = '04:30:07.000'
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

cd = ContinuousData(date, detector, data_type, rate_gen, use_SAA=True, clean_SAA=True)
print('Done with ContinuousData Precalculation')

ep = ExternalProps(date)
print('Done with ExternalProps Precalculation')

#build model
source_list = []
echan_list=[5] #has to be List! One entry is also possible!
#create the sources which are seperatly for all echans, in this case we only use one echan!
for echan in echan_list:
    source_list += setup_sources(cd,ep,echan)

#create the global sources
source_list_global = setup_sources_golbal(cd, ep, point_source_list=['CRAB'])

#add them together
source_list_total = source_list + source_list_global
if rank==0:
    print(source_list_total)
print('Done with Source Precalculation - Model is ready')

#Set the model with this
model = Model(*source_list_total)

print model.parameters

background_like = BackgroundLike(cd,model,echan_list)

#set the prior bounds
parameter_bounds = [(10**(-6),1000),(10**(-6),1000)] #Magnetic shielding 
#for the SAA's
num_SAAs = (len(model.parameters)-5)/2
for i in range(num_SAAs):
    parameter_bounds.append((1,10**4)) # Norm of SAA
    parameter_bounds.append((10**-5,10**-1)) # decay of SAA

parameter_bounds.append((10**-4,0.1)) #earth
parameter_bounds.append((10**-4,0.2)) #cgb
parameter_bounds.append((10**-4,100)) #crab

# Add bounds to the parameters for multinest
model.set_parameter_bounds(parameter_bounds)

#Instantiate Multinest Fit
mn_fit = MultiNestFit(background_like, model.parameters)

# Fit with multinest and define the number of live points one wants to use
mn_fit.minimize(n_live_points = 800)

# Plot Marginals
mn_fit.plot_marginals()

# Multinest Output dir
output_dir = mn_fit.output_dir

# Plot Residuals and fit with rank=0 
bin_width = 60
if rank==0:
    
    for echan in echan_list:
        
        residual_plot = background_like.display_model(echan, min_bin_width=bin_width, show_residuals=True, show_data=True,change_time=False,
                                                      plot_sources=True, show_grb_trigger=True, ppc=True, result_dir=output_dir)

        residual_plot.savefig(output_dir + 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.jpeg'.format(date, detector, echan, bin_width), dpi=300)

    print('Whole calculation took: {}'.format(datetime.now() - start))

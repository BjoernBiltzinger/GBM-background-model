import matplotlib
matplotlib.use('Agg')
#imports
from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.data.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.fitting.background_like import BackgroundLike
from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.geometry_calc import Geometry
from gbmbkgpy.utils.response_precalculation import Response_Precalculation
from gbmbkgpy.io.downloading import download_files
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot import Plotter
from gbmbkgpy.modeling.setup_sources import Setup

import numpy as np

from datetime import datetime


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()  

####################### Setup parameters ###############################
date='160310'
grb_name='GRB 160310A'
trigger_time='18:42:00.000'
detector = 'nb'
data_type = 'ctime'

############################# Data ######################################


#download files with rank=0; all other ranks have to wait!
if rank==0:
    download_files(data_type, detector, date)
    wait=True
else:
    wait=None
wait = comm.bcast(wait, root=0)

# Create the data object for the wanted day and detector

data = Data(date, detector, data_type)
print('Done with Data preparation')

# Create external properties object (for McIlwain L-parameter)

ep = ExternalProps(date)
print('Done with ExternalProps Precalculation')

############################# Model ########################################

################## Response precalculation ################

# Create a Response precalculation object, that precalculates the responses on a spherical grid arount the detector.
# These calculations use the full DRM's and thus include sat. scattering and partial loss of energy by the photons.

resp = Response_Precalculation(detector,date, Ngrid=40000, data_type=data_type)
print('Done with Rate_Gernerator Precalculation')

################## SAA precalculation ######################

######### SAA options ###########
# Use time after SAA? If no give the time which should be deleted after every SAA in s. If use_SAA=False no SAA sources
# are build
use_SAA = False
time_after_SAA = 5000
# Want to use separated time intervals that are shorter than 1000 seconds?
short_time_intervals = False
################################

# Build the SAA object

saa_calc = SAA_calc(data, use_SAA=use_SAA, time_after_SAA=time_after_SAA, short_time_intervals=short_time_intervals)


################### Geometry precalculation ##################

########## Geom options ###########

# For how many times during the day do you want to calculate the geometry? In between a linear interpolation is used.
n_bins_to_calculate=800
###################################

geom = Geometry(data, detector, date, n_bins_to_calculate)

##################### Setup Sources ##########################

# Create all individual sources and add them to a list

########## Setup options ###########
# List with all echans you want to use
echan_list = [2] #has to be  List! One entry is also possible

# Use CosmicRay source?
use_CR= True
# Use EarthAlbedo source?
use_Earth=True
# Use CGB source?
use_CGB=True
# Which PS should be included (given as list of names)
ps_list = ['CRAB']
# Fix the spectrum of the earth albedo?
fix_earth = False
# Fix the spectrum of the CGB?
fix_cgb = False
####################################

source_list = Setup(data, saa_calc, echan_list=echan_list, response_object=resp, albedo_cgb_object=albedo_cgb_obj,
                    use_SAA=use_SAA, use_CR=use_CR, use_Earth=use_Earth, use_CGB=use_CGB, point_source_list=ps_list,
                    fix_Earth=fix_earth, fix_CGB=fix_cgb)

print('Done with Source Precalculation - Model is ready')

###################### Setup Model #############################

model = Model(*source_list)


##################### Prior bounds #############################

######## Define bounds for all sources ###############

# SAA: Amplitude and decay constant
saa_bounds = [(1, 10**4), (10**-5, 10**-1)]

# CR: Constant and McIlwain normalization
cr_bounds = [(0.1,100), (0.1,100)]

# Amplitude of PS spectrum
ps_bound = [(1,100)]

# If earth spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
if fix_earth:
    earth_bound = [(0.001, 1)]
else:
    earth_bound = [(0.001, 1), (-8, -3), (1.1, 1.9), (20, 40)]

# If cgb spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
if fix_cgb:
    cgb_bound = [(0.01, 0.5)]
else:
    cgb_bound = [(0.01, 0.5), (0.5, 1.7), (2.2, 3.1), (27, 40)]

#######################################

parameter_bounds = []

# Echan individual sources
for i in echan_list:
    if use_SAA:
        for j in range(saa_calc.num_saa):
            parameter_bounds.append(saa_bounds)
    if use_CR:
        parameter_bounds.append(cr_bounds)
# Global sources for all echans
parameter_bounds.append(ps_bound)
parameter_bounds.append(earth_bound)
parameter_bounds.append(cgb_bound)

# Concatenate this
parameter_bounds = np.concatenate(parameter_bounds)

# Add bounds to the parameters for multinest
model.set_parameter_bounds(parameter_bounds)

################################## Backgroundlike Class #################################

# Class that calcualtes the likelihood
background_like = BackgroundLike(data, model, echan_list)

################################## Fitting ###############################################

#############Multinest options ##############

# Number of live points for mulitnest?
num_live_points = 600

# const_efficiency_mode of mulitnest?
const_efficiency_mode = True

#############################################

#Instantiate Multinest Fit
mn_fit = MultiNestFit(background_like, model.parameters)

# Fit with multinest and define the number of live points one wants to use
mn_fit.minimize(n_live_points=num_live_points, const_efficiency_mode=const_efficiency_mode)

# Plot Marginals
mn_fit.plot_marginals()

# Multinest Output dir
output_dir = mn_fit.output_dir

################################# Plotting ################################################

############## Plot options #################
# Choose a bin width to bin the data
bin_width = 1

# Change time to seconds from midnight?
change_time = True

# Show Residuals?
show_residuals = False

# Show data?
show_data = True

# Plot best fit of individual sources?
plot_sources = True

# Display the GRB trigger time? What times and names? Time in '20:57:03.000' format.
show_grb_trigger = True
times_mark = []
names_mark = []

# Create PPC plots?
ppc = True

# Plot Results with rank 0

if rank == 0:

    # Create Plotter object that creates the plots
    plotter = Plotter(data, model, echan_list)

    # Create one plot for every echan and save it
    for echan in echan_list:
        residual_plot = plotter.display_model(echan, min_bin_width=bin_width, show_residuals=show_residuals,
                                              show_data=show_data, plot_sources=plot_sources,
                                              show_grb_trigger=show_grb_trigger, change_time=change_time, ppc=ppc,
                                              result_dir=output_dir)
        residual_plot.savefig(output_dir + 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.pdf'.format(
            date, detector, echan, bin_width), dpi=300)

    # Print the duration of the script
    print('Whole calculation took: {}'.format(datetime.now() - start))


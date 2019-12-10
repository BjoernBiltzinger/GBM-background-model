import matplotlib

matplotlib.use('Agg')
# imports
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
from gbmbkgpy.modeling.albedo_cgb import Albedo_CGB_fixed, Albedo_CGB_free
from gbmbkgpy.io.package_data import get_path_of_external_data_dir                                                               
from gbmbkgpy.modeling.sun import Sun
from gbmbkgpy.io.file_utils import file_existing_and_readable

import os
from shutil import copyfile
import sys


### Argparse for passing custom_config file
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config_file', type=str, help='Name of the config file located in gbm_data/fits/')
parser.add_argument('-date', '--date', type=str, help='Date string')
parser.add_argument('-det', '--detector', type=str, help='Name detector')
parser.add_argument('-e', '--echan', type=int, help='Echan number')
args = parser.parse_args()

### Config file directories
config_default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_default.py')
config_custom_path = os.path.join(get_path_of_external_data_dir(), 'fits', 'config_custom.py')
config_custom_dir = os.path.join(get_path_of_external_data_dir(), 'fits')

import numpy as np

from datetime import datetime

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()

NO_REBIN = 1E-99

def print_progress(text):
    """
    Helper function that prints the input text only with rank 0
    """
    if rank == 0:
        print(text)


if args.config_file is not None:
    sys.path.append(config_custom_dir)
    module = __import__(args.config_file, globals(), locals(), ['*'])
    for k in dir(module):
        locals()[k] = getattr(module, k)

    config_custom_path = os.path.join(get_path_of_external_data_dir(), 'fits', args.config_file + '.py')
    print_progress('Using custom config file from {}'.format(config_custom_path))
    custom = True

# Import Config file, use the custom config file if it exists, otherwise use default config file
elif file_existing_and_readable(config_custom_path):
    sys.path.append(config_custom_dir)
    from config_custom import *
    print_progress('Using custom config file from {}'.format(config_custom_path))
    custom = True

else:
    from config_default import *
    print_progress('Using default config file')
    custom = False


####################### Setup parameters ###############################
date = general_dict['dates']
detector = general_dict['detector']
data_type = general_dict['data_type']
# List with all echans you want to use
echan_list = general_dict['echan_list']  # has to be  List! One entry is also possible
min_bin_width = general_dict.get('min_bin_width', NO_REBIN)

################# Overwrite with BASH arguments #########################
if args.date is not None:
    date = [args.date]
if args.detector is not None:
    detector = args.detector
if args.echan is not None:
    echan_list = [args.echan]
############################# Data ######################################

# download files with rank=0; all other ranks have to wait!
print_progress('Download data...')
if rank == 0:
    for d in date:
        download_files(data_type, detector, d)
    wait = True
else:
    wait = None
wait = comm.bcast(wait, root=0)
print_progress('Done')
# Create the data object for the wanted day and detector

print_progress('Prepare data...')
data = Data(date, detector, data_type, echan_list)
print_progress('Done')


################## SAA precalculation ######################

######### SAA options ###########
# Use time after SAA? If no give the time which should be deleted after every SAA in s. If use_SAA=False no SAA sources
# are build
time_after_SAA = saa_dict['time_after_saa']
# Want to use separated time intervals that are shorter than 1000 seconds?
short_time_intervals = saa_dict['short_time_intervals']
nr_decays = saa_dict['nr_decays']
################################

# Build the SAA object
if time_after_SAA is not None:
    print_progress('Precalculate SAA times and SAA mask. {} seconds after every SAA exit are excluded from fit...')
else:
    print_progress('Precalculate SAA times and SAA mask...')
saa_calc = SAA_calc(data, time_after_SAA=time_after_SAA, short_time_intervals=short_time_intervals, nr_decays=nr_decays)
print_progress('Done')

if min_bin_width > NO_REBIN:
    data.rebinn_data(min_bin_width, saa_calc.saa_mask)
    saa_calc.set_rebinned_saa_mask(data.rebinned_saa_mask)


# Create external properties object (for McIlwain L-parameter)
print_progress('Download and prepare external properties...')
ep = ExternalProps(date, det=detector, bgo_cr_approximation=setup_dict['bgo_cr_approximation'])
print_progress('Done')

############################# Model ########################################

################## Response precalculation ################

# Create a Response precalculation object, that precalculates the responses on a spherical grid arount the detector.
# These calculations use the full DRM's and thus include sat. scattering and partial loss of energy by the photons.
Ngrid = response_dict['Ngrid']
print_progress('Precalculate responses for {} points on sphere around detector...'.format(Ngrid))
resp = Response_Precalculation(detector, date, echan_list, Ngrid=Ngrid, data_type=data_type)
print_progress('Done')

################### Geometry precalculation ##################

########## Geom options ###########

# For how many times during the day do you want to calculate the geometry? In between a linear interpolation is used.
n_bins_to_calculate = geom_dict['n_bins_to_calculate']
###################################
print_progress('Precalculate geometry for {} times during the day...'.format(n_bins_to_calculate))
geom = Geometry(data, detector, date, n_bins_to_calculate)
print_progress('Done')

##################### Setup Sources ##########################

# Create all individual sources and add them to a list

########## Setup options ###########
assert (setup_dict['fix_earth'] and setup_dict['fix_cgb']) or (not setup_dict['fix_earth'] and not setup_dict['fix_cgb']),\
    'At the moment albeod and cgb spectrum have to be either both fixed or both free'

########### Albedo-CGB Object ###########
if setup_dict['fix_earth']:
    albedo_cgb_obj = Albedo_CGB_fixed(resp, geom)
else:
    albedo_cgb_obj = Albedo_CGB_free(resp, geom)

sun_obj = Sun(resp, geom, echan_list)
    
print_progress('Create Source list...')

source_list = Setup(data=                   data,
                    saa_object=             saa_calc,
                    ep=                     ep,
                    geom_object=            geom,
                    sun_object=             sun_obj,
                    echan_list=             echan_list,
                    response_object=        resp,
                    albedo_cgb_object=      albedo_cgb_obj,
                    use_saa=                setup_dict['use_saa'],
                    use_constant=           setup_dict['use_constant'],
                    use_cr=                 setup_dict['use_cr'],
                    use_earth=              setup_dict['use_earth'],
                    use_cgb=                setup_dict['use_cgb'],
                    point_source_list=      setup_dict['ps_list'],
                    fix_ps=                 setup_dict['fix_ps'],
                    fix_earth=              setup_dict['fix_earth'],
                    fix_cgb=                setup_dict['fix_cgb'],
                    use_sun=                setup_dict['use_sun'],
                    nr_saa_decays=          saa_dict['nr_decays'],
                    bgo_cr_approximation=   setup_dict['bgo_cr_approximation'])


print_progress('Done')

###################### Setup Model #############################
print_progress('Build model with source_list...')
model = Model(*source_list)
print_progress('Done')

##################### Prior bounds #############################

######## Define bounds for all sources ###############
######## Define gaussian parameter for all sources ###############

parameter_bounds = {}

# Echan individual sources
for e in echan_list:

    if setup_dict['use_saa']:

        # If fitting only one day add additional 'SAA' decay to account for leftover excitation
        if len(date) == 1:
            offset = saa_dict['nr_decays']
        else:
            offset = 0

        for saa_nr in range(saa_calc.num_saa + offset):
            parameter_bounds['norm_saa-{}_echan-{}'.format(saa_nr, e)] = {
                'bounds': bounds_dict['saa_bound'][0],
                'gaussian_parameter': gaussian_dict['saa_bound'][0]
            }
            parameter_bounds['decay_saa-{}_echan-{}'.format(saa_nr, e)] = {
                'bounds': bounds_dict['saa_bound'][1],
                'gaussian_parameter': gaussian_dict['saa_bound'][1]
            }

    if setup_dict['use_constant']:
        parameter_bounds['constant_echan-{}'.format(e)] = {
            'bounds': bounds_dict['cr_bound'][0],
            'gaussian_parameter': gaussian_dict['cr_bound'][0]
        }

    if setup_dict['use_cr']:
        parameter_bounds['norm_magnetic_echan-{}'.format(e)] = {
            'bounds': bounds_dict['cr_bound'][1],
            'gaussian_parameter': gaussian_dict['cr_bound'][1]
        }

if setup_dict['use_sun']:
    parameter_bounds['sun_C'] = {
        'bounds': bounds_dict['sun_bound'][0],
        'gaussian_parameter': gaussian_dict['sun_bound'][0]
    }
    parameter_bounds['sun_index'] = {
        'bounds': bounds_dict['sun_bound'][1],
        'gaussian_parameter': gaussian_dict['sun_bound'][1]
    }
# Global sources for all echans

# If PS spectrum is fixed only the normalization, otherwise C, index
for i, ps in enumerate(setup_dict['ps_list']):
    if setup_dict['fix_ps'][i]:
        parameter_bounds['norm_point_source-{}'.format(ps)] = {
            'bounds': bounds_dict['ps_fixed_bound'][0],
            'gaussian_parameter': gaussian_dict['ps_fixed_bound'][0]
        }
    else:
        parameter_bounds['ps_{}_spectrum_fitted_C'.format(ps)] = {
            'bounds': bounds_dict['ps_free_bound'][0],
            'gaussian_parameter': gaussian_dict['ps_free_bound'][0]
        }
        parameter_bounds['ps_{}_spectrum_fitted_index'.format(ps)] = {
            'bounds': bounds_dict['ps_free_bound'][1],
            'gaussian_parameter': gaussian_dict['ps_free_bound'][1]
        }


if setup_dict['use_earth']:
    # If earth spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
    if setup_dict['fix_earth']:
        parameter_bounds['norm_earth_albedo'] = {
            'bounds': bounds_dict['earth_fixed_bound'][0],
            'gaussian_parameter': gaussian_dict['earth_fixed_bound'][0]
        }
    else:
        parameter_bounds['earth_albedo_spectrum_fitted_C'] = {
            'bounds': bounds_dict['earth_free_bound'][0],
            'gaussian_parameter': gaussian_dict['earth_free_bound'][0]
        }
        parameter_bounds['earth_albedo_spectrum_fitted_index1'] = {
            'bounds': bounds_dict['earth_free_bound'][1],
            'gaussian_parameter': gaussian_dict['earth_free_bound'][1]
        }
        parameter_bounds['earth_albedo_spectrum_fitted_index2'] = {
            'bounds': bounds_dict['earth_free_bound'][2],
            'gaussian_parameter': gaussian_dict['earth_free_bound'][2]
        }
        parameter_bounds['earth_albedo_spectrum_fitted_break_energy'] = {
            'bounds': bounds_dict['earth_free_bound'][3],
            'gaussian_parameter': gaussian_dict['earth_free_bound'][3]
        }

if setup_dict['use_cgb']:
    # If cgb spectrum is fixed only the normalization, otherwise C, index1, index2 and E_break
    if setup_dict['fix_cgb']:
        parameter_bounds['norm_cgb'] = {
            'bounds': bounds_dict['cgb_fixed_bound'][0],
            'gaussian_parameter': gaussian_dict['cgb_fixed_bound'][0]
        }
    else:
        parameter_bounds['CGB_spectrum_fitted_C'] = {
            'bounds': bounds_dict['cgb_free_bound'][0],
            'gaussian_parameter': gaussian_dict['cgb_free_bound'][0]
        }
        parameter_bounds['CGB_spectrum_fitted_index1'] = {
            'bounds': bounds_dict['cgb_free_bound'][1],
            'gaussian_parameter': gaussian_dict['cgb_free_bound'][1]
        }
        parameter_bounds['CGB_spectrum_fitted_index2'] = {
            'bounds': bounds_dict['cgb_free_bound'][2],
            'gaussian_parameter': gaussian_dict['cgb_free_bound'][2]
        }
        parameter_bounds['CGB_spectrum_fitted_break_energy'] = {
            'bounds': bounds_dict['cgb_free_bound'][3],
            'gaussian_parameter': gaussian_dict['cgb_free_bound'][3]
        }

# Add bounds to the parameters for multinest
model.set_parameter_bounds(parameter_bounds)


################################## Backgroundlike Class #################################

# Class that calcualtes the likelihood
print_progress('Create BackgroundLike class that conects model and data...')
background_like = BackgroundLike(data, model, saa_calc, echan_list)
print_progress('Done')

################################## Fitting ###############################################

############# Multinest options ##############

# Number of live points for mulitnest?
num_live_points = multi_dict['num_live_points']

# const_efficiency_mode of mulitnest?
const_efficiency_mode = multi_dict['constant_efficiency_mode']

#############################################

# Instantiate Multinest Fit
mn_fit = MultiNestFit(background_like, model.parameters)
# Fit with multinest and define the number of live points one wants to use
mn_fit.minimize_multinest(n_live_points=num_live_points, const_efficiency_mode=const_efficiency_mode)

# Plot Marginals
mn_fit.plot_marginals()

# Multinest Output dir
output_dir = mn_fit.output_dir

################################# Plotting ################################################

############## Plot options #################
# Choose a bin width to bin the data
bin_width = plot_dict['bin_width']

# Change time to seconds from midnight?
change_time = plot_dict['change_time']

# Show Residuals?
show_residuals = plot_dict['show_residuals']

# Show data?
show_data = plot_dict['show_data']

# Plot best fit of individual sources?
plot_sources = plot_dict['plot_sources']

# Display the GRB trigger time? What times and names? Time in '20:57:03.000' format.
show_grb_trigger = plot_dict['show_grb_trigger']
times_mark = plot_dict['times_mark']
names_mark = plot_dict['names_mark']

# Create PPC plots?
ppc = plot_dict['ppc']

# Time range for plot (as tuple, e.g. (5000,35000))
xlim = plot_dict['xlim']

# Count rates range for plot (as tuple, e.g. (0,60))
ylim = plot_dict['ylim']

# Should the legend be outside of the plot?
legend_outside = plot_dict['legend_outside']

print_progress('Create Plotter object...')
# Create Plotter object that creates the plots
plotter = Plotter(data, model, saa_calc, echan_list)
print_progress('Done')

# Create one plot for every echan and save it
for index, echan in enumerate(echan_list):
    print_progress('Create Plots for echan {} ...'.format(echan))
    residual_plot = plotter.display_model(index, min_bin_width=bin_width, show_residuals=show_residuals,
                                          show_data=show_data, plot_sources=plot_sources,
                                          show_grb_trigger=show_grb_trigger, change_time=change_time, ppc=ppc,
                                          result_dir=output_dir, xlim=xlim, ylim=ylim, legend_outside=legend_outside)
    if rank == 0:
        residual_plot.savefig(output_dir + 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.pdf'.format(
            date, detector, echan, bin_width), dpi=300)
plotter._save_plotting_data(output_dir + 'data_for_plots.hdf5', output_dir, echan_list)
if custom:
    copyfile(config_custom_path, output_dir + 'used_config.py')
else:
    copyfile(config_default_path, output_dir + 'used_config.py')
print_progress('Done')
# Print the duration of the script
print('Whole calculation took: {}'.format(datetime.now() - start))

import matplotlib

matplotlib.use('Agg')

import os
import yaml
from datetime import datetime

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot import Plotter
from gbmbkgpy.io.export import DataExporter

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()


def print_progress(text):
    """
    Helper function that prints the input text only with rank 0
    """
    if rank == 0:
        print(text)


############## Argparse for parsing bash arguments ################
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config_file', type=str, help='Name of the config file located in gbm_data/fits/')
parser.add_argument('-date', '--date', type=str, help='Date string')
parser.add_argument('-det', '--detector', type=str, help='Name detector')
parser.add_argument('-e', '--echan', type=int, help='Echan number')
args = parser.parse_args()

if args.config_file is not None:
    config_file = args.config_file
    print_progress('Using custom config file from {}'.format(args.config_file))

else:
    config_file = 'config_default.yml'
    print_progress('Using default config file')

# Load the config.yml
with open(config_file) as f:
    config = yaml.load(f)


############# Overwrite config with BASH arguments ################
if args.date is not None:
    config['general']['dates'] = [args.date]
if args.detector is not None:
    config['general']['detector'] = args.detector
if args.echan is not None:
    config['general']['echan_list'] = [args.echan]


############## Generate the GBM-background-model ##################
model_generator = BackgroundModelGenerator()

model_generator.from_config_dict(config)


############### Instantiate Minimizer #############################
if config['fit']['method'] == 'multinest':
    minimizer = MultiNestFit(
        likelihood=model_generator.likelihood,
        parameters=model_generator.model.free_parameters
    )

    # Fit with multinest and define the number of live points one wants to use
    minimizer.minimize_multinest(
        n_live_points=config['fit']['multinest']['num_live_points'],
        const_efficiency_mode=config['fit']['multinest']['constant_efficiency_mode']
    )

    # Create corner plot
    minimizer.create_corner_plot()
else:
    raise KeyError('Invalid fit method')

# Minimizer Output dir
output_dir = minimizer.output_dir


################# Data Export ######################################
if config['export']['save_unbinned']:
    config['general']['min_bin_width'] = 1E-99
    model_generator = BackgroundModelGenerator()
    model_generator.from_config_dict(config)

    model_generator.likelihood.set_free_parameters(minimizer.best_fit_values)


if config['export']['save_cov_matrix']:
    minimizer.comp_covariance_matrix()

data_exporter = DataExporter(
    data=              model_generator.data,
    model=             model_generator.model,
    saa_object=        model_generator.saa_calc,
    echan_list=        config['general']['echan_list'],
    best_fit_values=   minimizer.best_fit_values,
    covariance_matrix= minimizer.cov_matrix
)

result_file_name = "fit_result_{}_{}_e{}.hdf5".format(config['general']['dates'],
                                                      config['general']['detector'],
                                                      config['general']['echan_list'])

data_exporter.save_data(
    file_path=os.path.join(output_dir, result_file_name),
    result_dir=output_dir,
    save_ppc=config['export']['save_ppc']
)


################## Plotting ########################################
print_progress('Create Plotter object...')

# Create Plotter object that creates the plots
plotter = Plotter(
    data=model_generator.data,
    model=model_generator.model,
    saa_object=model_generator.saa_calc,
    echan_list=config['general']['echan_list']
)
print_progress('Done')

# Create one plot for every echan and save it
for index, echan in enumerate(config['general']['echan_list']):
    print_progress('Create Plots for echan {} ...'.format(echan))

    residual_plot = plotter.display_model(
        index=              index,
        min_bin_width=      config['plot']['bin_width'],
        show_residuals=     config['plot']['show_residuals'],
        show_data=          config['plot']['show_data'],
        plot_sources=       config['plot']['plot_sources'],
        show_grb_trigger=   config['plot']['show_grb_trigger'],
        change_time=        config['plot']['change_time'],
        ppc=                config['plot']['ppc'],
        result_dir=         output_dir,
        xlim=               config['plot']['xlim'],
        ylim=               config['plot']['ylim'],
        legend_outside=     config['plot']['legend_outside']
    )

    if rank == 0:
        plot_file_name = 'residual_plot_{}_det_{}_echan_{}_bin_width_{}.pdf'.format(config['general']['dates'],
                                                                                    config['general']['detector'],
                                                                                    echan,
                                                                                    config['plot']['bin_width'])
        residual_plot.savefig(os.path.join(output_dir + plot_file_name), dpi=300)


# Save used config file to output directory
with open(os.path.join(output_dir + 'used_config.yml'), 'w') as file:
    documents = yaml.dump(config, file)

print_progress('Done')
# Print the duration of the script
print('Whole calculation took: {}'.format(datetime.now() - start))

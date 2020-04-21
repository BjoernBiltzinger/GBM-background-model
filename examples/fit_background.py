import matplotlib

matplotlib.use("Agg")

import os
import yaml
from datetime import datetime

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.io.export import DataExporter

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = datetime.now()

############## Argparse for parsing bash arguments ################
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    help="Path to a custom config file for the background fit",
    default="config_default.yml",
)
parser.add_argument(
    "-config_plot",
    "--config_plot_file",
    type=str,
    help="Path to a custom config file for result plotting",
    default="config_plot_default.yml",
)
parser.add_argument("-date", "--date", type=str, help="Date string")
parser.add_argument("-dets", "--detectors", type=str, help="Name detector")
parser.add_argument("-echans", "--echans", type=int, help="Echan number")
args = parser.parse_args()

# Load the config.yml
with open(args.config_file) as f:
    config = yaml.load(f)

############# Overwrite config with BASH arguments ################
if args.date is not None:
    config["general"]["dates"] = [args.date]
if args.detector is not None:
    config["general"]["detectors"] = args.detectors
if args.echan is not None:
    config["general"]["echans"] = [args.echans]

############## Generate the GBM-background-model ##################
model_generator = BackgroundModelGenerator()

model_generator.from_config_dict(config)

############### Instantiate Minimizer #############################
if config["fit"]["method"] == "multinest":
    minimizer = MultiNestFit(
        likelihood=model_generator.likelihood,
        parameters=model_generator.model.free_parameters,
    )

    # Fit with multinest and define the number of live points one wants to use
    minimizer.minimize_multinest(
        n_live_points=config["fit"]["multinest"]["num_live_points"],
        const_efficiency_mode=config["fit"]["multinest"]["constant_efficiency_mode"],
    )

    # Create corner plot
    minimizer.create_corner_plot()
else:
    raise KeyError("Invalid fit method")

# Minimizer Output dir
output_dir = minimizer.output_dir

################# Data Export ######################################
if config["export"]["save_unbinned"]:
    config["general"]["min_bin_width"] = 1e-99
    model_generator = BackgroundModelGenerator()
    model_generator.from_config_dict(config)

    model_generator.likelihood.set_free_parameters(minimizer.best_fit_values)

if config["export"]["save_cov_matrix"]:
    minimizer.comp_covariance_matrix()

data_exporter = DataExporter(
    data=model_generator.data,
    model=model_generator.model,
    saa_object=model_generator.saa_calc,
    echan_list=config["general"]["echans"],
    best_fit_values=minimizer.best_fit_values,
    covariance_matrix=minimizer.cov_matrix,
)

result_file_name = "fit_result_{}_{}_e{}.hdf5".format(
    config["general"]["dates"],
    config["general"]["detectors"],
    config["general"]["echans"],
)

data_exporter.save_data(
    file_path=os.path.join(output_dir, result_file_name),
    result_dir=output_dir,
    save_ppc=config["export"]["save_ppc"],
)

################## Plotting ########################################
if rank == 0:
    plot_generator = ResultPlotGenerator.from_result_file(
        config_file=args.config_plot_file,
        result_data_file=os.path.join(output_dir, result_file_name),
    )

    plot_generator.create_plots(output_dir=output_dir)

################## Save Config ########################################
if rank == 0:
    # Save used config file to output directory
    with open(os.path.join(output_dir + "used_config.yml"), "w") as file:
        documents = yaml.dump(config, file)

# Print the duration of the script
print("Whole calculation took: {}".format(datetime.now() - start))

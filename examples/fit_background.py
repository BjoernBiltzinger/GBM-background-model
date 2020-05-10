#!/usr/bin/env python3

##################################################################
# Generic script to fit the physical background model for GBM
#
# Optional command line arguments:
# -c --config_file
# -cplot --config_file_plot
# -dates --dates
# -dets --detectors
# -e --echans
# -trig --trigger
#
# Run with mpi:
# mpiexec -n <nr_cores> python fit_background.py \
#                       -c <config_path> \
#                       -cplot <config_plot_path> \
#                       -dates <date1> <date2> \
#                       -dets <det1> <det2> \
#                       -e <echan1> <echan2> \
#
# Example using the default config file:
# mpiexec -n 4 python fit_background.py -dates 190417 -dets n1 -e 2
##################################################################

from datetime import datetime

start = datetime.now()

import matplotlib

matplotlib.use("Agg")

import os
import yaml
import argparse

from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.model_generator import TrigdatBackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.io.export import DataExporter

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


############## Argparse for parsing bash arguments ################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    help="Path to a custom config file for the background fit",
    default="config_default.yml",
)

parser.add_argument(
    "-cplot",
    "--config_file_plot",
    type=str,
    help="Path to a custom config file for result plotting",
    default="config_plot_default.yml",
)

parser.add_argument("-dates", "--dates", type=str, nargs="+", help="Date string")
parser.add_argument("-dets", "--detectors", type=str, nargs="+", help="Name detector")
parser.add_argument("-e", "--echans", type=int, nargs="+", help="Echan number")
parser.add_argument("-trig", "--trigger", type=str, help="Name of trigger")

args = parser.parse_args()

# Load the config.yml
with open(args.config_file) as f:
    config = yaml.load(f)

############# Overwrite config with BASH arguments ################

if args.dates is not None:
    config["general"]["dates"] = args.dates

if args.detectors is not None:
    config["general"]["detectors"] = args.detectors

if args.echans is not None:
    config["general"]["echans"] = args.echans

if args.trigger is not None:
    config["general"]["trigger"] = args.trigger

############## Generate the GBM-background-model ##################

start_precalc = datetime.now()

if config["general"]["data_type"] in ["ctime", "cspec"]:

    model_generator_class = BackgroundModelGenerator

    model_generator = BackgroundModelGenerator()

    model_generator.from_config_dict(config)

elif config["general"]["data_type"] == "trigdat":

    model_generator_class = TrigdatBackgroundModelGenerator

    model_generator = TrigdatBackgroundModelGenerator()

    model_generator.from_config_dict(config)

    model_generator.likelihood.set_grb_mask(
        f"{model_generator.data.trigtime - 15}-{model_generator.data.trigtime + 100}"
    )

comm.barrier()

stop_precalc = datetime.now()

############### Instantiate Minimizer #############################

start_fit = datetime.now()

output_dir = os.path.join(
    get_path_of_external_data_dir(),
    "fits",
    "mn_out",
    config["general"].get("trigger", "-".join(config["general"]["dates"])),
    "det_" + "-".join(config["general"]["detectors"]),
    "echan_" + "-".join([str(e) for e in config["general"]["echans"]]),
    datetime.now().strftime("%m-%d_%H-%M") + "/",
)


if config["fit"]["method"] == "multinest":
    minimizer = MultiNestFit(
        likelihood=model_generator.likelihood,
        parameters=model_generator.model.free_parameters,
    )

    # Fit with multinest and define the number of live points one wants to use
    minimizer.minimize_multinest(
        n_live_points=config["fit"]["multinest"]["num_live_points"],
        const_efficiency_mode=config["fit"]["multinest"]["constant_efficiency_mode"],
        output_dir=output_dir,
    )

    # Create corner plot
    minimizer.create_corner_plot()

else:

    raise KeyError("Invalid fit method")

# Minimizer Output dir
output_dir = minimizer.output_dir

comm.barrier()

stop_fit = datetime.now()

################# Data Export ######################################

start_export = datetime.now()

if config["export"]["save_unbinned"] and config["general"]["min_bin_width"] > 1e-9:

    # Create copy of config dictionary
    config_export = config

    config_export["general"]["min_bin_width"] = 1e-9

    model_generator = model_generator_class()

    model_generator.from_config_dict(config_export)

    model_generator.likelihood.set_free_parameters(minimizer.best_fit_values)

    # Wait for all ranks
    comm.barrier()

data_exporter = DataExporter(
    data=model_generator.data,
    model=model_generator.model,
    saa_object=model_generator.saa_calc,
    echans=config["general"]["echans"],
    best_fit_values=minimizer.best_fit_values,
)

result_file_name = "fit_result_dates_{}_dets_{}_echans_{}.hdf5".format(
    "-".join(config["general"]["dates"]),
    "-".join(config["general"]["detectors"]),
    "-".join([str(e) for e in config["general"]["echans"]]),
)

data_exporter.save_data(
    file_path=os.path.join(output_dir, result_file_name),
    result_dir=output_dir,
    save_ppc=config["export"]["save_ppc"],
)

comm.barrier()

stop_export = datetime.now()

################## Plotting ########################################
start_plotting = datetime.now()

if rank == 0:
    print("Start plotting")

    plot_generator = ResultPlotGenerator.from_result_file(
        config_file=args.config_file_plot,
        result_data_file=os.path.join(output_dir, result_file_name),
    )

    # If we fit the background to trigdat data, then we will highlight the active
    # time, that we excluded from the fit, in the plot
    if config["general"]["data_type"] == "trigdat":
        plot_generator.add_occ_region(
            occ_name="Active Time",
            time_start=model_generator.data.trigtime - 15,
            time_stop=model_generator.data.trigtime + 150,
            time_format="MET",
            color="red",
            alpha=0.1,
        )

    plot_generator.create_plots(output_dir=output_dir)

comm.barrier()

stop_plotting = datetime.now()

################## Save Config ########################################
if rank == 0:

    # Save used config file to output directory
    with open(os.path.join(output_dir + "used_config.yml"), "w") as file:
        documents = yaml.dump(config, file)

if rank == 0:
    # Print the duration of the script
    print("The precalculations took: {}".format(stop_precalc - start_precalc))
    print("The fit took: {}".format(stop_fit - start_fit))
    print("The result export took: {}".format(stop_export - start_export))
    print("The plotting took: {}".format(stop_plotting - start_plotting))
    print("Whole calculation took: {}".format(datetime.now() - start))

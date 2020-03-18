############################
# This script creates a unbinned fit result file from a passed
# config file and a binned fit_result file.
###########################

import os
import yaml
from datetime import datetime

from gbmbkgpy.io.importer import FitImporter
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
parser.add_argument('-config', type=str, help='Path of the config file', required=True)
parser.add_argument('-result', type=str, help='Path of the result file', required=True)
args = parser.parse_args()

# Load the config.yml
with open(args.config) as f:
    config = yaml.load(f)

config['general']['min_bin_width'] = 1E-99


##### Instantiate background model from result and config file ####
fit_importer = FitImporter(
    config=config,
    fit_result_hdf5=args.result
)


################# Data Export ######################################
data_exporter = DataExporter(
    data=              fit_importer.data,
    model=             fit_importer.model,
    saa_object=        fit_importer.saa_calc,
    echan_list=        config['general']['echan_list'],
    best_fit_values=   fit_importer.best_fit_values,
    covariance_matrix= None
)

result_file_name = "fit_result_{}_{}_e{}_unbinned.hdf5".format(config['general']['dates'],
                                                               config['general']['detector'],
                                                               config['general']['echan_list'])
output_dir = os.path.dirname(args.result)

data_exporter.save_data(
    file_path=os.path.join(output_dir, result_file_name),
    result_dir=output_dir,
    save_ppc=config['export']['save_ppc']
)

print_progress('Done')
# Print the duration of the script
print('Whole calculation took: {}'.format(datetime.now() - start))

#############################################################################

################ Test download of data and preparation of data ##############

#############################################################################




################ Imports ###################

from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.io.downloading import download_files
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.file_utils import file_existing_and_readable
import os
import h5py
import numpy as np


# File to hdf5 file with expected values for tests
test_hdf5_path = 'test_expected.hdf5'





################# Setup1 for cspec ####################
date = ['160310', '190101']
detector = 'nb'
data_type = 'cspec'
echan_list = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]

################# Test data download #################

# Check where the data should be
data_path = get_path_of_external_data_dir()
file_path1 = os.path.join(data_path, data_type, date[0], 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, detector, date[0]))
file_path2 = os.path.join(data_path, data_type, date[1], 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, detector, date[1]))

# If it is already there, delete it
if file_existing_and_readable(file_path1):
    os.remove(file_path1)
if file_existing_and_readable(file_path2):
    os.remove(file_path2)

# Download the data
for d in date:
    download_files(data_type, detector, d)

# Check if the data is at the correct place
assert file_existing_and_readable(file_path1)
assert file_existing_and_readable(file_path2)


############## Check data preparation ################

data = Data(date, detector, data_type, echan_list)

# Import the expected values of the sum of all mean times and all counts
f = h5py.File(test_hdf5_path, 'r')
mean_time_c = f['/data_preparation/cspec_mean_time_sum'][()]
counts_c = f['/data_preparation/cspec_counts_sum'][()]
day_start_times_c = f['/data_preparation/cspec_day_start_times_sum'][()]
day_stop_times_c = f['/data_preparation/cspec_day_stop_times_sum'][()]
f.close()

# Check if the time bins and count arrays are the same
assert np.sum(data.mean_time) == mean_time_c
assert np.sum(data.counts) == counts_c
assert np.sum(data.day_start_times) == day_start_times_c
assert np.sum(data.day_stop_times) == day_stop_times_c


################# Setup1 for ctime ####################
date = ['140310', '180101']
detector = 'na'
data_type = 'ctime'
echan_list = [2, 3, 4]

################# Test data download #################

# Check where the data should be
data_path = get_path_of_external_data_dir()
file_path1 = os.path.join(data_path, data_type, date[0], 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, detector, date[0]))
file_path2 = os.path.join(data_path, data_type, date[1], 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, detector, date[1]))

# If it is already there, delete it
if file_existing_and_readable(file_path1):
    os.remove(file_path1)
if file_existing_and_readable(file_path2):
    os.remove(file_path2)

# Download the data
for d in date:
    download_files(data_type, detector, d)

# Check if the data is at the correct place
assert file_existing_and_readable(file_path1)
assert file_existing_and_readable(file_path2)


############## Check data preparation ################

data = Data(date, detector, data_type, echan_list)

# Import the expected values of the sum of all mean times and all counts
f = h5py.File(test_hdf5_path, 'r')
mean_time_c = f['/data_preparation/ctime_mean_time_sum'][()]
counts_c = f['/data_preparation/ctime_counts_sum'][()]
day_start_times_c = f['/data_preparation/ctime_day_start_times_sum'][()]
day_stop_times_c = f['/data_preparation/ctime_day_stop_times_sum'][()]
f.close()

# Check if the time bins and count arrays are the same
assert np.sum(data.mean_time) == mean_time_c
assert np.sum(data.counts) == counts_c
assert np.sum(data.day_start_times) == day_start_times_c
assert np.sum(data.day_stop_times) == day_stop_times_c


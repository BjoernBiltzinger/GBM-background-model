from gbmbkgpy.utils import data_cleaning
from gbmbkgpy.utils.data_cleaning import DataCleaner
from gbmbkgpy.io.downloading import download_files
import numpy as np
import os
from datetime import date, timedelta

using_mpi = False
file_dir = os.path.join(os.getenv('GBMDATA'), 'ml/data')

try:
    # see if we have mpi and/or are using parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        rank = 0
except:
    rank = 0


def daterange(start_date, end_date):
    return [start_date + timedelta(n) for n in range(int((end_date - start_date).days) + 1)]


# setup paramters
detector = 'nb'
data_type = 'ctime'
echan = 1
min_bin_width = 1

features = np.empty((0, 13))
counts = np.empty((0, 8))

date_start = date(2015, 1, 1)
date_stop = date(2015, 12, 31)

days = daterange(date_start, date_stop)

for day in days:
    date = day.strftime('%y%m%d')
    print('Start with {}'.format(date))

    # download files with rank=0; all other ranks have to wait!
    if rank == 0:
        try:
            download_files(data_type, detector, date)
            wait = True
            failed = False
        except Exception as e:
            print(e)
            wait = True
            failed = True
    else:
        wait = None
        failed = False

    if using_mpi:
        wait = comm.bcast(wait, root=0)
        failed = comm.bcast(failed, root=0)

    if failed:
        continue

    print('Downlaod complete')
    dc = DataCleaner(date, detector, data_type, min_bin_width=min_bin_width, training=True)

    if rank == 0:
        print('Stack features and counts')
        print('features: {}, counts: {}'.format(len(dc.rebinned_features), len(dc.rebinned_counts)))
        assert len(dc.rebinned_features) == len(dc.rebinned_counts)

        features = np.vstack((features, dc.rebinned_features))
        counts = np.vstack((counts, dc.rebinned_counts))

        wait = True
    else:
        wait = None
    if using_mpi:
        wait = comm.bcast(wait, root=0)
    del dc

if rank == 0:
    filename = os.path.join(file_dir, "cleaned_data_{}-{}_{}.npz".format(date_start.strftime('%y%m%d'), date_stop.strftime('%y%m%d'), detector))

    if os.path.isfile(filename):
        wait = True
        raise Exception("Error: output file already exists")
    np.savez_compressed(filename, counts=counts, features=features)
    wait = True
else:
    wait = None

if using_mpi:
    wait = comm.bcast(wait, root=0)

print('DONE')

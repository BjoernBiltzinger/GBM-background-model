from gbmbkgpy.utils import data_cleaning
from gbmbkgpy.utils.data_cleaning import DataCleaner
from gbmbkgpy.io.downloading import download_files, download_lat_spacecraft
from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file, get_path_of_external_data_dir
from gbmbkgpy.io.file_utils import file_existing_and_readable
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta, datetime
from gbmgeometry import GBMTime
import astropy.time as astro_time

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
save_intermediate = True
inter_files = []

features = np.empty((0, 14))
counts = np.empty((0, 8))
count_rates = np.empty((0, 8))

date_start = date(2018, 1, 1)
date_stop = date(2018, 1, 3)

days = daterange(date_start, date_stop)

grb_triggers = pd.read_json(os.path.join(file_dir, 'grb_triggers.json'))
grb_trigger_intervals = grb_triggers['trigger_interval']

for day in days:
    date = day.strftime('%y%m%d')
    print('Start with {}'.format(date))

    _year = '20%s' % date[:2]
    _month = date[2:-2]
    _day = date[-2:]

    astro_day = astro_time.Time("%s-%s-%s" % (_year, _month, _day))
    gbm_time = GBMTime(astro_day)
    mission_week = np.floor(gbm_time.mission_week.value)

    failed = False
    # download files with rank=0; all other ranks have to wait!
    if rank == 0:
        try:
            download_files(data_type, detector, date)
        except Exception as e:
            print(e)
            failed = True

        try:
            lat_filepath_1 = get_path_of_data_file('lat', 'lat_spacecraft_weekly_w%d_p202_v001.fits' % mission_week)
            if not file_existing_and_readable(lat_filepath_1):
                download_lat_spacecraft(mission_week)
        except Exception as e:
            print(e)
            failed = True

        try:
            lat_filepath_2 = get_path_of_data_file('lat', 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week + 1))
            if not file_existing_and_readable(lat_filepath_2):
                download_lat_spacecraft(mission_week + 1)
        except Exception as e:
            print(e)
            failed = True

        try:
            lat_filepath_3 = get_path_of_data_file('lat', 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week - 1))
            if not file_existing_and_readable(lat_filepath_3):
                download_lat_spacecraft(mission_week - 1)
        except Exception as e:
            print(e)
            failed = True

        wait = True
    else:
        wait = None
        failed = False

    if using_mpi:
        wait = comm.bcast(wait, root=0)
        failed = comm.bcast(failed, root=0)

    if failed:
        continue

    print('Downlaod complete')
    dc = DataCleaner(date, detector, data_type, min_bin_width=min_bin_width, training=True, trigger_intervals=grb_trigger_intervals)

    if rank == 0:
        print('features: {}, counts: {}'.format(len(dc.rebinned_features), len(dc.rebinned_counts)))
        assert len(dc.rebinned_features) == len(dc.rebinned_counts)

        if save_intermediate:
            inter_filename = os.path.join(file_dir, "days", "{}_inter_data_{}.npz".format(date, detector))
            inter_files.append(inter_filename)
            dc.save_data(inter_filename)
        else:
            print('Stack features and counts')
            features = np.vstack((features, dc.rebinned_features))
            counts = np.vstack((counts, dc.rebinned_counts))
            count_rates = np.vstack((count_rates, dc.rebinned_count_rates))

        wait = True
    else:
        wait = None
    if using_mpi:
        wait = comm.bcast(wait, root=0)
    del dc

if rank == 0:
    if save_intermediate:
        inter_file_listname = os.path.join(file_dir, "inter_days_{}__{}-{}.npz".format(detector, date_start.strftime('%y%m%d'),
                                                                              date_stop.strftime('%y%m%d'),))
        np.savez_compressed(inter_file_listname, inter_files=inter_files)

        print('Stack features and counts')
        for day_file in inter_files:
            with np.load(day_file, allow_pickle=True) as fhandle:
                counts = np.vstack((counts, fhandle['counts']))
                count_rates = np.vstack((count_rates, fhandle['count_rates']))
                features = np.vstack((features, fhandle['features']))

    combined_data_file = os.path.join(file_dir, "cleaned_data_{}-{}_d{}__{}.npz".format(date_start.strftime('%y%m%d'),
                                                                          date_stop.strftime('%y%m%d'),
                                                                          detector,
                                                                          datetime.now().strftime('%H_%M')))
    if os.path.isfile(combined_data_file):
        wait = True
        raise Exception("Error: output file already exists")
    np.savez_compressed(combined_data_file, counts=counts, count_rates=count_rates, features=features)
    wait = True
else:
    wait = None

if using_mpi:
    wait = comm.bcast(wait, root=0)

print('DONE')

import astropy.io.fits as fits

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

import astropy.time as astro_time

import numpy as np

import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmgeometry import GBMTime


try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True ###################33

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


class Data(object):

    def __init__(self, date, detector, data_type):
        """
        Initalize the ContinousData Class, which contains the information about the time bins 
        and counts of the data.
        """
        self._data_type = data_type
        self._det = detector
        self._day = date

        # Download data-file and poshist file if not existing:
        datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(self._data_type, self._det, self._day)
        datafile_path = os.path.join(get_path_of_external_data_dir(), self._data_type, self._day, datafile_name)

        poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', self._day)
        poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)

        # If MPI is used only one rank should download the data; the others wait
        if using_mpi:
            if rank==0:
                if not file_existing_and_readable(datafile_path):
                    download_data_file(self._day, self._data_type, self._det)

                if not file_existing_and_readable(poshistfile_path):
                    download_data_file(self._day, 'poshist')
            comm.Barrier()
        else:
            if not file_existing_and_readable(datafile_path):
                download_data_file(self._day, self._data_type, self._det)

            if not file_existing_and_readable(poshistfile_path):
                download_data_file(self._day, 'poshist')

        # Save poshistfile_path for later usage
        self._pos_hist = poshistfile_path


        # Open the datafile of the CTIME/CSPEC data and read in all needed quantities
        with fits.open(datafile_path) as f:
            self._counts = f['SPECTRUM'].data['COUNTS']
            self._bin_start = f['SPECTRUM'].data['TIME']
            self._bin_stop = f['SPECTRUM'].data['ENDTIME']

            self._exposure = f['SPECTRUM'].data['EXPOSURE']

            self._ebins_start = f['EBOUNDS'].data['E_MIN']
            self._ebins_stop = f['EBOUNDS'].data['E_MAX']
            
        self._ebins_size = self._ebins_stop - self._ebins_start

        # Sometimes there are corrupt time bins where the time bin start = time bin stop
        # So we have to delete these times bins
        i = 0
        while i < len(self._bin_start):
            if self._bin_start[i] == self._bin_stop[i]:
                self._bin_start = np.delete(self._bin_start, [i])
                self._bin_stop = np.delete(self._bin_stop, [i])
                self._counts = np.delete(self._counts, [i], axis=0)
                self._exposure=np.delete(self._exposure, [i])
                print('Deleted empty time bin', i)
            else:
                i+=1
                
        # Sometimes the poshist file does not cover the whole time coverd by the CTIME/CSPEC file.
        # So we have to delete these time bins 


        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f['GLAST POS HIST'].data['SCLK_UTC']
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]
        # check for all time bins if they are outside of this interval
        i=0
        counter=0
        while i<len(self._bin_start):
            if self._bin_start[i]<min_time_pos or self._bin_stop[i]>max_time_pos:
                self._bin_start = np.delete(self._bin_start, i)
                self._bin_stop = np.delete(self._bin_stop, i)
                self._counts = np.delete(self._counts, i, 0)
                self._exposure = np.delete(self._exposure, i)
                counter+=1
            else:
                i+=1

        # Print how many time bins were deleted
        if counter>0:
            print(str(counter) + ' time bins had to been deleted because they were outside of the time interval covered'
                                 'by the poshist file...')

        # Calculate and save some important quantities
        self._n_entries = len(self._bin_start)
        self._counts_combined = np.sum(self._counts, axis=1)
        self._counts_combined_mean = np.mean(self._counts_combined)
        self._counts_combined_rate = self._counts_combined / self.time_bin_length
        self._n_time_bins, self._n_channels = self._counts.shape

        # Calculate the MET time for the day
        day = self._day
        year = '20%s' % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        self._day_met = GBMTime(day_at).met
        
    @property
    def day(self):
        return self._day

    @property
    def data_type(self):
        return self._data_type

    @property
    def ebins(self):
        return np.vstack((self._ebins_start, self._ebins_stop)).T

    @property
    def detector_id(self):

        return self._det[-1]

    @property
    def n_channels(self):

        return self._n_channels

    @property
    def n_time_bins(self):

        return self._n_time_bins

    @property
    def rates(self):
        return self._counts / self._exposure.reshape((self._n_entries, 1))

    @property
    def counts(self):
        return self._counts

    @property
    def counts_combined(self):
        return self._counts_combined

    @property
    def counts_combined_rate(self):
        return self._counts_combined_rate

    @property
    def exposure(self):
        return self._exposure

    @property
    def time_bin_start(self):
        return self._bin_start

    @property
    def time_bin_stop(self):
        return self._bin_stop

    @property
    def time_bins(self):
        return np.vstack((self._bin_start, self._bin_stop)).T

    @property
    def time_bin_length(self):
        return self._bin_stop - self._bin_start

    @property
    def mean_time(self):
        return np.mean(self.time_bins, axis=1)


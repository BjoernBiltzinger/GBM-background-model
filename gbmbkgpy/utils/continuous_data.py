import astropy.io.fits as fits
import numpy as np
import collections
import matplotlib.pyplot as plt

from gbmgeometry import PositionInterpolator, gbm_detector_list

import scipy.interpolate as interpolate

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

import astropy.time as astro_time
import astropy.coordinates as coord
import math
import numpy as np
from scipy import interpolate
import os
from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_data_file, get_path_of_external_data_dir
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.plotting.step_plots import step_plot, slice_disjoint, disjoint_patch_plot
from gbmgeometry import GBMTime

import pymap3d as pm
import datetime
from gbmbkgpy.io.downloading import download_lat_spacecraft


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


class ContinuousData(object):

    def __init__(self, date, detector, data_type, rate_generator_DRM=None, use_SAA=True, clean_SAA=False):
        self._data_type = data_type
        self._det = detector
        self._day = date
        self._use_SAA = use_SAA
        self._clean_SAA = clean_SAA
        self._rate_generator_DRM = rate_generator_DRM
        #assert 'ctime' in self._data_type, 'currently only working for CTIME data'
        #assert 'n' in self._det, 'currently only working NAI detectors'
        ### Download data-file and poshist file if not existing:
        datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(self._data_type, self._det, self._day)
        datafile_path = os.path.join(get_path_of_external_data_dir(), self._data_type, self._day, datafile_name)

        poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', self._day)
        poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)
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

        ###

        self._pos_hist = poshistfile_path

        with fits.open(datafile_path) as f:
            self._counts = f['SPECTRUM'].data['COUNTS']
            self._bin_start = f['SPECTRUM'].data['TIME']
            self._bin_stop = f['SPECTRUM'].data['ENDTIME']

            self._exposure = f['SPECTRUM'].data['EXPOSURE']

            self._ebins_start = f['EBOUNDS'].data['E_MIN']
            self._ebins_stop = f['EBOUNDS'].data['E_MAX']
        self._ebins_size = self._ebins_stop - self._ebins_start
        ###
        #manual_start_time = 5.3646*10**8
        #manual_stop_time = 5.3654*10**8
        #self._bin_stop = self._bin_stop[self._bin_start>manual_start_time]
        #self._counts = self._counts[self._bin_start>manual_start_time]
        #self._exposure = self._exposure[self._bin_start>manual_start_time]
        #self._bin_start = self._bin_start[self._bin_start>manual_start_time]
        #self._bin_stop = self._bin_stop[self._bin_start<manual_stop_time]
        #self._counts = self._counts[self._bin_start<manual_stop_time]
        #self._exposure = self._exposure[self._bin_start<manual_stop_time]
        #self._bin_start = self._bin_start[self._bin_start<manual_stop_time]
        #print(self._bin_start)

        # Delete entries if in data file there are time bins with same start and end time
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
        # Delete time bins that are outside the interval covered by the poshist file
        # Get boundary for time interval covered by the poshist file
        with fits.open(poshistfile_path) as f:
            pos_times = f['GLAST POS HIST'].data['SCLK_UTC']
        min_time_pos = pos_times[0]
        max_time_pos = pos_times[-1]
        # check all time bins if they are outside of this interval
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
        if counter>0:
            print(str(counter) + ' time bins had to been deleted because they were outside of the time interval covered'
                                 'by the poshist file...')
        self._n_entries = len(self._bin_start)
        self._counts_combined = np.sum(self._counts, axis=1)
        self._counts_combined_mean = np.mean(self._counts_combined)
        self._counts_combined_rate = self._counts_combined / self.time_bin_length
        self._n_time_bins, self._n_channels = self._counts.shape
        # Start precomputation of arrays:
        self._setup_geometery()
        self._compute_saa_regions()
        self._min_duration = 7000
        if self._min_duration!=None:
            self._delete_short_timeintervals(self._min_duration)
        if rate_generator_DRM!=None:
            self._earth_rate_array()
            self._cgb_rate_array()
            self._response_sum()
        # Calculate the MET time for the day
        day = self._day
        year = '20%s' % day[:2]
        month = day[2:-2]
        dd = day[-2:]
        day_at = astro_time.Time("%s-%s-%s" % (year, month, dd))
        self._day_met = GBMTime(day_at).met
        #TEST
        self.west_angle_interpolator_define()
        self.north_angle_interpolator_define()
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

    def sun_earth_angle(self, met):
        return self._sun_earth_angle_interpolator(met)

    def get_quaternion(self, met):

        return self._position_interpolator.quaternion(met)
    def point_angle(self, met):
        return self._angle_point_interpolator(met)
    def cgb_background(self, time_bins):

        return np.ones_like(time_bins)

    def _setup_geometery(self):
        n_bins_to_calculate = 800.

        self._position_interpolator = PositionInterpolator(poshist=self._pos_hist)

        # ok we need to get the sun angle
        print(self._n_time_bins)
        n_skip = int(np.ceil(self._n_time_bins / n_bins_to_calculate))

        sun_angle = []
        sun_time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = [] #earth pos in icrs frame (skycoord)

        #additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos =[]

        #ps testing
        det_ra = [] #det ra in icrs frame
        det_dec = [] #det dec in icrs frame
        sun_earth_angle = []

        #testing secondary CR
        west =[]

        ra=[]
        dec=[]

        phi_west=[]
        theta_west=[]

        earth_angle=[]

        ra_p = 83.633
        dec_p = 22.015
        angle_point=[]
        sun_pos_icrs=[]
        pointing_earth_frame_phi=[]
        pointing_earth_frame_theta=[]
        pointing_vector = []
        if using_mpi:
            #if using mpi split the times at which the geometry is calculated to all ranks
            list_times_to_calculate = self.mean_time[::n_skip]
            self._times_per_rank = float(len(list_times_to_calculate))/float(size)
            self._times_lower_bound_index = int(np.floor(rank*self._times_per_rank))
            self._times_upper_bound_index = int(np.floor((rank+1)*self._times_per_rank))
            print(rank)
            if rank==0:
                with progress_bar(len(list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]),
                                  title='Calculating sun and earth position. This shows the progress of rank 0. All other should be about the same.') as p:
                    for mean_time in list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]:
                        quaternion_step = self._position_interpolator.quaternion(mean_time)
                        sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                        det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                           sc_pos=sc_pos_step,
                                                           time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                        sun_angle.append(det.sun_angle.value)
                        sun_time.append(mean_time)
                        az, zen = det.earth_az_zen_sat
                        earth_az.append(az)
                        earth_zen.append(zen)
                        earth_position.append(det.earth_position)

                        quaternion.append(quaternion_step)
                        sc_pos.append(sc_pos_step)

                        #test
                        sun_earth_angle.append(det.earth_angle.value)
                        #west.append(det.angle_pointing_west())
                        ra_s, dec_s = det.det_ra_dec_icrs
                        ra.append(ra_s)
                        dec.append(dec_s)

                        #phi_west_s, theta_west_s = det.pointing_west_coord_frame()
                        #phi_west.append(phi_west_s)
                        #theta_west.append(theta_west_s)
                        pointing_earth_frame_phi.append(det.pointing_earth_frame_phi)
                        pointing_earth_frame_theta.append(det.pointing_earth_frame_theta)
                        earth_angle.append(det.earth_angle.value)
                        pointing_vector.append(det.pointing_vector_earth_frame)
                        p.increase()
            else:

                for mean_time in list_times_to_calculate[self._times_lower_bound_index:self._times_upper_bound_index]:
                    quaternion_step = self._position_interpolator.quaternion(mean_time)
                    sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(self._position_interpolator.utc(mean_time)))
                    
                    sun_angle.append(det.sun_angle.value)
                    sun_time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)
                    
                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)
                    
                    #test                                                                                                                                                                                                                                                  
                    sun_earth_angle.append(det.earth_angle.value)
                    #west.append(det.angle_pointing_west())
                    ra_s, dec_s = det.det_ra_dec_icrs
                    ra.append(ra_s)
                    dec.append(dec_s)

                    #phi_west_s, theta_west_s = det.pointing_west_coord_frame()
                    #phi_west.append(phi_west_s)
                    #theta_west.append(theta_west_s)
                    pointing_earth_frame_phi.append(det.pointing_earth_frame_phi)
                    pointing_earth_frame_theta.append(det.pointing_earth_frame_theta)
                    earth_angle.append(det.earth_angle.value)
                    pointing_vector.append(det.pointing_vector_earth_frame)
            #get the last data point with the last rank                                                                                                                                                                                                                    
            if rank == size - 1:

                mean_time = self.mean_time[-2]
                quaternion_step = self._position_interpolator.quaternion(mean_time)
                sc_pos_step = self._position_interpolator.sc_pos(mean_time)

                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)


                #test
                sun_earth_angle.append(det.earth_angle.value)
                #west.append(det.angle_pointing_west())
                ra_s, dec_s = det.det_ra_dec_icrs
                ra.append(ra_s)
                dec.append(dec_s)

                #phi_west_s, theta_west_s = det.pointing_west_coord_frame()
                #phi_west.append(phi_west_s)
                #theta_west.append(theta_west_s)
                pointing_earth_frame_phi.append(det.pointing_earth_frame_phi)
                pointing_earth_frame_theta.append(det.pointing_earth_frame_theta)

                earth_angle.append(det.earth_angle.value)

                pointing_vector.append(det.pointing_vector_earth_frame)

            #make the list numpy arrays
            sun_angle = np.array(sun_angle)
            sun_time = np.array(sun_time)
            earth_az = np.array(earth_az)
            earth_zen = np.array(earth_zen)
            earth_position = np.array(earth_position)

            quaternion = np.array(quaternion)
            sc_pos = np.array(sc_pos)

            #test
            sun_earth_angle = np.array(sun_earth_angle)
            west = np.array(west)
            ra = np.array(ra)
            dec = np.array(dec)

            phi_west = np.array(phi_west)
            theta_west = np.array(theta_west)

            pointing_vector_x = np.array(pointing_vector)[:,0]
            pointing_vector_y = np.array(pointing_vector)[:,1]
            pointing_vector_z = np.array(pointing_vector)[:,2]
            
            #gather all results in rank=0
            sun_angle_gather = comm.gather(sun_angle, root=0)
            sun_time_gather = comm.gather(sun_time, root=0)
            earth_az_gather = comm.gather(earth_az, root=0)
            earth_zen_gather = comm.gather(earth_zen, root=0)
            earth_position_gather = comm.gather(earth_position, root=0)


            quaternion_gather = comm.gather(quaternion, root=0)
            sc_pos_gather = comm.gather(sc_pos, root=0)

            #test
            sun_earth_angle_gather = comm.gather(sun_earth_angle, root=0)
            west_gather = comm.gather(west, root=0)
            ra_gather = comm.gather(ra, root=0)
            dec_gather = comm.gather(dec, root=0)

            phi_west_gather = comm.gather(phi_west, root=0)
            theta_west_gather = comm.gather(theta_west, root=0)
            pointing_earth_frame_phi_gather = comm.gather(pointing_earth_frame_phi, root=0)
            pointing_earth_frame_theta_gather = comm.gather(pointing_earth_frame_theta, root=0)
            earth_angle_gather = comm.gather(earth_angle, root=0)

            pointing_vector_x_gather = comm.gather(pointing_vector_x, root=0)
            pointing_vector_y_gather = comm.gather(pointing_vector_y, root=0)
            pointing_vector_z_gather = comm.gather(pointing_vector_z, root=0)
            #make one list out of this
            if rank == 0:
                sun_angle_gather = np.concatenate(sun_angle_gather)
                sun_time_gather = np.concatenate(sun_time_gather)
                earth_az_gather = np.concatenate(earth_az_gather)
                earth_zen_gather = np.concatenate(earth_zen_gather)
                earth_position_gather = np.concatenate(earth_position_gather)

                quaternion_gather=np.concatenate(quaternion_gather)
                sc_pos_gather = np.concatenate(sc_pos_gather)

                #test
                sun_earth_angle_gather = np.concatenate(sun_earth_angle_gather)
                west_gather = np.concatenate(west_gather)
                ra_gather = np.concatenate(ra_gather)
                dec_gather = np.concatenate(dec_gather)

                phi_west_gather = np.concatenate(phi_west_gather)
                theta_west_gather = np.concatenate(theta_west_gather)
                pointing_earth_frame_phi_gather = np.concatenate(pointing_earth_frame_phi_gather)
                pointing_earth_frame_theta_gather = np.concatenate(pointing_earth_frame_theta_gather)
                earth_angle_gather = np.concatenate(earth_angle_gather)

                pointing_vector_x_gather =  np.concatenate(pointing_vector_x_gather)
                pointing_vector_y_gather =  np.concatenate(pointing_vector_y_gather)
                pointing_vector_z_gather =  np.concatenate(pointing_vector_z_gather)
            #broadcast the final arrays again to all ranks
            sun_angle = comm.bcast(sun_angle_gather, root=0)
            sun_time = comm.bcast(sun_time_gather, root=0)
            earth_az = comm.bcast(earth_az_gather, root=0)
            earth_zen = comm.bcast(earth_zen_gather, root=0)
            earth_position = comm.bcast(earth_position_gather, root=0)

            quaternion = comm.bcast(quaternion_gather, root=0)
            sc_pos = comm.bcast(sc_pos_gather, root=0)

            #test
            sun_earth_angle = comm.bcast(sun_earth_angle_gather, root=0)
            west = comm.bcast(west_gather, root=0)
            det_ra = comm.bcast(ra_gather, root=0)
            det_dec = comm.bcast(dec_gather, root=0)

            phi_west = comm.bcast(phi_west_gather, root=0)
            theta_west = comm.bcast(theta_west_gather, root=0)
            pointing_earth_frame_phi = comm.bcast(pointing_earth_frame_phi_gather, root=0)
            pointing_earth_frame_theta = comm.bcast(pointing_earth_frame_theta_gather, root=0)
            earth_angle = comm.bcast(earth_angle_gather, root=0)

            pointing_vector_x = comm.bcast(pointing_vector_x_gather, root=0)
            pointing_vector_y = comm.bcast(pointing_vector_y_gather, root=0)
            pointing_vector_z = comm.bcast(pointing_vector_z_gather, root=0)
        else:
            # go through a subset of times and calculate the sun angle with GBM geometry

            ###SINGLECORE CALC###
            with progress_bar(n_bins_to_calculate, title='Calculating sun and earth position') as p:

                for mean_time in self.mean_time[::n_skip]:
                    quaternion_step = self._position_interpolator.quaternion(mean_time)
                    sc_pos_step = self._position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(self._position_interpolator.utc(mean_time)))

                    sun_angle.append(det.sun_angle.value)
                    sun_time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)

                    #test
                    ra, dec = det.det_ra_dec_icrs
                    det_ra.append(ra)
                    det_dec.append(dec)
                    sun_earth_angle.append(det.sun_earth_angle.value)
                    #phi_west_s, theta_west_s = det.pointing_west_coord_frame()
                    #phi_west.append(phi_west_s)
                    #theta_west.append(theta_west_s)
                    #west.append(det.angle_pointing_west())
                    sun_pos_icrs.append(det.sun_position_icrs)
                    angle_point.append(det.det_angle_to_ra_dec_icrs(ra_p,dec_p).value)
                    pointing_earth_frame_phi.append(det.pointing_earth_frame_phi)
                    pointing_earth_frame_theta.append(det.pointing_earth_frame_theta)
                    p.increase()

            # get the last data point

            mean_time = self.mean_time[-2]

            quaternion_step = self._position_interpolator.quaternion(mean_time)
            sc_pos_step = self._position_interpolator.sc_pos(mean_time)
            det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                               sc_pos=sc_pos_step,
                                               time=astro_time.Time(self._position_interpolator.utc(mean_time)))

            sun_angle.append(det.sun_angle.value)
            sun_time.append(mean_time)
            az, zen = det.earth_az_zen_sat
            earth_az.append(az)
            earth_zen.append(zen)
            earth_position.append(det.earth_position)

            quaternion.append(quaternion_step)
            sc_pos.append(sc_pos_step)

            # test
            ra, dec = det.det_ra_dec_icrs
            det_ra.append(ra)
            det_dec.append(dec)
            sun_earth_angle.append(det.sun_earth_angle.value)
            #phi_west_s, theta_west_s = det.pointing_west_coord_frame()
            #phi_west.append(phi_west_s)
            #theta_west.append(theta_west_s)
            #west.append(det.angle_pointing_west())
            angle_point.append(det.det_angle_to_ra_dec_icrs(ra_p,dec_p).value)
            sun_pos_icrs.append(det.sun_position_icrs)

            pointing_earth_frame_phi.append(det.pointing_earth_frame_phi)
            pointing_earth_frame_theta.append(det.pointing_earth_frame_theta)


        self._sun_angle = sun_angle
        self._sun_time = sun_time
        self._earth_az = earth_az
        self._earth_zen = earth_zen
        self._earth_position = earth_position

        self._quaternion = quaternion
        self._sc_pos = sc_pos

        #test
        self._det_ra = np.array(det_ra)
        self._det_dec = np.array(det_dec)
        self._sun_earth_angle = np.array(sun_earth_angle)
        self._sun_earth_angle_interpolator = interpolate.interp1d(self._sun_time, self._sun_earth_angle)
        #self._west = west
        self._angle_point = angle_point

        #self._phi_west = np.array(phi_west)
        #self._theta_west = np.array(theta_west)

        self._earth_angle = np.array(earth_angle)
        # interpolate it

        self._sun_angle_interpolator = interpolate.interp1d(self._sun_time, self._sun_angle)


        # interpolate test
        #self._west_interpolator = interpolate.interp1d(self._sun_time, self._west)
        self._det_ra_interpolator = interpolate.interp1d(self._sun_time, self._det_ra)
        self._det_dec_interpolator = interpolate.interp1d(self._sun_time, self._det_dec)

        #self._phi_west_interpolator = interpolate.interp1d(self._sun_time, self._phi_west)
        #self._theta_west_interpolator = interpolate.interp1d(self._sun_time, self._theta_west)
        if using_mpi:
            self._earth_angle_interpolator = interpolate.interp1d(self._sun_time, self._earth_angle)
        if not using_mpi:
            self._sun_pos_icrs = np.array(sun_pos_icrs)
            self._angle_point_interpolator = interpolate.interp1d(self._sun_time, self._angle_point)
        self._pointing_earth_frame_phi=np.array(pointing_earth_frame_phi)
        self._pointing_earth_frame_theta=np.array(pointing_earth_frame_theta)
        self._pointing_earth_frame_phi_interpolator = interpolate.interp1d(self._sun_time, self._pointing_earth_frame_phi)
        self._pointing_earth_frame_theta_interpolator = interpolate.interp1d(self._sun_time, self._pointing_earth_frame_theta)


        ####################################################################################################################
        if using_mpi:        
            lon_geo_inter, lat_geo_inter, rad_geo_inter = self.build_lat_spacecraft_lon(int('20{}'.format(self.day[0:2])), int(self.day[2:4]), int(self.day[4:]), self.time_bins[0,0], self.time_bins[-1,1])
            lon = lon_geo_inter(self._sun_time)
            lat = lat_geo_inter(self._sun_time)
            rad = rad_geo_inter(self._sun_time)

            dates = np.array([])
            for met in self._sun_time:
                if met <= 252460801.000:
                    utc_tt_diff = 65.184
                elif met <= 362793602.000:
                    utc_tt_diff = 66.184
                elif met <= 457401603.000:
                    utc_tt_diff = 67.184
                elif met <= 504921604.000:
                    utc_tt_diff = 68.184
                else:
                    utc_tt_diff = 69.184
                mjdutc = ((met - utc_tt_diff) / 86400.0) + 51910 + 0.0007428703703
                a=astro_time.Time(mjdutc, scale='utc', format='mjd').datetime.strftime('%y-%m-%d %H-%M-%S')
                dates = np.append(dates, datetime.datetime.strptime(a,'%y-%m-%d %H-%M-%S'))

        
            pointing_mag = pm.eci2ecef(np.transpose((10000*pointing_vector_x,10000*pointing_vector_y,10000*pointing_vector_z)),dates)
            start_x_ecef,start_y_ecef,start_z_ecef = pm.geodetic2ecef(lat,lon,rad)
            pointing_mag_x, pointing_mag_y, pointing_mag_z = pm.ecef2ned(pointing_mag[:,0]+start_x_ecef, pointing_mag[:,1]+start_y_ecef, pointing_mag[:,2]+start_z_ecef, lat, lon, rad)
            mag_pointing_vec = np.array([pointing_mag_x,pointing_mag_y,pointing_mag_z]).T
            for i in range(len(mag_pointing_vec)):
                mag_pointing_vec[i] = mag_pointing_vec[i]/(np.sqrt(mag_pointing_vec[i,0]**2+mag_pointing_vec[i,1]**2+mag_pointing_vec[i,2]**2))
            angles_down_ned = np.array([])
            for i in range(len(mag_pointing_vec)):
                angles_down_ned = np.append(angles_down_ned, np.arccos(np.dot(mag_pointing_vec[i],np.array([0,0,1]))))
            angles_east_ned = np.array([])
            for i in range(len(mag_pointing_vec)):
                angles_east_ned = np.append(angles_east_ned, np.arccos(np.dot(mag_pointing_vec[i],np.array([0,1,0]))))
                
            self._ned_east_interpolator = interpolate.interp1d(self._sun_time, angles_east_ned)
            self._ned_down_interpolator = interpolate.interp1d(self._sun_time, angles_down_ned)

        
        del sun_angle, sun_time, earth_az, earth_zen
    
    def _compute_saa_regions(self):

        # find where the counts are zero

        min_saa_bin_width = 1
        bins_to_add = 8

        ############################################################################################################
        #self._counts_combined[1]=0
        #self._counts_combined[2]=0
        ############################################################################################################

        self._zero_idx = self._counts_combined == 0.
        idx = (self._zero_idx).nonzero()[0]
        slice_idx = np.array(slice_disjoint(idx))

        # Only the slices which are longer than 8 time bins are used as saa (only for ctime data)
        if self._data_type=='cspec':
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > 0)]
        else:
            slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > min_saa_bin_width)]
            #slice_idx = slice_idx[np.where(slice_idx[:, 1] - slice_idx[:, 0] > 0)]
        
        # Add bins_to_add to bin_mask to exclude the bins with corrupt data:
        # Check first that the start and stop stop of the mask is not the beginning or end of the day
        slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] =\
            slice_idx[:, 0][np.where(slice_idx[:, 0] >= 8)] - bins_to_add

        slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] =\
            slice_idx[:, 1][np.where(slice_idx[:, 1] <= self._n_time_bins - 1 - bins_to_add)] + bins_to_add



        # now find the times of the exits

        if slice_idx[-1 , 1] == self._n_time_bins - 1:

            # the last exit is just the end of the array
            self._saa_exit_idx = slice_idx[:-1, 1]

        else:

            self._saa_exit_idx = slice_idx[:, 1]

        self._saa_exit_mean_times = self.mean_time[self._saa_exit_idx]
        self._saa_exit_bin_start = self._bin_start[self._saa_exit_idx]
        self._saa_exit_bin_stop = self._bin_stop[self._saa_exit_idx]

        # make a saa mask from the slices:
        self._saa_mask = np.ones_like(self._counts_combined, bool)

        for i in range(len(slice_idx)):
            self._saa_mask[slice_idx[i, 0]:slice_idx[i, 1] + 1] = False
            self._zero_idx[slice_idx[i, 0]:slice_idx[i, 1] + 1] = True
        

        # deleting 5000s after every saa exit => ignore saa's
        if not self._use_SAA:
            time_after_saa = 5000###5000
            if self._bin_stop[slice_idx[0, 0]] - time_after_saa > self._bin_start[0]:
                print('a')
                #self._saa_mask[0:slice_idx[0, 0] + 1] = False
                #self._zero_idx[0:slice_idx[0, 0] + 1] = True
            else:
                j = 0
                while time_after_saa > self._bin_start[j] - self._bin_start[0]:
                    j += 1
                self._saa_mask[0:j] = False
                self._zero_idx[0:j] = True
            print(slice_idx)
            print(self._saa_mask)

            for i in range(len(slice_idx) - 1):
                if self._bin_stop[slice_idx[i + 1, 0]] - self._bin_start[slice_idx[i, 1]] < time_after_saa:
                    self._saa_mask[slice_idx[i, 1]:slice_idx[i + 1, 0]] = False
                    self._zero_idx[slice_idx[i, 1]:slice_idx[i + 1, 0]] = True
                else:
                    j = 0
                    while self._bin_start[slice_idx[i, 1]] + time_after_saa > self._bin_start[slice_idx[i, 1] + j]:
                        j += 1
                    self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
                    self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True

            if self._bin_stop[slice_idx[-1, 1]] + time_after_saa > self._bin_stop[-1]:
                self._saa_mask[slice_idx[-1, 1]:len(self._counts_combined) + 1] = False
                self._zero_idx[slice_idx[-1, 1]:len(self._counts_combined) + 1] = True
            else:
                j = 0
                while self._bin_start[slice_idx[-1, 1]] + time_after_saa > self._bin_start[slice_idx[-1, 1] + j]:
                    j += 1
                self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
                self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True
            
        
        # deleting 300s after very sharp SAA's
        if self._clean_SAA:
            # if self._bin_stop[slice_idx[0, 0]] - 300 > self._bin_start[0]:
            #     self._saa_mask[0:slice_idx[0, 0] + 1] = False
            #     self._zero_idx[0:slice_idx[0, 0] + 1] = True
            # else:
            #     j = 0
            #     while 300 > self._bin_start[j] - self._bin_start[0]:
            #         j += 1
            #     self._saa_mask[0:j] = False
            #     self._zero_idx[0:j] = True

            for i in range(len(slice_idx) - 1):

                saa_amplitude = np.mean(self._counts_combined[slice_idx[i, 1]:slice_idx[i, 1] + 10] /
                                        self.time_bin_length[slice_idx[i, 1]:slice_idx[i, 1] + 10])

                if saa_amplitude > 1.5 * self._counts_combined_mean:

                    if self._bin_stop[slice_idx[i + 1, 0]] - self._bin_start[slice_idx[i, 1]] < 300:
                        self._saa_mask[slice_idx[i, 1]:slice_idx[i + 1, 0]] = False
                        self._zero_idx[slice_idx[i, 1]:slice_idx[i + 1, 0]] = True
                    else:
                        j = 0
                        while self._bin_start[slice_idx[i, 1]] + 300 > self._bin_start[slice_idx[i, 1] + j]:
                            j += 1
                        self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
                        self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True

            # if self._bin_stop[slice_idx[-1, 1]] + 300 > self._bin_stop[-1]:
            #     self._saa_mask[slice_idx[-1, 1]:len(self._counts_combined) + 1] = False
            #     self._zero_idx[slice_idx[-1, 1]:len(self._counts_combined) + 1] = True
            # else:
            #     j = 0
            #     while self._bin_start[slice_idx[-1, 1]] + 300 > self._bin_start[slice_idx[-1, 1] + j]:
            #         j += 1
            #     self._saa_mask[slice_idx[i, 1]:slice_idx[i, 1] + j] = False
            #     self._zero_idx[slice_idx[i, 1]:slice_idx[i, 1] + j] = True

        self._saa_slices = slice_idx

    def _delete_short_timeintervals(self, min_duration):
        # get index intervals of SAA mask
        index_start = [0]
        index_stop = []

        for i in range(len(self._saa_mask)-1):
            if self._saa_mask[i]==False and self._saa_mask[i+1]==True:
                index_stop.append(i-1)
            if self._saa_mask[i]==True and self._saa_mask[i+1]==False:
                index_start.append(i)

        if len(index_start)>len(index_stop):
            index_stop.append(-1)

        assert len(index_start)==len(index_stop)

        #set saa_mask=False between end and next start if time is <min_duration
        for i in range(len(index_stop)-1):
            if self.time_bin_stop[index_start[i+1]]-self.time_bin_start[index_stop[i]]<min_duration:
                self._saa_mask[index_stop[i]-5:index_start[i+1]+5]=np.ones_like(self._saa_mask[index_stop[i]-5:index_start[i+1]+5])==2
                self._zero_idx[index_stop[i]-5:index_start[i+1]+5]=np.ones_like(self._zero_idx[index_stop[i]-5:index_start[i+1]+5])==1
                
    @property
    def quaternion(self):

        return self._quaternion

    @property
    def sc_pos(self):

        return self._sc_pos

    @property
    def pointing(self):

        return self._pointing

    @property
    def interpolation_time(self):

        return self._sun_time

    def sun_angle(self, met):

        return self._sun_angle_interpolator(met)


    def earth_angle(self, met):

        return self._earth_angle_interpolator(met)

    @property
    def earth_position(self):

        return self._earth_position

    @property
    def saa_mask(self):

        return self._saa_mask


    @property
    def saa_mean_times(self):

        return self._saa_exit_mean_times


    def saa_initial_values(self, echan):

        start_value_array = []

        # Add mean of first 10 time bins for leftover decay from day before
        start_value_array.append(np.mean(self._counts[0:11, echan] / self.time_bin_length[0:11]))

        for i, exit_idx in enumerate(self._saa_exit_idx):
            start_value_array.append(
                np.mean(self._counts[exit_idx:exit_idx+10, echan] / self.time_bin_length[exit_idx:exit_idx+10]))

        return np.array(start_value_array)


    def saa_initial_decay_constants(self, echan):

        amplitudes_t0 = self.saa_initial_values(echan)

        amplitudes_t1 = []

        # Add mean of first 10 time bins for leftover decay from day before
        amplitudes_t1.append(np.mean(self._counts[100:121, echan] / self.time_bin_length[100:121]))

        for i, exit_idx in enumerate(self._saa_exit_idx):
            amplitudes_t1.append(
                np.mean(self._counts[exit_idx + 100: exit_idx + 120, echan] /
                        self.time_bin_length[exit_idx + 100: exit_idx + 120]))

        initial_decay_constants = (np.log(amplitudes_t1) - np.log(amplitudes_t0)) /\
                                  (self.mean_time[exit_idx + 100] - self.mean_time[exit_idx])

        # Replace positive values
        initial_decay_constants[np.where(initial_decay_constants > 0)] = \
            initial_decay_constants[np.where(initial_decay_constants > 0)] * -1.

        return np.array(initial_decay_constants)


    def plot_light_curve(self,channel=0, ax=None):

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()



        step_plot(self.time_bins, self.rates[:,0], ax, fill=False, fill_min=0,color='green')


        # disjoint_patch_plot(ax, self._bin_start, self._bin_stop, ~self._zero_idx, )


        ax.set_ylabel('rate (cnts/s)')
        ax.set_xlabel('MET (s)')

        return fig


    def plot_angle(self):

        fig, ax = plt.subplots()

        ax.plot(self.mean_time[:-1],self.sun_angle(self.mean_time[:-1]))
        ax.plot(self.mean_time[:-1], self.earth_angle(self.mean_time[:-1]))

        ax.set_xlabel('MET')
        ax.set_ylabel('Angle (deg)')

        return fig


    def plot_eff_angle(self):

        fig, ax = plt.subplots()


        x_grid = np.linspace(-180,180,200)

        for i in range(self._n_channels):

            ax.plot(x_grid,self.effective_angle(i,x_grid))


        ax.set_xlabel('angle (deg)')
        ax.set_ylabel('effective area')

    @property
    def use_SAA(self):

        return self._use_SAA

    def _earth_rate_array(self):
        """
        Calculate the earth_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the earth_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the earth
        albedo.
        :return:
        """
        points = self._rate_generator_DRM.points
        earth_rates = self._rate_generator_DRM.earth_rate
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
                print(upper_index)
            else:
                upper_index = self._times_upper_bound_index
            
            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            earth_pos = np.array(earth_pos_inter_times) #
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []

            det_earth_angle = []#
            #point_earth_angle_all_inter = [] #                                                                                                                                                                                                                                
            #point_base_rate_all_inter = [] # 
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])

                det = np.array([1,0,0])#
                det_earth_angle.append(np.arccos(np.dot(pos, det))*180/np.pi)#
                #point_earth_angle = [] #                                                                                                                                                                                                                                      
                #point_base_rate = [] #
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    #point_earth_angle.append(angle_earth)#
                    if angle_earth < opening_angle_earth:
                        B=0
                        earth_rate += earth_rates[i]*np.exp(B*angle_earth)#TODO RING EFFECT
                        #point_base_rate.append(earth_rates[i])# 
                    #else:#                                                                                                                                                                                                                                                    
                        #point_base_rate.append(np.zeros_like(earth_rates[i]))#
                array_earth_rate.append(earth_rate)
                #point_base_rate_all_inter.append(point_base_rate)#
                #point_earth_angle_all_inter.append(point_earth_angle)#
                
            array_earth_rate = np.array(array_earth_rate)
            det_earth_angle = np.array(det_earth_angle)#
            #point_earth_angle = np.array(point_earth_angle_all_inter)#                                                                                                                                                                                                  
            #point_base_rate = np.array(point_base_rate_all_inter)#
            #del point_earth_angle_all_inter, point_base_rate_all_inter, earth_pos_inter_times
            array_earth_rate_g = comm.gather(array_earth_rate, root=0)
            det_earth_angle_g = comm.gather(det_earth_angle, root=0)#
            earth_pos_g = comm.gather(earth_pos, root=0)#
            #point_earth_angle_g = comm.gather(point_earth_angle, root=0)
            #point_base_rate_g = comm.gather(point_base_rate, root=0)
            if rank == 0:
                array_earth_rate_g = np.concatenate(array_earth_rate_g)
                det_earth_angle_g = np.concatenate(det_earth_angle_g)
                earth_pos_g = np.concatenate(earth_pos_g)
                #point_earth_angle_g = np.concatenate(point_earth_angle_g)
                #point_base_rate_g = np.concatenate(point_base_rate_g)
            array_earth_rate = comm.bcast(array_earth_rate_g, root=0)
            det_earth_angle = comm.bcast(det_earth_angle_g, root=0)
            earth_pos = comm.bcast(earth_pos_g,root=0)#
            #point_earth_angle = comm.bcast(point_earth_angle_g, root=0)
            #point_base_rate = comm.bcast(point_base_rate_g, root=0)
            #del array_earth_rate_g, point_earth_angle_g, point_base_rate_g
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_earth_rate = []
            #point_earth_angle_all_inter = [] #
            #point_base_rate_all_inter = [] #
            for pos in self._earth_pos_inter_times:
                earth_rate = np.zeros_like(earth_rates[0])
                #point_earth_angle = [] #
                #point_base_rate = [] #
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    #point_earth_angle.append(angle_earth)#
                    if angle_earth < opening_angle_earth:
                        #point_base_rate.append(earth_rates[i])#
                        earth_rate += earth_rates[i]
                    #else:#
                        #point_base_rate.append(np.zeros_like(earth_rates[i]))#
                array_earth_rate.append(earth_rate)
                #point_base_rate_all_inter.append(point_base_rate)#
                #point_earth_angle_all_inter.append(point_earth_angle)#
            #point_base_rate = point_base_rate_all_inter
            #point_earth_angle = point_earth_angle_all_inter
        array_earth_rate = np.array(array_earth_rate).T
        #point_earth_angle = np.array(point_earth_angle)#
        #if rank==0:
            #print('Earth pos')
            #print(earth_pos[:10])
            #print('earth_rate')
            #print(array_earth_rate[4][:10])
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #surf = ax.scatter(earth_pos[:,0],earth_pos[:,1],earth_pos[:,2], s=0.4, c=array_earth_rate[4], cmap='plasma')
            #ax.scatter(1,0,0,s=10,c='red')
            #fig.colorbar(surf)
            #fig.savefig('testing_B_{}.pdf'.format(B))
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #surf = ax.scatter(points[:,0],points[:,1],points[:,2], s=0.4, c=earth_rates[:,4], cmap='plasma')
            #fig.colorbar(surf)
            #fig.savefig('testing_2.pdf')
        #point_base_rate = np.array(point_base_rate)#
        #self._point_earth_angle_interpolator = interpolate.interp1d(self._sun_time, point_earth_angle, axis=0)#
        #self._point_base_rate_interpolator = interpolate.interp1d(self._sun_time, point_base_rate, axis=0)#
        self._earth_rate_interpolator = interpolate.interp1d(self._sun_time, array_earth_rate)
        #del point_base_rate, point_earth_angle, array_earth_rate
    def _cgb_rate_array(self):
        """
        Calculate the cgb_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the cgb_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are not occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the cgb
        spectrum.
        :return:
        """
        points = self._rate_generator_DRM.points
        cgb_rates = self._rate_generator_DRM.cgb_rate
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
            else:
                upper_index = self._times_upper_bound_index

            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_rate = []
            for pos in self._earth_pos_inter_times:
                cgb_rate = np.zeros_like(cgb_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_rate += cgb_rates[i]
                array_cgb_rate.append(cgb_rate)
            array_cgb_rate = np.array(array_cgb_rate)

            array_cgb_rate_g = comm.gather(array_cgb_rate, root=0)
            if rank == 0:
                array_cgb_rate_g = np.concatenate(array_cgb_rate_g)
            array_cgb_rate = comm.bcast(array_cgb_rate_g, root=0)
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_rate = []
            for pos in self._earth_pos_inter_times:
                cgb_rate = np.zeros_like(cgb_rates[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_rate += cgb_rates[i]
                array_cgb_rate.append(cgb_rate)
        self._array_cgb_rate = np.array(array_cgb_rate).T
        self._cgb_rate_interpolator = interpolate.interp1d(self._sun_time, self._array_cgb_rate)


    def cgb_rate_array(self, met):
        """
        Interpolation function for the CGB continuum rate in a certain Ebin
        :param met: times at which to interpolate
        :return: array with the CGB rates expected over whole day in a certain Ebin
        """

        return self._cgb_rate_interpolator(met)

    def earth_rate_array(self, met):
        """
        Interpolation function for the Earth continuum rate in a certain Ebin
        :param met: times at which to interpolate
        :return: array with the Earth rates expected over whole day in a certain Ebin
        """

        return self._earth_rate_interpolator(met)

    @property
    def cgb_rate_interpolation_time(self):
        return self._array_cgb_rate

    @property
    def earth_rate_interpolation_time(self):
        return self._array_earth_rate

    @property
    def earth_az_interpolation_time(self):
        return self._earth_az

    @property
    def earth_zen_interpolation_time(self):
        return self._earth_zen

    @property
    def earth_pos_interpolation_time(self):
        return self._earth_pos_inter_times

    @property
    def saa_slices(self):
        return self._saa_slices

    @property
    def rate_generator_DRM(self):
        return self._rate_generator_DRM

    #test
    @property
    def det_ra_icrs(self):
        return self._det_ra

    @property
    def det_dec_icrs(self):
        return self._det_dec

    @property
    def times_lower_bound_index(self):
        """
        :return: the lower bound index of the part of the interpolation list covered by this rank
        """
        return self._times_lower_bound_index

    @property
    def times_upper_bound_index(self):
        """
        :return: the upper bound index of the part of the interpolation list covered by this rank
        """
        return self._times_upper_bound_index

    
    #test
    def west(self, met):
        return self._west_interpolator(met)

    def ra(self, met):
        return self._det_ra_interpolator(met)
    def dec(self, met):
        return self._det_dec_interpolator(met)

    def phi_west(self, met):
        return self._phi_west_interpolator(met)

    def theta_west(self, met):
        return self._theta_west_interpolator(met)
    def earth_angle(self, met):
        return self._earth_angle_interpolator(met)


    def point_earth_angle(self, met):
        return self._point_earth_angle_interpolator(met)

    def point_base_rate(self, met):
        return self._point_base_rate_interpolator(met)

    @property
    def sun_pos_icrs(self):
        return self._sun_pos_icrs

    def point_earth_phi(self,met):
        return self._pointing_earth_frame_phi_interpolator(met)

    def point_earth_theta(self,met):
        return self._pointing_earth_frame_theta_interpolator(met) 

    def west_angle_interpolator_define(self):
        sc_pos = self.sc_pos
        sun_time = self.interpolation_time
        sc_pos_norm = []
        for pos in sc_pos:
            sc_pos_norm.append(pos/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
        sc_pos_norm=np.array(sc_pos_norm)
        west_vector_bin = np.array([sc_pos_norm[:,1],-sc_pos_norm[:,0],np.zeros(len(sc_pos_norm[:,0]))]).T
        west_vector_bin_norm = []
        for pos in west_vector_bin:
            west_vector_bin_norm.append(pos/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
        west_vector_bin_norm=np.array(west_vector_bin_norm)

        point_phi=self._pointing_earth_frame_phi
        point_theta=self._pointing_earth_frame_theta
        
        pointing_vector = np.array([np.cos(point_phi)*np.cos(point_theta), np.sin(point_phi)*np.cos(point_theta), np.sin(point_theta)]).T

        self._phi_west = np.arccos(west_vector_bin_norm[:,0]*pointing_vector[:,0]+west_vector_bin_norm[:,1]*pointing_vector[:,1]+west_vector_bin_norm[:,2]*pointing_vector[:,2])*180/np.pi

        self._west_angle_interpolator = interpolate.interp1d(self._sun_time, self._phi_west)


    def north_angle_interpolator_define(self):
        sc_pos = self.sc_pos
        sun_time = self.interpolation_time
        sc_pos_norm = []
        for pos in sc_pos:
            sc_pos_norm.append(pos/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
        sc_pos_norm=np.array(sc_pos_norm)
        north_vector_bin = np.array([np.zeros(len(sc_pos_norm[:,0])),np.zeros(len(sc_pos_norm[:,0])),sc_pos_norm[:,2]]).T
        north_vector_bin_norm = []
        for pos in north_vector_bin:
            north_vector_bin_norm.append(pos/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
        north_vector_bin_norm=np.array(north_vector_bin_norm)

        point_phi=self._pointing_earth_frame_phi
        point_theta=self._pointing_earth_frame_theta

        pointing_vector = np.array([np.cos(point_phi)*np.cos(point_theta), np.sin(point_phi)*np.cos(point_theta), np.sin(point_theta)]).T

        self._phi_north = np.arccos(north_vector_bin_norm[:,0]*pointing_vector[:,0]+north_vector_bin_norm[:,1]*pointing_vector[:,1]+north_vector_bin_norm[:,2]*pointing_vector[:,2])*180/np.pi

        self._north_angle_interpolator = interpolate.interp1d(self._sun_time, self._phi_north)

        
    def west_angle(self, met):
        return self._west_angle_interpolator(met)

    def north_angle(self, met):
        return self._north_angle_interpolator(met)

    def ned_east(self, met):
        return self._ned_east_interpolator(met)

    def ned_down(self, met):
        return self._ned_down_interpolator(met)


    def build_lat_spacecraft_lon(self, year, month, day, min_met, max_met):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file

        day = astro_time.Time("%s-%s-%s" %(year, month, day))

        gbm_time = GBMTime(day)

        mission_week = np.floor(gbm_time.mission_week.value)


        filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % mission_week
        filepath = get_path_of_data_file('lat', filename)


        if not file_existing_and_readable(filepath):

            download_lat_spacecraft(mission_week)


        # lets check that this file has the right info

        week_before = False
        week_after = False

        with fits.open(filepath) as f:

            if (f['PRIMARY'].header['TSTART'] >= min_met):

                # we need to get week before

                week_before = True

                before_filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week - 1)
                before_filepath = get_path_of_data_file('lat', before_filename)

                if not file_existing_and_readable(before_filepath):
                    download_lat_spacecraft(mission_week - 1)


            if (f['PRIMARY'].header['TSTOP'] <= max_met):

                # we need to get week after

                week_after = True

                after_filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % (mission_week + 1)
                after_filepath = get_path_of_data_file('lat', after_filename)

                if not file_existing_and_readable( after_filepath):
                    download_lat_spacecraft(mission_week + 1)


            # first lets get the primary file

            lat_time = np.mean( np.vstack( (f['SC_DATA'].data['START'],f['SC_DATA'].data['STOP'])),axis=0)
            lat_geo = f['SC_DATA'].data['LAT_GEO']
            lon_geo = f['SC_DATA'].data['LON_GEO']
            rad_geo = f['SC_DATA'].data['RAD_GEO']


        # if we need to append anything to make up for the
        # dates not being included in the files
        # do it here... thanks Fermi!
        if week_before:

            with fits.open(before_filepath) as f:

                lat_time_before = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                lat_geo_before = f['SC_DATA'].data['LAT_GEO']
                lon_geo_before = f['SC_DATA'].data['LON_GEO']
                rad_geo_before = f['SC_DATA'].data['RAD_GEO']


            lat_geo = np.append(lat_geo_before, mc_b)
            lon_geo = np.append(lon_geo_before, mc_l)
            rad_geo = np.append(rad_geo_before, rad_geo)
            lat_time = np.append(lat_time_before, lat_time)

        if week_after:

            with fits.open(after_filepath) as f:
                lat_time_after = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                lat_geo_after = f['SC_DATA'].data['LAT_GEO']
                lon_geo_after = f['SC_DATA'].data['LON_GEO']
                rad_geo_after = f['SC_DATA'].data['RAD_GEO']

            lon_geo = np.append(lon_geo, lon_geo_after)
            lat_geo = np.append(lat_geo, lat_geo_after)
            rad_geo = np.append(rad_geo, rad_geo_after)
            lat_time = np.append(lat_time, lat_time_after)

        """
        # save them
        #TODO: do we need use the mean here?
        self._mc_l = mc_l
        self._mc_b = mc_b
        self._mc_time = lat_time
        # interpolate them
        self._mc_b_interp = interpolate.interp1d(self._mc_time, self._mc_b)
        self._mc_l_interp = interpolate.interp1d(self._mc_time, self._mc_l)
        """
        #remove the self-variables for memory saving
        lon_geo_interp = interpolate.interp1d(lat_time, lon_geo)
        lat_geo_interp = interpolate.interp1d(lat_time, lat_geo)
        rad_geo_interp = interpolate.interp1d(lat_time, rad_geo)
        return lon_geo_interp, lat_geo_interp, rad_geo_interp

    @property
    def ebins_size(self):
        return self._ebins_size
        
    def _response_sum(self):
        """
        Calculate the cgb_rate_array for all interpolation times for which the geometry was calculated. This supports
        MPI to reduce the calculation time.
        To calculate the cgb_rate_array the responses created on a grid in rate_gernerator_DRM are used. All points
        that are not occulted by the earth are added, assuming a spectrum specified in rate_generator_DRM for the cgb
        spectrum.
        :return:
        """
        points = self._rate_generator_DRM.points
        responses = self._rate_generator_DRM.responses
        sr_points = 4 * np.pi / len(points)
        # get the earth direction at the interpolation times; zen angle from -90 to 90
        earth_pos_inter_times = []
        if using_mpi:
            # last rank has to cover one more index. Caused by the calculation of the Geometry for the last time
            # bin of the day
            if rank == size - 1:
                upper_index = self._times_upper_bound_index + 1
            else:
                upper_index = self._times_upper_bound_index

            for i in range(self._times_lower_bound_index, upper_index):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []
            for pos in self._earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
                array_cgb_response_sum.append(cgb_response_time)
                array_earth_response_sum.append(earth_response_time)
            array_cgb_response_sum = np.array(array_cgb_response_sum)
            array_earth_response_sum = np.array(array_earth_response_sum)
            array_cgb_response_sum_g = comm.gather(array_cgb_response_sum, root=0)
            array_earth_response_sum_g = comm.gather(array_earth_response_sum, root=0) 
            if rank == 0:
                array_cgb_response_sum_g = np.concatenate(array_cgb_response_sum_g)
                array_earth_response_sum_g = np.concatenate(array_earth_response_sum_g)
            array_cgb_response_sum = comm.bcast(array_cgb_response_sum_g, root=0)
            array_earth_response_sum = comm.bcast(array_earth_response_sum_g, root=0)
        else:
            for i in range(0, len(self._earth_zen)):
                earth_pos_inter_times.append(
                    np.array([np.cos(self._earth_zen[i] * (np.pi / 180)) * np.cos(self._earth_az[i] * (np.pi / 180)),
                              np.cos(self._earth_zen[i] * (np.pi / 180)) * np.sin(self._earth_az[i] * (np.pi / 180)),
                              np.sin(self._earth_zen[i] * (np.pi / 180))]))
            self._earth_pos_inter_times = np.array(earth_pos_inter_times)
            # define the opening angle of the earth in degree
            opening_angle_earth = 67
            array_cgb_response_sum = []
            array_earth_response_sum = []
            for pos in self._earth_pos_inter_times:
                cgb_response_time = np.zeros_like(responses[0])
                earth_response_time = np.zeros_like(responses[0])
                for i, pos_point in enumerate(points):
                    angle_earth = np.arccos(np.dot(pos, pos_point)) * (180 / np.pi)
                    if angle_earth > opening_angle_earth:
                        cgb_response_time += responses[i]
                    else:
                        earth_response_time += responses[i]
                array_cgb_response_sum.append(cgb_response_time)                                                                                                                                                 
                array_earth_response_sum.append(earth_response_time)
                
        self._array_cgb_response_sum = np.array(array_cgb_response_sum)*sr_points
        self._array_earth_response_sum = np.array(array_earth_response_sum)*sr_points

    @property
    def response_array_earth(self):
        return self._array_earth_response_sum


    @property
    def response_array_cgb(self):
        return self._array_cgb_response_sum

    @property
    def Ebin_source(self):
        return self._rate_generator_DRM.Ebin_in_edge

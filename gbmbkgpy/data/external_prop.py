import astropy.io.fits as fits
import astropy.time as astro_time
import astropy.units as u
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from gbmgeometry import GBMTime
import os
from gbmbkgpy.io.downloading import download_flares, download_lat_spacecraft
from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.package_data import get_path_of_data_file
from gbmbkgpy.modeling.point_source import PointSrc
import csv

class ExternalProps(object):

    def __init__(self, day_list):
        """
        Build the external properties for a given day
        :param day: YYMMDD
        """

        assert len(day_list[0]) == 6, 'Day must be in format YYMMDD'

        # Global list which weeks where already added to the lat data (to prevent double entries later)
        self._weeks = np.array([])

        for i, date in enumerate(day_list):
            mc_l, mc_b, lat_time, lat_geo, lon_geo = self._one_day_build_lat_spacecraft(date)
            if i==0:
                self._mc_l = mc_l
                self._mc_b = mc_b
                self._lat_time = lat_time
                self._lat_geo = lat_geo
                self._lon_geo = lon_geo
            else:
                self._mc_l = np.append(self._mc_l, mc_l)
                self._mc_b = np.append(self._mc_b, mc_b)
                self._lat_time = np.append(self._lat_time, lat_time)
                self._lat_geo = np.append(self._lat_geo, lat_geo)
                self._lon_geo = np.append(self._lon_geo, lon_geo)

        self._mc_l_interp = interpolate.interp1d(self._lat_time, self._mc_l)

    def build_point_sources(self, rsp, geom, echan_list):
        """
        Build all PS saved in the txt file
        :param rsp: response_precalculation
        :param geom: geometry_precalculation
        :return:
        """
        self._rsp = rsp
        self._geom = geom
        self._build_point_sources()

    def build_some_source(self, rsp, geom, source_list, echan_list):
        """
        Build the PS form the text file with a certain name
        :param rsp: response_precalculation
        :param geom: geometry_precalculation
        :param source_list: which sources to buld
        :return:
        """
        self._rsp = rsp
        self._geom = geom
        self._build_some_source(source_list, echan_list)

    def mc_l(self, met):
        """
        Get MC L for a given MET
        :param met: 
        :return: 
        """

        return self._mc_l_interp(met)


    def mc_b(self, met):
        """
        Get MC B for a given MET
        :param met: 
        :return: 
        """

        return self._mc_b_interp(met)

    def lon_geo(self, met):

        return self._lon_geo_interp(met)

    def lat_geo(self, met):

        return self._lat_geo_interp(met)

    @property
    def point_sources(self):
        return self._point_sources_dic

    def _one_day_build_lat_spacecraft(self, date):
        """This function reads a LAT-spacecraft file and stores the data in arrays of the form: lat_time, mc_b, mc_l.\n
        Input:\n
        readfile.lat_spacecraft ( week = WWW )\n
        Output:\n
        0 = time\n
        1 = mcilwain parameter B\n
        2 = mcilwain parameter L"""

        # read the file
        year = '20%s' % date[:2]
        month = date[2:-2]
        dd = date[-2:]

        day = astro_time.Time("%s-%s-%s" %(year, month, dd))

        min_met = GBMTime(day).met

        max_met = GBMTime(day + u.day).met

        gbm_time = GBMTime(day)

        mission_week = np.floor(gbm_time.mission_week.value)


        filename = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % mission_week
        filepath = get_path_of_data_file('lat', filename)

        if not file_existing_and_readable(filepath):

            download_lat_spacecraft(mission_week)

        # Init all arrays as empty arrays

        lat_time = np.array([])
        mc_l = np.array([]) 
        mc_b = np.array([]) 
        lon_geo = np.array([]) 
        lat_geo = np.array([]) 

        
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
            if mission_week not in self._weeks:
                lat_time = np.mean( np.vstack( (f['SC_DATA'].data['START'],f['SC_DATA'].data['STOP'])), axis=0)
                mc_l = f['SC_DATA'].data['L_MCILWAIN']
                mc_b = f['SC_DATA'].data['B_MCILWAIN']
                lon_geo = f['SC_DATA'].data['LON_GEO']
                lat_geo = f['SC_DATA'].data['LAT_GEO']
                
                self._weeks = np.append(self._weeks, mission_week)
        # if we need to append anything to make up for the
        # dates not being included in the files
        # do it here... thanks Fermi!
        if week_before and (mission_week-1) not in self._weeks:

            with fits.open(before_filepath) as f:

                lat_time_before = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                mc_l_before = f['SC_DATA'].data['L_MCILWAIN']
                mc_b_before = f['SC_DATA'].data['B_MCILWAIN']
                lon_geo_before = f['SC_DATA'].data['LON_GEO']
                lat_geo_before = f['SC_DATA'].data['LAT_GEO']

            mc_b = np.append(mc_b_before, mc_b)
            mc_l = np.append(mc_l_before, mc_l)
            lon_geo = np.append(lon_geo_before, lon_geo)
            lat_geo = np.append(lat_geo_before, lat_geo)
            lat_time = np.append(lat_time_before, lat_time)

            self._weeks = np.append(self._weeks, mission_week-1)

        if week_after and (mission_week+1) not in self._weeks:

            with fits.open(after_filepath) as f:
                lat_time_after = np.mean(np.vstack((f['SC_DATA'].data['START'], f['SC_DATA'].data['STOP'])), axis=0)
                mc_l_after = f['SC_DATA'].data['L_MCILWAIN']
                mc_b_after = f['SC_DATA'].data['B_MCILWAIN']
                lon_geo_after = f['SC_DATA'].data['LON_GEO']
                lat_geo_after = f['SC_DATA'].data['LAT_GEO']
            mc_b = np.append(mc_b, mc_b_after)
            mc_l = np.append(mc_l, mc_l_after)
            lon_geo = np.append(lon_geo, lon_geo_after)
            lat_geo = np.append(lat_geo, lat_geo_after)
            lat_time = np.append(lat_time, lat_time_after)
            self._weeks = np.append(self._weeks, mission_week + 1)

        return mc_l, mc_b, lat_time, lat_geo, lon_geo

    def _build_point_sources(self, echan_list, free_spectrum=[]):
        """This function reads the point_sources.dat file and returns the sources in the form: names, coordinates[ra][dec]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = source_names
        1 = coordinates\n"""
        file_path = get_path_of_data_file('background_point_sources/', 'point_sources.dat')

        self._ps_df = pd.read_table(file_path, names=['name', 'ra', 'dec'])

        # instantiate dic of point source objects
        self._point_sources_dic = {}

        ###Single core calc###
        for i, row in enumerate(self._ps_df.itertuples()):
            if len(free_spectrum) > 0 and free_spectrum[i]:
                self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._rsp, self._geom,
                                                           echan_list)
            else:
                self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._rsp, self._geom,
                                                           echan_list, index=2.114)

    def _build_some_source(self, source_list, echan_list, free_spectrum=[]):
        """
        This function builds the PS specified in source_list
        :param source_list: list of PS to build
        :return:
        """
        file_path = get_path_of_data_file('background_point_sources/', 'point_sources.dat')

        self._ps_df = pd.read_table(file_path, names=['name', 'ra', 'dec'])

        # instantiate dic of point source objects
        self._point_sources_dic = {}

        ###Single core calc###
        for row in self._ps_df.itertuples():
            for i, element in enumerate(source_list):
                if row[1]==element:
                    if len(free_spectrum) > 0 and free_spectrum[i]:
                        self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._rsp, self._geom,
                                                                   echan_list)
                    else:
                        self._point_sources_dic[row[1]] = PointSrc(row[1], row[2], row[3], self._rsp, self._geom,
                                                                   echan_list, index=2.114)

    def lat_acd(self, time_bins, use_side):

        data_path = '/home/bbiltzing/output_acd.csv'

        with open(data_path, 'r') as f:
            lines = csv.reader(f)
            lines_final = []
            for line in lines:
                lines_final.append(','.join(line))


        timestamps = []
        dets = []
        counts = []
        delta_times = []
        sides = []
        
        for line in lines_final[1:]:
            timestamp, det, count, delta_time, side = line.split(',')
            timestamps.append(float(timestamp))
            dets.append(int(det))
            counts.append(int(count))
            delta_times.append(float(delta_time))
            sides.append(side)
        timestamps = np.array(timestamps)
        dets = np.array(dets)
        counts = np.array(counts)
        delta_times = np.array(delta_times)


        counts_A=[]
        delta_times_A=[]
        counts_B=[]
        delta_times_B=[]
        counts_C=[]
        delta_times_C=[]
        counts_D=[]
        delta_times_D=[]
        for i,time in enumerate(timestamps[::108]):
            counts_A.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='A']))
            delta_times_A.append(np.array([delta_times[j] for j in range(i*108, (i+1)*108) if sides[j]=='A'])) 
            counts_B.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='B']))
            delta_times_B.append(np.array([delta_times[j] for j in range(i*108, (i+1)*108) if sides[j]=='B']))
            counts_C.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='C']))
            delta_times_C.append(np.array([delta_times[j] for j in range(i*108, (i+1)*108) if sides[j]=='C']))
            counts_D.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='D']))
            delta_times_D.append(np.array([delta_times[j] for j in range(i*108, (i+1)*108) if sides[j]=='D']))
        counts_A=np.array(counts_A)
        counts_B=np.array(counts_B)
        counts_C=np.array(counts_C)
        counts_D=np.array(counts_D)
        delta_times_A=np.array(delta_times_A)
        delta_times_B=np.array(delta_times_B)
        delta_times_C=np.array(delta_times_C)
        delta_times_D=np.array(delta_times_D)
        
        rate_A = np.sum(counts_A, axis=1)/np.sum(delta_times_A, axis=1)
        rate_B = np.sum(counts_B, axis=1)/np.sum(delta_times_B, axis=1)
        rate_C = np.sum(counts_C, axis=1)/np.sum(delta_times_C, axis=1)
        rate_D = np.sum(counts_D, axis=1)/np.sum(delta_times_D, axis=1)

        counts_A_all = np.sum(counts_A, axis=1)
        time_delta_A_all = np.sum(delta_times_A, axis=1)
        counts_B_all = np.sum(counts_B, axis=1)
        time_delta_B_all = np.sum(delta_times_B, axis=1)
        counts_C_all = np.sum(counts_C, axis=1)
        time_delta_C_all = np.sum(delta_times_C, axis=1)
        counts_D_all = np.sum(counts_D, axis=1)
        time_delta_D_all = np.sum(delta_times_D, axis=1)

        sum_timestamps = 50
        
        binned_timestamps = []
        rate_A_binned = []
        rate_B_binned = []
        rate_C_binned = []
        rate_D_binned = []
        for i in range(len(timestamps[::108])/sum_timestamps):
            binned_timestamps.append((timestamps[::108][(i+1)*sum_timestamps-1]+timestamps[::108][i*sum_timestamps])/2)
            rate_A_binned.append(np.sum(counts_A_all[i*sum_timestamps:(i+1)*sum_timestamps])/np.sum(time_delta_A_all[i*sum_timestamps:(i+1)*sum_timestamps]))
            rate_B_binned.append(np.sum(counts_B_all[i*sum_timestamps:(i+1)*sum_timestamps])/np.sum(time_delta_B_all[i*sum_timestamps:(i+1)*sum_timestamps]))
            rate_C_binned.append(np.sum(counts_C_all[i*sum_timestamps:(i+1)*sum_timestamps])/np.sum(time_delta_C_all[i*sum_timestamps:(i+1)*sum_timestamps]))
            rate_D_binned.append(np.sum(counts_D_all[i*sum_timestamps:(i+1)*sum_timestamps])/np.sum(time_delta_D_all[i*sum_timestamps:(i+1)*sum_timestamps]))
        rate_A_binned = np.array(rate_A_binned)
        rate_B_binned = np.array(rate_B_binned)
        rate_C_binned = np.array(rate_C_binned)
        rate_D_binned = np.array(rate_D_binned)
        binned_timestamps = np.array(binned_timestamps)
        
        if use_side=='A':
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #ax = fig.add_osubplot(111)
            #ax.scatter(timestamps[::108], rate_A, s=0.3)
            #fig.savefig('ext_prop_lat_acd.pdf')
            
            rate_A_binned[rate_A_binned>210] = rate_A_binned[rate_A_binned>210] - (rate_A_binned[rate_A_binned>210]).min()
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_A_binned)
        elif use_side=='B':
            #rate_B_binned[rate_B_binned>100] = rate_B_binned[rate_B_binned>100] - (rate_B_binned[rate_B_binned>100]).min()
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_B_binned)
        elif use_side=='C':
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(binned_timestamps, rate_C_binned)
            fig.savefig('ext_prop_lat_acd.pdf')

            rate_C_binned[rate_C_binned>210] = rate_C_binned[rate_C_binned>210] - (rate_C_binned[rate_C_binned>210]).min()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(binned_timestamps, rate_C_binned)
            fig.savefig('ext_prop_lat_acd_2.pdf')


            
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_C_binned)
        elif use_side=='D':
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_D_binned)

        return interpolate_acd(time_bins)

    def acd_saa_mask(self, time_bins):

        data_path = '/home/bbiltzing/output_acd.csv'

        with open(data_path, 'r') as f:
            lines = csv.reader(f)
            lines_final = []
            for line in lines:
                lines_final.append(','.join(line))


        timestamps = []
        dets = []
        counts = []
        delta_times = []
        sides = []

        for line in lines_final[1:]:
            timestamp, det, count, delta_time, side = line.split(',')
            timestamps.append(float(timestamp))
            dets.append(int(det))
            counts.append(int(count))
            delta_times.append(float(delta_time))
            sides.append(side)
        timestamps = np.array(timestamps)
        dets = np.array(dets)
        counts = np.array(counts)
        delta_times = np.array(delta_times)


        counts_A=[]
        counts_B=[]
        counts_C=[]
        counts_D=[]

        for i,time in enumerate(timestamps[::108]):
            counts_A.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='A']))
            counts_B.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='B']))
            counts_C.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='C']))
            counts_D.append(np.array([counts[j] for j in range(i*108, (i+1)*108) if sides[j]=='D']))
        counts_A_all = np.sum(counts_A, axis=1)
        counts_B_all = np.sum(counts_B, axis=1)
        counts_C_all = np.sum(counts_C, axis=1)
        counts_D_all = np.sum(counts_D, axis=1)

        timestamp_zero = np.sum(np.array([counts_A_all,counts_B_all,counts_C_all,counts_D_all]),axis=0) > 1
        
        #set last timestamp before and after SAA also to False
        i=0
        while i<len(timestamp_zero)-1:
            if timestamp_zero[i+1] == False:
                timestamp_zero[i] = False
            elif timestamp_zero[i] == False and timestamp_zero[i+1]==True:
                timestamp_zero[i+1] = False
                # jump next list value
                i+=1
            i+=1
        
        timestamp_index = []
        i=0
        for time_bin in time_bins:
            while i<len(timestamps):
                if time_bin[0]>timestamps[::108][i]:
                    timestamp_index.append(i)
                    break
                else:
                    i+=1

        mask = timestamp_zero[timestamp_index]

        return mask
                

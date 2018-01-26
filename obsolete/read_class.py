#!/usr/bin python2.7

import os

import numpy as np
import pyfits

from obsolete.work_module_refactor import ExternalProps
from obsolete.work_module_refactor import calculate
from obsolete.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()

class readfile:
    """This class contains all functions for reading the files needed for the GBM background model"""
    
    def cspec(self, detector, day, seconds = 0):
        """This function reads a cspec file and stores the data in arrays of the form: echan[emin, emax], total_counts, echan_counts[echan], exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n
        Input:\n
        readfile.ctime ( detector, day = YYMMDD, seconds = SSS )\n
        0 = energy channel interval\n
        1 = total number of counts\n
        2 = number of counts per energy channel\n
        3 = total count rate\n
        4 = count rate per energy channel\n
        5 = bin time interval[start, end]\n
        6 = good time interval[start, end]\n
        7 = time of exposure\n"""
        
        #read the file. Check if one wants to read a specific trigger file or a daily file
        if seconds == 0:
            filename = 'glg_cspec_' + str(detector) + '_' + str(day) + '_v00.pha'
        else:
            filename = 'glg_cspec_' + str(detector) + '_bn' + str(day) + str(seconds) + '_v00.pha'
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        fits_path = os.path.join(os.path.dirname(__dir__), 'cspec')
        filepath = os.path.join(fits_path, str(filename))
        fits = pyfits.open(filepath)
        energy = fits[1].data
        spectrum = fits[2].data
        goodtime = fits[3].data
        fits.close()
        
        #extract the data
        emin = energy['E_MIN'] #lower limit of the energy channels
        emax = energy['E_MAX'] #upper limit of the energy channels
        echan = np.zeros((len(emin),2), float) #combine the energy limits of the energy channels in one matrix
        echan[:,0] = emin
        echan[:,1] = emax
        counts = spectrum['COUNTS']
        total_counts = np.sum(counts, axis=1) #total number of counts for each time intervall
        echan_counts = np.vstack(([counts[:,0].T], [counts[:,1].T], [counts[:,2].T], [counts[:,3].T], [counts[:,4].T], [counts[:,5].T], [counts[:,6].T], [counts[:,7].T], [counts[:,8].T], [counts[:,9].T], [counts[:,10].T], [counts[:,11].T], [counts[:,12].T], [counts[:,13].T], [counts[:,14].T], [counts[:,15].T], [counts[:,16].T], [counts[:,17].T], [counts[:,18].T], [counts[:,19].T], [counts[:,20].T], [counts[:,21].T], [counts[:,22].T], [counts[:,23].T], [counts[:,24].T], [counts[:,25].T], [counts[:,26].T], [counts[:,27].T], [counts[:,28].T], [counts[:,29].T], [counts[:,30].T], [counts[:,31].T], [counts[:,32].T], [counts[:,33].T], [counts[:,34].T], [counts[:,35].T], [counts[:,36].T], [counts[:,37].T], [counts[:,38].T], [counts[:,39].T], [counts[:,40].T], [counts[:,41].T], [counts[:,42].T], [counts[:,43].T], [counts[:,44].T], [counts[:,45].T], [counts[:,46].T], [counts[:,47].T], [counts[:,48].T], [counts[:,49].T], [counts[:,50].T], [counts[:,51].T], [counts[:,52].T], [counts[:,53].T], [counts[:,54].T], [counts[:,55].T], [counts[:,56].T], [counts[:,57].T], [counts[:,58].T], [counts[:,59].T], [counts[:,60].T], [counts[:,61].T], [counts[:,62].T], [counts[:,63].T], [counts[:,64].T], [counts[:,65].T], [counts[:,66].T], [counts[:,67].T], [counts[:,68].T], [counts[:,69].T], [counts[:,70].T], [counts[:,71].T], [counts[:,72].T], [counts[:,73].T], [counts[:,74].T], [counts[:,75].T], [counts[:,76].T], [counts[:,77].T], [counts[:,78].T], [counts[:,79].T], [counts[:,80].T], [counts[:,81].T], [counts[:,82].T], [counts[:,83].T], [counts[:,84].T], [counts[:,85].T], [counts[:,86].T], [counts[:,87].T], [counts[:,88].T], [counts[:,89].T], [counts[:,90].T], [counts[:,91].T], [counts[:,92].T], [counts[:,93].T], [counts[:,94].T], [counts[:,95].T], [counts[:,96].T], [counts[:,97].T], [counts[:,98].T], [counts[:,99].T], [counts[:,100].T], [counts[:,101].T], [counts[:,102].T], [counts[:,103].T], [counts[:,104].T], [counts[:,105].T], [counts[:,106].T], [counts[:,107].T], [counts[:,108].T], [counts[:,109].T], [counts[:,110].T], [counts[:,111].T], [counts[:,112].T], [counts[:,113].T], [counts[:,114].T], [counts[:,115].T], [counts[:,116].T], [counts[:,117].T], [counts[:,118].T], [counts[:,119].T], [counts[:,120].T], [counts[:,121].T], [counts[:,122].T], [counts[:,123].T], [counts[:,124].T], [counts[:,125].T], [counts[:,126].T], [counts[:,127].T])) #number of counts as a table with respect to the energy channel -> echan_counts[0] are the counts for the first energy channel
        exptime = spectrum['EXPOSURE'] #length of the time intervall
        total_rate = np.divide(total_counts, exptime) #total count rate for each time intervall
        echan_rate = np.divide(echan_counts, exptime) #count rate per time intervall for each energy channel
        cstart = spectrum['TIME'] #start time of the time intervall
        cstop = spectrum['ENDTIME'] #end time of the time intervall
        bin_time = np.zeros((len(cstart),2), float) #combine the time limits of the counting intervals in one matrix
        bin_time[:,0] = cstart
        bin_time[:,1] = cstop
        gtstart = goodtime['START'] #start time of data collecting times (exiting SAA)
        gtstop = goodtime['STOP'] #end time of data collecting times (entering SAA)
        #times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
        good_time = np.zeros((len(gtstart),2), float) #combine the time limits of the goodtime intervals in one matrix
        good_time[:,0] = gtstart
        good_time[:,1] = gtstop
        return echan, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime










    def ctime(self, detector, day, seconds = 0):
        """This function reads a cspec file and stores the data in arrays of the form: echan[emin, emax], total_counts, echan_counts[echan], exptime, total_rate, echan_rate, cstart, cstop, gtstart, gtstop\n
        Input:\n
        readfile.ctime ( detector, day = YYMMDD, seconds = SSS )\n
        0 = energy channel interval\n
        1 = total number of counts\n
        2 = number of counts per energy channel\n
        3 = total count rate\n
        4 = count rate per energy channel\n
        5 = bin time interval[start, end]\n
        6 = good time interval[start, end]\n
        7 = time of exposure\n"""
        
        #read the file. Check if one wants to read a specific trigger file or a daily file
        if seconds == 0:
            filename = 'glg_ctime_' + str(detector) + '_' + str(day) + '_v00.pha'
        else:
            filename = 'glg_ctime_' + str(detector) + '_bn' + str(day) + str(seconds) + '_v00.pha'
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        fits_path = os.path.join(os.path.dirname(__dir__), 'ctime')
        filepath = os.path.join(fits_path, str(filename))
        fits = pyfits.open(filepath)
        energy = fits[1].data
        spectrum = fits[2].data
        goodtime = fits[3].data
        fits.close()
        
        #extract the data
        emin = energy['E_MIN'] #lower limit of the energy channels
        emax = energy['E_MAX'] #upper limit of the energy channels
        echan = np.zeros((len(emin),2), float) #combine the energy limits of the energy channels in one matrix
        echan[:,0] = emin
        echan[:,1] = emax
        counts = spectrum['COUNTS']
        total_counts = np.sum(counts, axis=1) #total number of counts for each time intervall
        echan_counts = np.vstack(([counts[:,0].T], [counts[:,1].T], [counts[:,2].T], [counts[:,3].T], [counts[:,4].T], [counts[:,5].T], [counts[:,6].T], [counts[:,7].T])) #number of counts as a table with respect to the energy channel -> echan_counts[0] are the counts for the first energy channel
        exptime = spectrum['EXPOSURE'] #length of the time intervall
        total_rate = np.divide(total_counts, exptime) #total count rate for each time intervall
        echan_rate = np.divide(echan_counts, exptime) #count rate per time intervall for each energy channel
        cstart = spectrum['TIME'] #start time of the time intervall
        cstop = spectrum['ENDTIME'] #end time of the time intervall
        bin_time = np.zeros((len(cstart),2), float) #combine the time limits of the counting intervals in one matrix
        bin_time[:,0] = cstart
        bin_time[:,1] = cstop
        gtstart = goodtime['START'] #start time of data collecting times (exiting SAA)
        gtstop = goodtime['STOP'] #end time of data collecting times (entering SAA)
        #times are in Mission Elapsed Time (MET) seconds. See Fermi webside or read_poshist for more information.
        good_time = np.zeros((len(gtstart),2), float) #combine the time limits of the goodtime intervals in one matrix
        good_time[:,0] = gtstart
        good_time[:,1] = gtstop
        return echan, total_counts, echan_counts, total_rate, echan_rate, bin_time, good_time, exptime










    def poshist(self, day):
        """This function reads a posthist file and stores the data in arrays of the form: sat_time, sat_pos, sat_lat, sat_lon, sat_q.\n
        Input:\n
        readfile.poshist ( day = YYMMDD )\n
        Output:\n
        0 = time\n
        1 = position (x, y, z)\n
        2 = latitude\n
        3 = longitude\n
        4 = quaternion matrix (q1, q2, q3, q4)"""
        
        #read the file
        filename = 'glg_poshist_all_' + str(day) + '_v00.fit'
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        fits_path = os.path.join(os.path.dirname(__dir__), 'poshist')
        filepath = os.path.join(fits_path, str(filename))
        fits = pyfits.open(filepath)
        data = fits[1].data
        fits.close()
        
        #extract the data
        sat_time = data.SCLK_UTC #Mission Elapsed Time (MET) seconds. The reference time used for MET is midnight (0h:0m:0s) on January 1, 2001, in Coordinated Universal Time (UTC). The FERMI convention is that MJDREF=51910 (UTC)=51910.0007428703703703703 (TT)
        sat_pos = np.array([data.POS_X, data.POS_Y, data.POS_Z]) #Position in J2000 equatorial coordinates
        sat_lat = data.SC_LAT
        sat_lon = data.SC_LON #Earth-angles -> considers earth rotation (needed for SAA)
        sat_q = np.array([data.QSJ_1, data.QSJ_2, data.QSJ_3, data.QSJ_4]) #Quaternionen -> 4D-space with which one can describe rotations (rocking motion); regarding the satellite system with respect to the J2000 geocentric coordinate system
        return sat_time, sat_pos, sat_lat, sat_lon, sat_q










    def saa(self):
        """This function reads the saa.dat file and returns the polygon in the form: saa[lat][lon]\n
        Input:\n
        readfile.saa()\n
        Output\n
        0 = saa[latitude][longitude]\n"""
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        saa_path = os.path.join(os.path.dirname(__dir__), 'saa')
        filepath = os.path.join(saa_path, 'saa.dat')
        poly = open(filepath)
        lines = poly.readlines()
        poly.close()
        saa_lat = []
        saa_lon = []#define latitude and longitude arrays
        for line in lines:#write file data into the arrays
            p = line.split()
            saa_lat.append(float(p[0]))
            saa_lon.append(float(p[1]))
        saa = np.array([saa_lat, saa_lon])#merge the arrays
        return saa

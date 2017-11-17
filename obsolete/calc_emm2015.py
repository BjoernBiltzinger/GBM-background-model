#!/usr/bin python2.7

import math
import os
import subprocess
from subprocess import Popen

import numpy as np
from astropy.table import Table
from numpy import linalg as LA

from gbmbkgpy.utils.external_prop import ExternalProps
from gbmbkgpy.work_module_refactor import calculate
from gbmbkgpy.work_module_refactor import detector
from gbmbkgpy.work_module_refactor import writefile

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()


def calc_altitude(day):
    """This function calculates the satellite's altitude for one day and stores the data in arrays of the form: altitude, earth_radius, sat_time\n
    Input:\n
    calc_altitude ( day = JJMMDD )\n
    Output:\n
    0 = altitude of the satellite above the WGS84 ellipsoid\n
    1 = radius of the earth at the position of the satellite\n
    2 = time (MET) in seconds"""
    
    poshist = rf.poshist(day)
    sat_time = poshist[0]
    sat_pos = poshist[1]
    sat_lat = poshist[2]
    sat_lon = poshist[3]
    
    ell_a = 6378137
    ell_b = 6356752.3142
    sat_lat_rad = sat_lat*2*math.pi/360.
    
    earth_radius = ell_a * ell_b / ((ell_a**2 * np.sin(sat_lat_rad)**2 + ell_b**2 * np.cos(sat_lat_rad)**2)**0.5)
    
    altitude = (LA.norm(sat_pos, axis=0) - earth_radius)/1000.
    
    return altitude, earth_radius, sat_time


def write_coord_file(day):
    """This function writes four coordinate files of the satellite for one day and returns the following information about the files: filepaths, directory\n
    Input:\n
    write_coord_file ( day = JJMMDD )\n
    Output:\n
    0 = filepaths\n
    1 = directory"""
    
    poshist = rf.poshist(day)
    sat_time = poshist[0]
    sat_pos = poshist[1]
    sat_lat = poshist[2]
    sat_lon = poshist[3] - 180.
    
    geometrie = calc_altitude(day)
    altitude = geometrie[0]
    earth_radius = geometrie[1]
    
    decimal_year = calc.met_to_date(sat_time)[4]
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'magnetic_field/' + str(day)
    fits_path = os.path.join(os.path.dirname(__dir__), directory)
    
    filename1 = 'magn_coor_' + str(day) + '_kt_01.txt'
    filename2 = 'magn_coor_' + str(day) + '_kt_02.txt'
    filename3 = 'magn_coor_' + str(day) + '_kt_03.txt'
    filename4 = 'magn_coor_' + str(day) + '_kt_04.txt'
        
    filepath1 = os.path.join(fits_path, str(filename1))
    filepath2 = os.path.join(fits_path, str(filename2))
    filepath3 = os.path.join(fits_path, str(filename3))
    filepath4 = os.path.join(fits_path, str(filename4))
    filepaths = [filepath1, filepath2, filepath3, filepath4]
    
    if not os.path.exists(fits_path):
        try:
            os.makedirs(fits_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    emm_file = open(filepath1, 'w')
    for i in range(0, len(sat_time)/4):
        emm_file.write(str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
    emm_file.close()
    
    emm_file = open(filepath2, 'w')
    for i in range(len(sat_time)/4, len(sat_time)/2):
        emm_file.write(str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
    emm_file.close()
    
    emm_file = open(filepath3, 'w')
    for i in range(len(sat_time)/2, len(sat_time)*3/4):
        emm_file.write(str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
    emm_file.close()
    
    emm_file = open(filepath4, 'w')
    for i in range(len(sat_time)*3/4, len(sat_time)):
        emm_file.write(str(decimal_year[i]) + ' E K' + str(altitude[i]) + ' ' + str(sat_lat[i]) + ' ' + str(sat_lon[i]) + '\n')
    emm_file.close()
    
    return filepaths, directory


def write_magn_file(day):
    """This function calls the c-programme of the EMM-2015 magnetic field model to calculate and write the magnetic field data for the four given coordinate files for one day and returns the paths of the magnetic field files: out_paths\n
    Input:\n
    write_magn_file ( day = JJMMDD )\n
    Output:\n
    0 = out_paths"""
    
    coord_files = write_coord_file(day)
    filepaths = coord_files[0]
    directory = coord_files[1]
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    fits_path_emm = os.path.join(os.path.dirname(__dir__), 'EMM2015_linux')
    emm_file = os.path.join(fits_path_emm, 'emm_sph_fil')
    
    out_paths = []
    for i in range(0, len(filepaths)):
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(__dir__), directory)
        out_name = 'magn_' + str(day) + '_kt_0' + str(i + 1) + '.txt'
        out_file = os.path.join(path, out_name)
        out_paths.append(out_file)
    
    
    for i in range(0, len(filepaths)):
        cmd = str(emm_file) + ' f ' + str(filepaths[i]) + ' ' + str(out_paths[i])
        result = Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=fits_path_emm)
    return out_paths


def write_magn_fits_file(day):
    """This function reads the magnetic field files of one day, writes the data into a fit file and returns the filepath: fitsfilepath\n
    Input:\n
    write_magn_fits_file ( day = JJMMDD )\n
    Output:\n
    0 = fitsfilepath"""
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    directory = 'magnetic_field/' + str(day)
    path = os.path.join(os.path.dirname(__dir__), directory)
    name1 = 'magn_' + str(day) + '_kt_01.txt'
    filename1 = os.path.join(path, name1)
    name2 = 'magn_' + str(day) + '_kt_02.txt'
    filename2 = os.path.join(path, name2)
    name3 = 'magn_' + str(day) + '_kt_03.txt'
    filename3 = os.path.join(path, name3)
    name4 = 'magn_' + str(day) + '_kt_04.txt'
    filename4 = os.path.join(path, name4)
    filenames = [filename1, filename2, filename3, filename4]
    
    outname = 'magn_' + str(day) + '_kt.txt'
    outfilename = os.path.join(path, outname)
    
    with open(outfilename, 'w') as outfile:
        with open(filenames[0]) as infile:
            for line in infile:
                outfile.write(line)
        for fname in filenames[1:]:
            with open(fname) as infile:
                for i, line in enumerate(infile):
                    if i > 0:
                        outfile.write(line)
    
    content = Table.read(outfilename, format='ascii')
    fitsname = 'magn_' + str(day) + '_kt.fits'
    fitsfilepath = os.path.join(path, fitsname)
    content.write(fitsfilepath, overwrite=True)
    
    return fitsfilepath


days = np.array([150927, 151120, 151126])

#for day in days:
#    write_magn_file(days[1])#write the four magnetic files for a day

#for day in days:
#    write_magn_fits_file(day)#combine existing four magnetic files of a day

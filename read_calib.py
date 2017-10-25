#!/usr/bin python2.7

import subprocess
from subprocess import Popen, PIPE
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfits
from numpy import linalg as LA
import ephem
from scipy import interpolate
import scipy.optimize as optimization
from astropy.time import Time
from astropy.table import Table
import fileinput
from datetime import datetime
from work_module import calculate
from work_module import detector
from work_module import readfile
from work_module import writefile
calc = calculate()
det = detector()
rf = readfile()
wf = writefile()


def read_calib(detector_type):
    """This function reads the calibration files of a given detector type and stores the data in arrays of the form: .\n
    Input:\n
    read_calib ( detector_type = NaI/BGO )\n
    Output:\n
    0 = """
    
    #Read the data files
    if str(detector_type) == 'BGO':
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        fits_path = os.path.join(os.path.dirname(__dir__), 'calibration')
        
        filename = 'Eff_Area-BGO-EQM-Y88-all_keV-Angles.txt'
        
        filepath = os.path.join(fits_path, str(filename))
        BGO_file = open(filepath)
        #store the file contents in a list of every row
        lines_898 = BGO_file.readlines()[2:38]
        BGO_file.close()
        BGO_file = open(filepath)
        lines_1836 = BGO_file.readlines()[39:75]
        BGO_file.close()
        
        x = []
        y1 = []
        y1_err = []
        y2 = []
        y2_err = []
        
        for line in lines_898:
            p1 = line.split()
            x.append(float(p1[0]))
            y1.append(float(p1[1]))
            y1_err.append(float(p1[2]))
        
        for line in lines_1836:
            p2 = line.split()
            y2.append(float(p2[1]))
            y2_err.append(float(p2[2]))
        
        x.append(x[22])
        y1.append(y1[22])
        y1_err.append(y1_err[22])
        y2.append(y2[22])
        y2_err.append(y2_err[22])
        x = np.array(x)
        x[23:] = x[23:] - 360.
        y1 = np.array(y1)
        y1_err = np.array(y1_err)
        y2 = np.array(y2)
        y2_err = np.array(y2_err)
        
        data = np.array([x,y1,y1_err,y2,y2_err])
        data = data[:,data[0].argsort()]
        
    elif str(detector_type) == 'NaI':
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        fits_path = os.path.join(os.path.dirname(__dir__), 'calibration')
        
        filename1 = 'Eff_Area-NaI-FM04-Cs137-32.89keV-Angles.txt'
        filename2 = 'Eff_Area-NaI-FM04-Cs137-661.66keV-Angles.txt'
        filename3 = 'Eff_Area-NaI-FM04-Hg203-279.2keV-Angles.txt'
        
        filepath1 = os.path.join(fits_path, str(filename1))
        filepath2 = os.path.join(fits_path, str(filename2))
        filepath3 = os.path.join(fits_path, str(filename3))
        NaI_file_33 = open(filepath1)
        #store the file contents in a list of every row
        lines_33 = NaI_file_33.readlines()
        NaI_file_33.close()
        
        NaI_file_662 = open(filepath2)
        #store the file contents in a list of every row
        lines_662 = NaI_file_662.readlines()
        NaI_file_662.close()
        
        NaI_file_279 = open(filepath3)
        #store the file contents in a list of every row
        lines_279 = NaI_file_279.readlines()
        NaI_file_279.close()
        
        x = []
        y1 = []
        y1_err = []
        y2 = []
        y2_err = []
        y3 = []
        y3_err = []
        
        for line in lines_33:
            p1 = line.split()
            x.append(float(p1[0]))
            y1.append(float(p1[1]))
            y1_err.append(float(p1[2]))
        
        for line in lines_279:
            p2 = line.split()
            y2.append(float(p2[1]))
            y2_err.append(float(p2[2]))
        
        for line in lines_662:
            p3 = line.split()
            y3.append(float(p3[1]))
            y3_err.append(float(p3[2]))
        
        x.append(x[27])
        y1.append(y1[27])
        y1_err.append(y1_err[27])
        y2.append(y2[27])
        y2_err.append(y2_err[27])
        y3.append(y3[27])
        y3_err.append(y3_err[27])
        x = np.array(x)
        x[28:] = x[28:] - 360.
        y1 = np.array(y1)
        y1_err = np.array(y1_err)
        y2 = np.array(y2)
        y2_err = np.array(y2_err)
        y3 = np.array(y3)
        y3_err = np.array(y3_err)
        
        data = np.array([x,y1,y1_err,y2,y2_err,y3,y3_err])
        data = data[:,data[0].argsort()]
        
    else:
        print 'Wrong input. See read_calib.__doc__ for more information.'
        data = []
    
    return data

"""detector_type = 'BGO'
data = read_calib(detector_type)

#plot the data and set the styles
#r-- is a red ------ line (just one - would be a regular line)
#bs are blue squares
#g^ are green triangles
#yo are yellow circles

x1 = data[0]
vector = data[3]
x2 = np.arange(181)*2 - 180.
weights = data[4]

smoothness = 200.

tck = interpolate.splrep(x1, vector, weights, s=smoothness)
new_vector = interpolate.splev(x2, tck, der=0)

new_vector[:90] = new_vector[91:][::-1]


outname = 'peak_eff_area_angle_calib_GBM_' + str(detector_type) + '_1836keV.txt'

__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'calibration'
path = os.path.join(os.path.dirname(__dir__), directory)

outfilename = os.path.join(path, outname)

with open(outfilename, 'w') as outfile:
    for i in range(0, len(new_vector)):
        line = str(x2[i]) + '   ' + str(new_vector[i]) + '\n'
        outfile.write(line)"""

outname1 = 'peak_eff_area_angle_calib_GBM_NaI_33keV.txt'
outname2 = 'peak_eff_area_angle_calib_GBM_NaI_279keV.txt'
outname3 = 'peak_eff_area_angle_calib_GBM_NaI_662keV.txt'
outname4 = 'peak_eff_area_angle_calib_GBM_BGO_898keV.txt'
outname5 = 'peak_eff_area_angle_calib_GBM_BGO_1836keV.txt'

__dir__ = os.path.dirname(os.path.abspath(__file__))
directory = 'calibration'
path = os.path.join(os.path.dirname(__dir__), directory)

outfilename1 = os.path.join(path, outname1)
outfilename2 = os.path.join(path, outname2)
outfilename3 = os.path.join(path, outname3)
outfilename4 = os.path.join(path, outname4)
outfilename5 = os.path.join(path, outname5)
outfilenames = [outfilename1, outfilename2, outfilename3, outfilename4, outfilename5]


files1 = open(outfilename1)
lines1 = files1.readlines()
files1.close()
files2 = open(outfilename2)
lines2 = files2.readlines()
files2.close()
files3 = open(outfilename3)
lines3 = files3.readlines()
files3.close()
files4 = open(outfilename4)
lines4 = files4.readlines()
files4.close()
files5 = open(outfilename5)
lines5 = files5.readlines()
files5.close()


lines = np.array([lines1, lines2, lines3, lines4, lines5])

x = []
y = [[],[],[],[],[]]

for line in lines[0]:
    p = line.split()
    x.append(float(p[0]))
    y[0].append(float(p[1]))

for i in range(1, len(lines)):
    for line in lines[i]:
        p = line.split()
        y[i].append(float(p[1]))

x = np.array(x)
y = np.array(y)

newname = 'peak_eff_area_angle_calib_GBM_all.txt'
newfilename = os.path.join(path, newname)

with open(newfilename, 'w') as newfile:
    for i in range(0, len(x)):
        line = str(x[i]) + '   ' + str(y[0,i]) + '   ' + str(y[1,i]) + '   ' + str(y[2,i]) + '   ' + str(y[3,i]) + '   ' + str(y[4,i]) + '\n'
        newfile.write(line)


content = Table.read(newfilename, format='ascii')
fitsname = 'peak_eff_area_angle_calib_GBM_all.fits'
fitsfilepath = os.path.join(path, fitsname)
content.write(fitsfilepath, overwrite=True)










'''plt.plot(data[0], data[3], 'r-', x2, new_vector, 'b-')#, data[0], data[5], 'g-')

#labeling the axes
plt.ylabel('effective area in cm^2')
plt.xlabel('angles in degrees')

#plot-title
plt.title('effective area of the ' + str(detector_type) + ' detectors including a fit of smoothness ' + str(smoothness))

#the range of the axes: [xmin, xmax, ymin, ymax]
#plt.axis([-10, 100, 100, 600])

#anzeigen lassen
plt.show()'''

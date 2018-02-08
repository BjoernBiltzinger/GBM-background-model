import os
import shutil
import urllib2

from gbmbkgpy.io.package_data import get_path_of_data_dir
from astropy.utils.data import download_file

def download_flares(year):
    """This function downloads a yearly solar flar data file and stores it in the appropriate folder\n
    Input:\n
    download.flares ( year = YYYY )\n"""

    # create the appropriate folder if it doesn't already exist and switch to it
    file_path = os.path.join(get_path_of_data_dir(), 'flares/')
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    #os.chdir(file_path)
    if year == 2017:
        url = (
        'ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_' + str(
            year) + '-ytd.txt')
    else:
        url = (
        'ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_' + str(
            year) + '.txt')
    file_name = '%s.dat' % str(year)

    path_to_file = download_file(url)

    shutil.move(path_to_file, file_path + file_name)



def download_lat_spacecraft(week):
    """This function downloads a weekly lat-data file and stores it in the appropriate folder\n
    Input:\n
    download.lat_spacecraft ( week = XXX )\n"""

    # create the appropriate folder if it doesn't already exist and switch to it
    file_path = os.path.join(get_path_of_data_dir(), 'lat/')
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    #os.chdir(file_path)

    url = 'http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w%d_p202_v001.fits' % week

    path_to_file = download_file(url)

    file_name = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % week

    shutil.move(path_to_file, file_path + file_name)

def download_data_file(date, type, detector):
    # create the appropriate folder if it doesn't already exist and switch to it

    year = '20%s' % date[:2]
    month = date[2:-2]
    day = date[-2:]

    file_path = os.path.join(get_path_of_data_dir(), type, '/', date)
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    # os.chdir(file_path)

    url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{}/{}/{}/current/glg_spechist_{}_{}_v00.fit'.format(year, month, day, detector, date)

    path_to_file = download_file(url)

    file_name = 'glg_spechist_n{}_{}_v00.fit'.format(year, month, day, detector, date)

    shutil.move(path_to_file, file_path + file_name)
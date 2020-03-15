import os
import shutil
from gbmbkgpy.io.file_utils import file_existing_and_readable

from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_external_data_dir
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

    data_path = get_path_of_external_data_dir()
    if not os.access(data_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(data_path)

    file_path = os.path.join(get_path_of_external_data_dir(), 'lat/')

    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    try:
        url = 'http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w%d_p202_v001.fits' % week

        path_to_file = download_file(url)
    except:
        url = 'http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w0%d_p202_v001.fits' % week

        path_to_file = download_file(url)
    file_name = 'lat_spacecraft_weekly_w%d_p202_v001.fits' % week

    shutil.move(path_to_file, file_path + file_name)


def download_data_file(date, type, detector='all'):
    """
    Download CTIME / CSPEC or POSHIST files

    :param date: string like '180407'
    :param type: string like 'ctime', 'cspec', 'poshist'
    :param detector: string like 'n1', 'n2' or 'all' for poshist
    :return:
    """

    year = '20%s' % date[:2]
    month = date[2:-2]
    day = date[-2:]

    data_path = get_path_of_external_data_dir()
    if not os.access(data_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(data_path)

    # poshist files are not stored in a sub folder of the date
    if type == 'poshist':
        file_path = os.path.join(data_path, type)
        file_type = 'fit'
    else:
        folder_path = os.path.join(data_path, type)
        if not os.access(folder_path, os.F_OK):
            print("Making New Directory")
            os.mkdir(folder_path)

        file_path = os.path.join(data_path, type, date)
        file_type = 'pha'

    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    try:
        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{0}/{1}/{2}/current/glg_{3}_{4}_{5}_v00.{6}'.format(year, month, day, type, detector, date, file_type)

        path_to_file = download_file(url)
    except:
        try:
            url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{0}/{1}/{2}/current/glg_{3}_{4}_{5}_v01.{6}'.format(year, month, day, type, detector, date, file_type)

            path_to_file = download_file(url)
        except:
            try:
                url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{0}/{1}/{2}/current/glg_{3}_{4}_{5}_v02.{6}'.format(year, month, day, type, detector, date, file_type)

                path_to_file = download_file(url)
            except:
                try:
                    url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{0}/{1}/{2}/current/glg_{3}_{4}_{5}_v03.{6}'.format(year, month, day, type, detector, date, file_type)

                    path_to_file = download_file(url)
                except:
                    try:
                        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{0}/{1}/{2}/current/glg_{3}_{4}_{5}_v04.{6}'.format(year, month, day, type, detector, date, file_type)

                        path_to_file = download_file(url)
                    except:
                        print('This url not found {}'.format(url))

    file_name = 'glg_{0}_{1}_{2}_v00.{3}'.format(type, detector, date, file_type)

    shutil.move(path_to_file, os.path.join(file_path, file_name))


def download_trigdata_file(trigger, type, detector='all'):
    """
    Download trigdata

    :param date: string like '180407'
    :param type: string like 'ctime', 'cspec', 'poshist'
    :param detector: string like 'n1', 'n2' or 'all' for poshist
    :return:
    """

    year = '20%s' % trigger[:2]
    month = trigger[2:-2]
    day = trigger[-2:]

    data_path = get_path_of_external_data_dir()
    if not os.access(data_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(data_path)


    folder_path = os.path.join(data_path, type)
    if not os.access(folder_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(folder_path)

    file_path = os.path.join(data_path, type, year)
    file_type = 'fit'

    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    try:
        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}/current/glg_{2}_{3}_bn{4}_v00.{5}'.format(year, trigger, type, detector, trigger, file_type)
        path_to_file = download_file(url)
    except:
        try:
            url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}/current/glg_{2}_{3}_bn{4}_v01.{5}'.format(year, trigger, type, detector, trigger, file_type)
            path_to_file = download_file(url)
        except:
            try:
                url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}/current/glg_{2}_{3}_bn{4}_v02.{5}'.format(year, trigger, type, detector, trigger, file_type)

                path_to_file = download_file(url)
            except:
                try:
                    url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}/current/glg_{2}_{3}_bn{4}_v03.{5}'.format(year, trigger, type, detector, trigger, file_type)

                    path_to_file = download_file(url)
                except:
                    try:
                        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}/current/glg_{2}_{3}_bn{4}_v04.{5}'.format(year, trigger, type, detector, trigger, file_type)
                        path_to_file = download_file(url)
                    except:
                        print('This url not found {}'.format(url))

    file_name = 'glg_{0}_{1}_bn{2}_v00.{3}'.format(type, detector, trigger, file_type)

    shutil.move(path_to_file, os.path.join(file_path, file_name))


def download_files(data_type, det, day):
    ### Download data-file and poshist file if not existing:
    datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, det, day)
    datafile_path = os.path.join(get_path_of_external_data_dir(), data_type, day, datafile_name)

    poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', day)
    poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)

    if not file_existing_and_readable(datafile_path):
        download_data_file(day, data_type, det)

    if not file_existing_and_readable(poshistfile_path):
        download_data_file(day, 'poshist')

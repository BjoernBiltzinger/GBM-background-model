import os
import shutil
from urllib.error import HTTPError
from astropy.utils.data import download_file

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.utils.mpi import check_mpi
from gbmbkgpy.io.package_data import get_path_of_data_dir, get_path_of_external_data_dir

using_mpi, rank, size, comm = check_mpi()

def download_flares(year):
    """This function downloads a yearly solar flar data file and stores it in the appropriate folder\n
    Input:\n
    download.flares ( year = YYYY )\n"""

    # create the appropriate folder if it doesn't already exist and switch to it
    file_path = os.path.join(get_path_of_data_dir(), "flares/")
    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    if year == 2017:
        url = (
            "ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_"
            + str(year)
            + "-ytd.txt"
        )
    else:
        url = (
            "ftp://ftp.ngdc.noaa.gov/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_"
            + str(year)
            + ".txt"
        )
    file_name = "%s.dat" % str(year)

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

    file_path = os.path.join(get_path_of_external_data_dir(), "lat/")

    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    try:
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w%d_p202_v001.fits"
            % week
        )

        path_to_file = download_file(url)
    except:
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/lat_spacecraft_weekly_w0%d_p202_v001.fits"
            % week
        )

        path_to_file = download_file(url)
    file_name = "lat_spacecraft_weekly_w%d_p202_v001.fits" % week

    shutil.move(path_to_file, file_path + file_name)


def download_gbm_file(date, data_type, detector="all"):
    """
    Download CTIME / CSPEC or POSHIST files

    :param date: string like '180407'
    :param data_type: string like 'ctime', 'cspec', 'poshist'
    :param detector: string like 'n1', 'n2' or 'all' for poshist
    :return:
    """

    if rank == 0:
        assert data_type in ['ctime', 'cspec', 'poshist'], "Wrong data_type..."

        year = "20%s" % date[:2]
        month = date[2:-2]
        day = date[-2:]

        data_path = get_path_of_external_data_dir()

        file_path = data_path / data_type / date
        file_path.mkdir(parents=True, exist_ok=True)
        # poshist files are not stored in a sub folder of the date
        if data_type == "poshist":
            file_type = "fit"
        else:
            file_type = "pha"

        final_path = (file_path /
                      f"glg_{data_type}_{detector}_{date}_v00.{file_type}")

        if not final_path.exists():
            base_url = (f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/"
                        f"gbm/daily/{year}/{month}/{day}/current/"
                        f"glg_{data_type}_{detector}_{date}_v0")

            path_to_file = None
            for version in ["0", "1", "2", "3", "4"]:
                try:
                    path_to_file = download_file(f"{base_url}{version}.{file_type}")
                except HTTPError:
                    pass
                if path_to_file is not None:
                    break

            if path_to_file is None:
                print(f"No version found for the url {base_url}?.{file_type}")

            shutil.move(path_to_file, final_path)

    if using_mpi:
        comm.Barrier()


def download_trigdata_file(trigger, type, detector="all"):
    """
    Download trigdata

    :param date: string like '180407'
    :param type: string like 'ctime', 'cspec', 'poshist'
    :param detector: string like 'n1', 'n2' or 'all' for poshist
    :return:
    """

    year = "20%s" % trigger[2:4]
    month = trigger[4:6]
    day = trigger[6:8]

    data_path = get_path_of_external_data_dir()
    if not os.access(data_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(data_path)

    folder_path = os.path.join(data_path, type)
    if not os.access(folder_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(folder_path)

    file_path = os.path.join(data_path, type, year)
    file_type = "fit"

    if not os.access(file_path, os.F_OK):
        print("Making New Directory")
        os.mkdir(file_path)

    try:
        url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/{1}/current/glg_{2}_{3}_{4}_v00.{5}".format(
            year, trigger, type, detector, trigger, file_type
        )
        path_to_file = download_file(url)
    except:
        try:
            url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/{1}/current/glg_{2}_{3}_{4}_v01.{5}".format(
                year, trigger, type, detector, trigger, file_type
            )
            path_to_file = download_file(url)
        except:
            try:
                url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/{1}/current/glg_{2}_{3}_{4}_v02.{5}".format(
                    year, trigger, type, detector, trigger, file_type
                )

                path_to_file = download_file(url)
            except:
                try:
                    url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/{1}/current/glg_{2}_{3}_{4}_v03.{5}".format(
                        year, trigger, type, detector, trigger, file_type
                    )

                    path_to_file = download_file(url)
                except:
                    try:
                        url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/{1}/current/glg_{2}_{3}_{4}_v04.{5}".format(
                            year, trigger, type, detector, trigger, file_type
                        )
                        path_to_file = download_file(url)
                    except:
                        print("This url not found {}".format(url))

    file_name = "glg_{0}_{1}_{2}_v00.{3}".format(type, detector, trigger, file_type)

    shutil.move(path_to_file, os.path.join(file_path, file_name))


def download_files(data_type, det, day):
    ### Download data-file and poshist file if not existing:
    datafile_name = "glg_{0}_{1}_{2}_v00.pha".format(data_type, det, day)
    datafile_path = os.path.join(
        get_path_of_external_data_dir(), data_type, day, datafile_name
    )

    poshistfile_name = "glg_{0}_all_{1}_v00.fit".format("poshist", day)
    poshistfile_path = os.path.join(
        get_path_of_external_data_dir(), "poshist", poshistfile_name
    )

    if not file_existing_and_readable(datafile_path):
        download_gbm_file(day, data_type, det)

    if not file_existing_and_readable(poshistfile_path):
        download_gbm_file(day, "poshist")

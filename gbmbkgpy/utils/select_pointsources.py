import astropy.io.fits as fits
import os
from astropy.utils.data import download_file
import shutil
import urllib.request

import h5py
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
from multiprocessing import Pool
import itertools

from gbmbkgpy.io.package_data import get_path_of_data_file
from gbmbkgpy.utils.progress_bar import progress_bar

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


class SelectPointsources(object):
    def __init__(self, limit1550Crab, time_string=None, mjd=None):
        """
        :param limit1550Crab: Threshold in fractions of the Crab in the energy range 15-50 keV
        :param time_string: Day as string, e.g. 201201 fot 1st December 2020
        :param mjd: Day as mjd
        """
        assert (time_string is None) or (
            mjd is None
        ), "Either enter time_string or mjd, not both!"

        if time_string is not None:
            t = Time(
                f"20{time_string[:2]}-{time_string[2:4]}-{time_string[4:6]}T00:00:00.000",
                format="isot",
                scale="utc",
            )

            self._time = t.mjd

        elif mjd is not None:
            self._time = mjd

        else:
            raise AssertionError("Please give either time_string or mjd")

        self._limit = (
            limit1550Crab * 0.000220 * 1000
        )  # 0.000220 cnts/s/cm^2 is 1 mCrab in 15-50 keV band for Swift

        # If file does not exist we have to create it
        if not os.path.exists(
            get_path_of_data_file("background_point_sources/", "pointsources_swift.h5")
        ):
            print(
                "The pointsource_swift.h5 file does not exist in the data folder."
                "To use the point source selection you need to create this file."
            )
            with tempfile.TemporaryDirectory() as tmpdirname:
                build_swift_pointsource_database(tmpdirname)
            if not os.path.exists(
                get_path_of_data_file(
                    "background_point_sources/", "pointsources_swift.h5"
                )
            ):
                raise AssertionError(
                    "The pointsource_swift.h5 file still does not exist in the data folder. Aborting..."
                )

        # Check if the time covered by the pointsource_swift file covers the day we want to use
        with h5py.File(
            get_path_of_data_file("background_point_sources/", "pointsources_swift.h5"),
            "r",
        ) as h:
            times_all = np.zeros((len(h.keys()), 20000))
            for i, key in enumerate(h.keys()):
                times = h[key]["Time"][()]
                times_all[i][: len(times)] = times

        min_mjd = np.min(times_all[times_all > 0])
        max_mjd = np.max(times_all)

        min_date = (Time(min_mjd, format="mjd").isot).split("T")[0]
        max_date = (Time(max_mjd, format="mjd").isot).split("T")[0]

        if not (self._time > min_mjd and self._time < max_mjd):
            start = yes_or_no(
                f"Your current pointsources_swift.h5 file does not cover the time"
                " you want to use. Do you want to update it?"
            )
            if start:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    build_swift_pointsource_database(tmpdirname)
        self.ps_sign_swift()

    def ps_sign_swift(self):
        """
        Return dict with the pointsources above a certain threshold on a given day in Swift
        :return: Dict with pointsources on this day above threshold
        """
        with h5py.File(
            get_path_of_data_file("background_point_sources/", "pointsources_swift.h5"),
            "r",
        ) as h:
            rates_all = np.zeros(len(h.keys()))
            errors_all = np.zeros(len(h.keys()))
            names = []
            ra = []
            dec = []
            for i, key in enumerate(h.keys()):
                names.append(key.replace("p", "+"))
                ra.append(h[key].attrs["Ra"])
                dec.append(h[key].attrs["Dec"])
                rates = h[key]["Rates"][()]
                errors = h[key]["Error"][()]
                times = h[key]["Time"][()]
                num = np.argmin(np.abs(times - self._time))
                if np.abs(times[num] - self._time) > 30:
                    # Closest data point is more than 30 days away. Skip this source.
                    # print(f"The closest data point for {key} is {np.abs(times[num]-time)} days away. Skip this source.")
                    pass
                else:
                    rates_all[i] = rates[num]
                    errors_all[i] = errors[num]

        names = np.array(names)
        ra = np.array(ra)
        dec = np.array(dec)

        sign_indices = rates_all > self._limit

        names_s, rates_all_s, errors_all_s, ra_s, dec_s = (
            names[sign_indices],
            rates_all[sign_indices],
            errors_all[sign_indices],
            ra[sign_indices],
            dec[sign_indices],
        )
        ps_dict = {}
        for i in range(len(names_s)):
            ps_dict[names_s[i]] = {
                "Rates": rates_all_s[i],
                "Errors": errors_all_s[i],
                "Ra": ra_s[i],
                "Dec": dec_s[i],
            }

        self._ps_dict = ps_dict

        return ps_dict

    def write_psfile(self, filename):
        if using_mpi:
            if rank == 0:
                do_it = True
            else:
                do_it = False
        else:
            do_it = True

        if do_it:

            if self._ps_dict is None:
                self.ps_sign_swift()

            if os.path.exists(filename):
                os.remove(filename)

            with open(filename, "w") as f:
                for key in self._ps_dict.keys():
                    ra = self._ps_dict[key]["Ra"]
                    dec = self._ps_dict[key]["Dec"]
                    f.write(f"{key}\t{ra}\t{dec}\n")

        if using_mpi:
            comm.barrier()

    def plot_ps(self):
        """
        Plot point sources in mollweide projection.
        :return: Figure
        """
        if self._ps_dict is None:
            self.ps_sign_swift()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="mollweide")
        colors = plt.cm.jet(np.linspace(0, 1, len(self._ps_dict)))
        for i, key in enumerate(self._ps_dict.keys()):
            ra = self._ps_dict[key]["Ra"]
            dec = self._ps_dict[key]["Dec"]
            rate = self._ps_dict[key]["Rates"]
            # Color point sources differently and set the marker size according to the rate
            ax.scatter(
                np.where(
                    np.deg2rad(ra) > np.pi, np.deg2rad(ra) - 2 * np.pi, np.deg2rad(ra)
                ),
                np.deg2rad(dec),
                label=key,
                color=colors[i],
                s=100 * rate,
            )

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=3)
        ax.grid()
        return fig

    def animate_swift_ps(self, limit1550Crab=None, mjd_list=None, dpi=150):
        """
        Animate point sources above a threshold seen by Swift as a function of time
        :param limit1550Crab: Threshold in fractions of the Crab in the energy range 15-50 keV
        :param mjd: Days as mjd; np.array
        :return: Animation object with important pointsources as function of time
        """
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection="mollweide")

        if limit1550Crab is None:
            limit = self._limit
        else:
            limit = limit1550Crab * 0.000220 * 1000

        if mjd_list is None:
            mjd_list = np.linspace(self._time - 50, self._time + 50, 100, dtype=int)

        ax.scatter([], [])
        ax.grid()

        # Read in the h5 file only once to save time;
        with h5py.File(
            get_path_of_data_file("background_point_sources/", "pointsources_swift.h5"),
            "r",
        ) as h:
            rates_all = np.zeros((len(h.keys()), 20000))
            errors_all = np.zeros((len(h.keys()), 20000))
            times_all = np.zeros((len(h.keys()), 20000))
            names = []
            ra = []
            dec = []
            for i, key in enumerate(h.keys()):
                names.append(key)
                ra.append(h[key].attrs["Ra"])
                dec.append(h[key].attrs["Dec"])
                rates = h[key]["Rates"][()]
                errors = h[key]["Error"][()]
                times = h[key]["Time"][()]

                rates_all[i][: len(rates)] = rates
                errors_all[i][: len(errors)] = errors
                times_all[i][: len(times)] = times

        names = np.array(names)
        ra = np.array(ra)
        dec = np.array(dec)

        ra_rad = np.where(
            np.deg2rad(ra) > np.pi, np.deg2rad(ra) - 2 * np.pi, np.deg2rad(ra)
        )

        dec_rad = np.deg2rad(dec)

        # Helper function to get the point sources above the threshold for all wanted days
        def get_ps(time):
            time_norm = np.abs(times_all - time)
            num = np.argmin(time_norm, axis=1)
            rates = np.zeros(len(rates_all))
            for i in range(len(rates_all)):
                if time_norm[i, num[i]] < 30:
                    rates[i] = rates_all[i, num[i]]
            mask = rates > limit
            names_this = names[mask]
            ra_this = ra_rad[mask]
            dec_this = dec_rad[mask]
            rates_this = rates[mask]
            return names_this, ra_this, dec_this, rates_this

        # Init of Animation
        def init():
            names, ras, decs, rates = get_ps(mjd_list[0])
            ax.set_title(f"{mjd_list[0]} MJD")
            ax.scatter(ras, decs, s=300 * rates)
            for i, txt in enumerate(names):
                ax.annotate(txt, (ras[i], decs[i] + 0.2), alpha=0.6)
            return (ax,)

        # Animation update
        def update(frame):
            names, ras, decs, rates = get_ps(mjd_list[frame])
            ax.cla()
            ax.set_title(f"{mjd_list[frame]} MJD")
            ax.scatter(ras, decs, s=300 * rates)
            ax.grid()
            for i, txt in enumerate(names):
                ax.annotate(txt, (ras[i], decs[i]), alpha=0.6)
            return (ax,)

        ani = FuncAnimation(
            fig, update, frames=np.arange(len(mjd_list)), init_func=init, blit=False
        )

        return ani


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[0] == "y" or reply[0] == "yes":
            return True
        if reply[0] == "n" or reply[0] == "no":
            return False


def download_ps_file(save_swift_data_folder, remote_file_name):
    final_path = f"{save_swift_data_folder}/{remote_file_name}.fits"
    # if not os.path.exists(final_path):
    try:
        url = f"https://swift.gsfc.nasa.gov/results/transients/weak/{remote_file_name}.lc.fits"
        path_to_file = download_file(url)
        shutil.move(path_to_file, final_path)
    except:
        url = (
            f"https://swift.gsfc.nasa.gov/results/transients/{remote_file_name}.lc.fits"
        )
        path_to_file = download_file(url)
        shutil.move(path_to_file, final_path)


def build_swift_pointsource_database(save_swift_data_folder, multiprocessing=False):
    """
    Build the swift pointsource database.
    :param save_data_folder: Folder where the swift data files are saved
    """
    if using_mpi:
        if rank == 0:
            do_it = True
        else:
            do_it = False
    else:
        do_it = True

    if do_it:

        if os.path.exists(
            get_path_of_data_file("background_point_sources/", "pointsources_swift.h5")
        ):

            with h5py.File(
                get_path_of_data_file(
                    "background_point_sources/", "pointsources_swift.h5"
                ),
                "r",
            ) as h:
                times_all = np.zeros((len(h.keys()), 20000))
                for i, key in enumerate(h.keys()):
                    times = h[key]["Time"][()]
                    times_all[i][: len(times)] = times

            min_mjd = np.min(times_all[times_all > 0])
            max_mjd = np.max(times_all)

            min_date = (Time(min_mjd, format="mjd").isot).split("T")[0]
            max_date = (Time(max_mjd, format="mjd").isot).split("T")[0]

            print(
                "You are about to recreate the point source database, which contains all"
                "point sources seen by Swift and their according brightness in the 15-50 keV"
                "band in Swift from start of Swift to today. This will take a while and about 500 MB "
                "will be downloaded."
            )
            print(
                "#############################################################################"
            )
            start = yes_or_no(
                f"Your current point source database covers the time from {min_mjd} mjd to {max_mjd} mjd ({min_date} to {max_date}). Do you want to update it?"
            )

        else:
            print(
                "You are about to create the point source database, which contains all"
                "point sources seen by Swift  and their according brightness in the 15-50 keV"
                "band in Swift from start of Swift to today."
                "This will take a while and about 500 MB "
                "will be downloaded."
            )
            print(
                "#############################################################################"
            )
            start = yes_or_no("Do you want to create it now?")

        if start:
            print("Start (re)creation of point source database...")
            print("Parse point source names from website...")
            # Parse pointsource names from website
            fp = urllib.request.urlopen(
                "https://swift.gsfc.nasa.gov/results/transients/"
            )
            mybytes = fp.read()

            mystr = mybytes.decode("utf8")
            fp.close()
            sp = mystr.split("a href")[1:]

            final = []
            for s in sp:
                if s[2:6] == "weak":
                    e = s.split("/")[1].split(">")[0][:-1]
                else:
                    e = s.split(">")[0][2:-1]
                if e != "":
                    final.append(e)
            final = final[16:-11]
            print("Download all needed swift datafiles...")
            # Download the datafile for every datafile - This will take a while...

            if multiprocessing:

                # this uses all available threds on the machine, this might not be save...
                with Pool() as pool:

                    download_arguments = zip(
                        itertools.repeat(save_swift_data_folder), final
                    )

                    results = pool.starmap(download_ps_file, download_arguments)

            else:
                with progress_bar(
                    len(final), title="Download all needed swift datafiles..."
                ) as p:

                    for remote_file_name in final:

                        download_ps_file(save_swift_data_folder, remote_file_name)

                        p.increase()

            print("Save everything we need in hdf5 point source database...")
            # Save everything in a h5 file for convenience and speed
            with h5py.File(
                get_path_of_data_file(
                    "background_point_sources/", "pointsources_swift.h5"
                ),
                "w",
            ) as h:

                for name in os.listdir(save_swift_data_folder):

                    path = os.path.join(save_swift_data_folder, name)
                    with fits.open(path) as f:
                        gr = h.create_group(name.split(".fits")[0])
                        gr.create_dataset("Rates", data=f["RATE"].data["RATE"])
                        gr.create_dataset("Error", data=f["RATE"].data["ERROR"])
                        gr.create_dataset("Time", data=f["RATE"].data["TIME"])
                        gr.attrs["Ra"] = f["RATE"].header["RA_OBJ"]
                        gr.attrs["Dec"] = f["RATE"].header["DEC_OBJ"]
            print("Done")
        else:
            print(
                "The point source database will [bold red]NOT[/bold red] be (re)created."
            )

    if using_mpi:
        comm.barrier()

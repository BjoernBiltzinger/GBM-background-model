import csv
import itertools
import os
import shutil
import tempfile
import urllib.request
from multiprocessing import Pool
import logging
import astropy.io.fits as fits
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.utils.data import download_file
from gbmbkgpy.io.file_utils import if_dir_containing_file_not_existing_then_make
from gbmbkgpy.io.package_data import (
    get_path_of_data_file,
    get_path_of_external_data_dir,
)
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmgeometry.utils.gbm_time import GBMTime
from matplotlib.animation import FuncAnimation
from scipy import interpolate

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
    def __init__(
        self,
        limit1550Crab,
        time_string=None,
        mjd=None,
        update=None,
        min_separation_angle=None,
    ):
        """
        :param limit1550Crab: Threshold in fractions of the Crab in the energy range 15-50 keV
        :param time_string: Day as string, e.g. 201201 fot 1st December 2020
        :param mjd: Day as mjd
        :param update: Update Catalog if selected time is outside file-range. \
        (True=force-update, False=no-update, None=ask)
        :param min_distance: Minimal separation distance in degree between the individual point sources \
        in case there are multiple ones keep the ps with higher rate
        """
        assert (time_string is None) or (
            mjd is None
        ), "Either enter time_string or mjd, not both!"

        if time_string is not None:
            self._t = Time(
                f"20{time_string[:2]}-{time_string[2:4]}-{time_string[4:6]}T00:00:00.000",
                format="isot",
                scale="utc",
            )

            self._time = self._t.mjd

        elif mjd is not None:
            self._time = mjd
            self._t = Time(mjd, format="mjd")

        else:
            raise AssertionError("Please give either time_string or mjd")

        self._limit = (
            limit1550Crab * 0.000220 * 1000
        )  # 0.000220 cnts/s/cm^2 is 1 mCrab in 15-50 keV band for Swift

        self._min_separation_angle = min_separation_angle

        self._bat_catalog = pd.read_table(
            get_path_of_data_file("background_point_sources/", "BAT_catalog_clean.dat"),
            names=["name1", "name2", "pl_index"],
        )

        self._ps_db_path = os.path.join(
            get_path_of_external_data_dir(),
            "background_point_sources",
            "pointsources_swift.h5",
        )

        self._ps_orbit_db_path = os.path.join(
            get_path_of_external_data_dir(),
            "background_point_sources",
            "pointsources_swift_orbit.h5",
        )

        self._ps_maxi_scans_db = os.path.join(
            get_path_of_external_data_dir(),
            "background_point_sources",
            "pointsources_maxi_scan.h5",
        )

        # If file does not exist we have to create it
        if not os.path.exists(self._ps_db_path):
            print(
                "The pointsource_swift.h5 file does not exist in the data folder."
                "To use the point source selection you need to create this file."
            )
            with tempfile.TemporaryDirectory() as tmpdirname:
                build_swift_pointsource_database(tmpdirname)
            if not os.path.exists(self._ps_db_path):
                raise AssertionError(
                    "The pointsource_swift.h5 file still does not exist in the data folder. Aborting..."
                )

        # Check if the time covered by the pointsource_swift file covers the day we want to use
        with h5py.File(self._ps_db_path, "r") as h:
            times_all = np.zeros((len(h.keys()), 20000))
            for i, key in enumerate(h.keys()):
                times = h[key]["Time"][()]
                times_all[i][: len(times)] = times

        min_mjd = np.min(times_all[times_all > 0])
        max_mjd = np.max(times_all)

        min_date = (Time(min_mjd, format="mjd").isot).split("T")[0]
        max_date = (Time(max_mjd, format="mjd").isot).split("T")[0]
        
        if not (self._time > min_mjd and self._time < max_mjd):
            force = False

            if update is None:
                start = yes_or_no(
                    f"Your current pointsources_swift.h5 file does not cover the time"
                    " you want to use. Do you want to update it?"
                )

            elif update:
                start = True
                force = True

            else:
                start = False

            if start:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    build_swift_pointsource_database(tmpdirname, force=force)

        self.ps_sign_swift()

    def ps_sign_swift(self):
        """
        Return dict with the pointsources above a certain threshold on a given day in Swift
        :return: Dict with pointsources on this day above threshold
        """
        with h5py.File(
            self._ps_db_path,
            "r",
        ) as h:
            rates_all = np.zeros(len(h.keys()))
            errors_all = np.zeros(len(h.keys()))
            names = []
            ra = []
            dec = []
            for i, key in enumerate(h.keys()):
                names.append(key.replace("p", "+").upper())
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

        if self._min_separation_angle is not None:
            filtered_idx = self.filter_ps_by_separation(
                ra_s, dec_s, rates_all_s, self._min_separation_angle
            )

            names_s, rates_all_s, errors_all_s, ra_s, dec_s = (
                names_s[filtered_idx],
                rates_all_s[filtered_idx],
                errors_all_s[filtered_idx],
                ra_s[filtered_idx],
                dec_s[filtered_idx],
            )

        ps_dict = {}
        for i in range(len(names_s)):

            # Add the spectral power law index from the bat catalog if existing
            # if not use pl_index=3
            res = self._bat_catalog.pl_index[
                self._bat_catalog[
                    self._bat_catalog.name2.str.upper() == names_s[i].upper()
                ].index
            ].values
            if len(res) == 0:
                pl_index = 3
            else:
                pl_index = float(res[0])

            ps_dict[names_s[i]] = {
                "Rates": rates_all_s[i],
                "Errors": errors_all_s[i],
                "Ra": ra_s[i],
                "Dec": dec_s[i],
                "bat_pl_index": pl_index,
            }

        self._ps_dict = ps_dict

        return ps_dict

    def filter_ps_by_separation(self, ra, dec, ps_rates, min_separation):
        """
        Recursively filter point_sources by their separation to the other point sources.
        """

        ps_locations = np.column_stack((ra, dec))
        original_index = np.array(range(len(ps_locations)))

        def filter_recursive(orig_index, locations, rates, min_sep):
            # Get the index of the closesd neigbour and the separation
            idx, sep, _ = match_coordinates_sky(
                SkyCoord(locations, unit=(u.deg, u.deg)),
                SkyCoord(locations, unit=(u.deg, u.deg)),
                nthneighbor=2,
            )
            # Check where separation smaller than threshold
            sep_small = sep.deg < min_sep

            # If no neighbour "too close" return the idx of the sources to keep
            if len(np.where(sep_small)[0]) == 0:
                return orig_index

            # sources with close neighbour and their close neigbours
            has_close = np.where(sep_small)[0]
            closest = idx[sep_small]

            neigbour_brighter = rates[has_close] < rates[closest]

            # keep the sources that don't have close neighbours,
            # the brighter sources and their brighter neigbours
            keep_idx = np.concatenate(
                (
                    np.where(~sep_small)[0],
                    has_close[~neigbour_brighter],
                    closest[neigbour_brighter],
                )
            )

            # Only keep the unique ids, as some source could have the same brighter neighbour
            keep_index = np.unique(keep_idx)

            locations = locations[keep_index]
            rates = rates[keep_index]
            orig_index = orig_index[keep_index]

            # run recursivly to filter all sources
            return filter_recursive(orig_index, locations, rates, min_sep)

        keep_index = filter_recursive(
            original_index, ps_locations, ps_rates, min_sep=min_separation
        )

        return keep_index

    def ps_time_variation(self):
        """
        Return dict with the pointsources above a certain threshold on a given day in Swift
        :return: Dict with pointsources on this day above threshold
        """
        print("Build time variablity of point sources")
        # If file does not exist we have to create it
        if not os.path.exists(self._ps_orbit_db_path):
            print(
                "The pointsource_swift_orbit.h5 file does not exist in the data folder."
                "To use the point source selection you need to create this file."
            )
            with tempfile.TemporaryDirectory() as tmpdirname:
                build_swift_pointsource_database(tmpdirname, orbit_resolution=True)

        # If file does not exist we have to create it
        if not os.path.exists(self._ps_orbit_db_path):
            print(
                "The pointsource_maxi_scans.h5 file does not exist in the data folder."
                "To use the point source selection you need to create this file."
            )
            with tempfile.TemporaryDirectory() as tmpdirname:
                build_maxi_pointsource_database(tmpdirname)

        # Use 3 days before and after for interpolation
        t_start_long = self._t - 3
        t_stop_long = self._t + 4

        met_start_long = GBMTime(t_start_long).met
        met_stop_long = GBMTime(t_stop_long).met

        t_start_day = self._t
        t_stop_day = self._t + 1

        met_start_day = GBMTime(t_start_day).met
        met_stop_day = GBMTime(t_stop_day).met

        ps_variation = {}

        with h5py.File(
            self._ps_orbit_db_path,
            "r",
        ) as ps_swift:
            with h5py.File(
                self._ps_maxi_scans_db,
                "r",
            ) as ps_maxi:

                for ps_name, ps in self._ps_dict.items():

                    h5_name = ps_name.replace("+", "p")

                    rates_swift = ps_swift[h5_name]["Rates"][()]
                    errors_swift = ps_swift[h5_name]["Error"][()]
                    times_swift = ps_swift[h5_name]["Time"][()]

                    # mask
                    mask_swift_long = np.logical_and(
                        times_swift > met_start_long, times_swift < met_stop_long
                    )

                    mask_swift_day = np.logical_and(
                        times_swift > met_start_day, times_swift < met_stop_day
                    )
                    num_data_points_swift_day = np.sum(mask_swift_day)
                    inter = [s for s in ps_maxi.keys() if ps_name.upper() in s]
                    if len(inter) > 0:
                        h5_name_maxi = inter[0]
                        rates_maxi = ps_maxi[h5_name_maxi]["Rates"][()]
                        errors_maxi = ps_maxi[h5_name_maxi]["Error"][()]
                        times_maxi = ps_maxi[h5_name_maxi]["Time"][()]

                        mask_maxi_long = np.logical_and(
                            times_maxi > met_start_long, times_maxi < met_stop_long
                        )

                        mask_maxi_day = np.logical_and(
                            times_maxi > met_start_day, times_maxi < met_stop_day
                        )
                        # Check which one has better coverage

                        num_data_points_maxi_day = np.sum(mask_maxi_day)

                    else:
                        num_data_points_maxi_day = -1

                    if num_data_points_maxi_day >= num_data_points_swift_day:
                        use_maxi = True
                        use_swift = False
                    else:
                        use_maxi = False
                        use_swift = True
                    print(
                        f"Maxi data points: {num_data_points_maxi_day}, Swift data points: {num_data_points_swift_day}"
                    )
                    if use_maxi:
                        print(f"Use maxi time variability for {ps_name}")
                        # Check number of data points on day
                        if num_data_points_maxi_day < 10:
                            print(
                                f"Only {num_data_points_maxi_day} data points for point source {ps_name}. This can go wrong...."
                            )

                        # Check that there is no big gap at start of day or at end of day
                        start_id = np.argwhere(times_maxi > met_start_day)[0]
                        if times_maxi[start_id] - times_maxi[start_id - 1] > 2 * 3600:
                            print(
                                f"At the start of the day there is {times_maxi[start_id]-times_maxi[start_id-1]} seconds with no data point for point source {ps_name}"
                            )

                        stop_id = np.argwhere(times_maxi < met_stop_day)[-1]
                        if times_maxi[stop_id + 1] - times_maxi[stop_id] > 2 * 3600:
                            print(
                                f"At the end of the day there is {times_maxi[stop_id+1]-times_maxi[stop_id]} seconds with no data point for point source {ps_name}"
                            )

                        rates = np.clip(rates_maxi[mask_maxi_long], a_min=0, a_max=None)
                        rates_norm = rates / (np.mean(rates))
                        errors = errors_maxi[mask_maxi_long]
                        errors_norm = errors / (np.mean(rates))
                        ps_variation[ps_name] = {
                            "times": times_maxi[mask_maxi_long],
                            "rates": rates_norm,
                            "errors": errors_norm,
                        }

                    if use_swift:
                        print(f"Use swift time variability for {ps_name}")
                        # Check number of data points on day
                        if num_data_points_swift_day < 10:
                            print(
                                f"Only {num_data_points_swift_day} data points for point source {ps_name}. This can go wrong...."
                            )

                        # Check that there is no big gap at start of day or at end of day
                        start_id = np.argwhere(times_swift > met_start_day)[0]
                        if times_swift[start_id] - times_swift[start_id - 1] > 2 * 3600:
                            print(
                                f"At the start of the day there is {times_swift[start_id]-times_swift[start_id-1]} seconds with no data point for point source {ps_name}"
                            )

                        stop_id = np.argwhere(times_swift < met_stop_day)[-1]
                        if times_swift[stop_id + 1] - times_swift[stop_id] > 2 * 3600:
                            print(
                                f"At the end of the day there is {times_swift[stop_id+1]-times_swift[stop_id]} seconds with no data point for point source {ps_name}"
                            )

                        rates = np.clip(
                            rates_swift[mask_swift_long], a_min=0, a_max=None
                        )
                        rates_norm = rates / (np.mean(rates))
                        errors = errors_swift[mask_swift_long]
                        errors_norm = errors / (np.mean(rates))
                        ps_variation[ps_name] = {
                            "times": times_swift[mask_swift_long],
                            "rates": rates_norm,
                            "errors": errors_norm,
                        }
                    # TODO combine maxi and swift (inter normalization?)

        ps_interpolators = {}

        for ps_name, ps in ps_variation.items():

            ps_interpolators[ps_name] = interpolate.UnivariateSpline(
                ps["times"], ps["rates"] / abs(np.mean(ps["rates"])), s=50, k=3
            )

        return ps_interpolators

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

            if_dir_containing_file_not_existing_then_make(filename)

            if os.path.exists(filename):
                os.remove(filename)

            with open(filename, "w") as f:
                for key in self._ps_dict.keys():
                    ra = self._ps_dict[key]["Ra"]
                    dec = self._ps_dict[key]["Dec"]
                    f.write(f"{key}\t{ra}\t{dec}\n")

        if using_mpi:
            comm.barrier()

    def write_all_psfile(self, filename):
        if using_mpi:
            if rank == 0:
                do_it = True
            else:
                do_it = False
        else:
            do_it = True

        if do_it:

            with h5py.File(
                    self._ps_db_path,
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

                names = np.array(names)
                ra = np.array(ra)
                dec = np.array(dec)

                if_dir_containing_file_not_existing_then_make(filename)

                if os.path.exists(filename):
                    os.remove(filename)

                with open(filename, "w") as f:
                    for i, _ in enumerate(names):
                        f.write(f"{names[i]}\t{ra[i]}\t{dec[i]}\n")

        if using_mpi:
            comm.barrier()

    def plot_ps(self, save_path=None):
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
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        lgd = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        ax.grid()
        if save_path != None:
            fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches="tight")
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
            self._ps_db_path,
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

    @property
    def ps_dict(self):
        return self._ps_dict


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[0] == "y" or reply[0] == "yes":
            return True
        if reply[0] == "n" or reply[0] == "no":
            return False


def download_ps_file(save_swift_data_folder, remote_file_name, orbit_resolution=False):
    final_path = f"{save_swift_data_folder}/{remote_file_name}.fits"
    # if not os.path.exists(final_path):
    if orbit_resolution:
        file_substring = ".orbit"
    else:
        file_substring = ""

    try:
        url = f"https://swift.gsfc.nasa.gov/results/transients/weak/{remote_file_name}{file_substring}.lc.fits"
        path_to_file = download_file(url)
        shutil.move(path_to_file, final_path)
    except:
        try:
            url = f"https://swift.gsfc.nasa.gov/results/transients/{remote_file_name}{file_substring}.lc.fits"
            path_to_file = download_file(url)
            shutil.move(path_to_file, final_path)
        except Exception as e:
            logging.exception(
                f"Downloading the fits file for {remote_file_name} resulted in {e}"
            )


def build_swift_pointsource_database(
    save_swift_data_folder, multiprocessing=False, force=False, orbit_resolution=False
):
    """
    Build the swift pointsource database.
    :param save_data_folder: Folder where the swift data files are saved
    """

    if orbit_resolution:
        filename = "pointsources_swift_orbit.h5"
    else:
        filename = "pointsources_swift.h5"

    database_file = os.path.join(
        get_path_of_external_data_dir(), "background_point_sources", filename
    )

    if_dir_containing_file_not_existing_then_make(database_file)

    if using_mpi:
        if rank == 0:
            do_it = True
        else:
            do_it = False
    else:
        do_it = True

    if do_it:

        if force:

            start = True

        else:

            if os.path.exists(database_file):

                with h5py.File(database_file, "r") as h:
                    times_all = np.zeros((len(h.keys()), 200000))
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

                # this uses all available threads on the machine, this might not be safe...
                with Pool() as pool:

                    if orbit_resolution:
                        download_arguments = zip(
                            itertools.repeat(save_swift_data_folder),
                            final,
                            itertools.repeat(True),
                        )

                    else:
                        download_arguments = zip(
                            itertools.repeat(save_swift_data_folder), final
                        )

                    results = pool.starmap(download_ps_file, download_arguments)

            else:
                with progress_bar(
                    len(final), title="Download all needed swift datafiles..."
                ) as p:

                    for remote_file_name in final:

                        download_ps_file(
                            save_swift_data_folder, remote_file_name, orbit_resolution
                        )

                        p.increase()

            print("Save everything we need in hdf5 point source database...")

            # Save everything in a h5 file for convenience and speed
            with h5py.File(database_file, "w") as h:

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
            print("The point source database will NOT be (re)created.")

    if using_mpi:
        comm.barrier()


def download_ps_maxi_file(save_maxi_data_folder, remote_file_name):
    final_path = f"{save_maxi_data_folder}/{remote_file_name}.fits"
    try:
        url = f"http://maxi.riken.jp/pubdata/v3.rkn/{remote_file_name}/glcscan_lcbg_hv0.csv"
        path_to_file = download_file(url)
        shutil.move(path_to_file, final_path)
    except:
        url = f"http://maxi.riken.jp/pubdata/v3.rkn/{remote_file_name}/glcscan_regbg_hv0.csv"
        path_to_file = download_file(url)
        shutil.move(path_to_file, final_path)


def calculate_met_from_mjd(mjd):
    """
    calculated the Fermi MET given MJD
    :return:
    """
    utc_tt_diff = np.ones(len(mjd)) * 69.184
    utc_tt_diff[mjd <= 57754.00000000] -= 1
    utc_tt_diff[mjd <= 57204.00000000] -= 1
    utc_tt_diff[mjd <= 56109.00000000] -= 1
    utc_tt_diff[mjd <= 54832.00000000] -= 1
    met = (mjd - 51910 - 0.0007428703703) * 86400.0 + utc_tt_diff

    return met


def build_maxi_pointsource_database(save_maxi_data_folder, multiprocessing=False):
    """
    Build the swift pointsource database.
    :param save_data_folder: Folder where the swift data files are saved
    """

    filename = "pointsources_maxi_scan.h5"

    database_file = os.path.join(
        get_path_of_external_data_dir(), "background_point_sources", filename
    )

    if_dir_containing_file_not_existing_then_make(database_file)

    if using_mpi:
        if rank == 0:
            do_it = True
        else:
            do_it = False
    else:
        do_it = True

    if do_it:

        if os.path.exists(database_file):

            with h5py.File(database_file, "r") as h:
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
                "point sources seen by Maxi and their according brightness in the 2-20 keV"
                "band in Maxi from start of Maxi to today. This will take a while and about 500 MB "
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
                "point sources seen by MAXI and their according brightness in the 15-50 keV"
                "band in MAXI from start of MAXI to today."
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

            import urllib

            fp = urllib.request.urlopen("http://maxi.riken.jp/pubdata/v3.rkn/")
            mybytes = fp.read()

            mystr = mybytes.decode("utf8")
            fp.close()

            sp = mystr.split("<th>")

            ids = []
            names = {}

            for line in sp:
                if "href" in line:
                    ids.append(line.split("<a href=")[1].split("/")[0][1:])
                    names[ids[-1]] = (
                        line.split("target_v3rkn")[1]
                        .split("</a>")[0][3:-1]
                        .replace(" ", "")
                        .upper()
                    )

            print("Download all needed swift datafiles...")
            # Download the datafile for every datafile - This will take a while...

            if multiprocessing:

                # this uses all available threds on the machine, this might not be save...
                with Pool() as pool:

                    download_arguments = zip(
                        itertools.repeat(save_maxi_data_folder), names
                    )

                    results = pool.starmap(download_ps_maxi_file, download_arguments)

            else:
                with progress_bar(
                    len(names), title="Download all needed swift datafiles..."
                ) as p:

                    for remote_file_name in names:

                        download_ps_maxi_file(save_maxi_data_folder, remote_file_name)

                        p.increase()

            print("Save everything we need in hdf5 point source database...")

            # Save everything in a h5 file for convenience and speed
            with h5py.File(database_file, "w") as h:

                for name in os.listdir(save_maxi_data_folder):
                    path = os.path.join(save_maxi_data_folder, name)
                    with open(path, newline="") as csvfile:

                        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
                        rates = np.array([])
                        errors = np.array([])
                        times = np.array([])
                        for i, row in enumerate(reader):
                            error = float(row[8])
                            rates = np.append(rates, float(row[7]))
                            errors = np.append(errors, error)
                            times = np.append(times, float(row[0]))
                        n = names[name.split(".fits")[0]].upper()
                        gr = h.create_group(n)
                        gr.create_dataset("Rates", data=rates)
                        gr.create_dataset("Error", data=errors)

                        # Transform maxi mjd times to GBM/SWIFT MET

                        mets = calculate_met_from_mjd(times)

                        gr.create_dataset("Time", data=mets)
                        # gr.attrs["Ra"] = f["LCDAT_PIBAND4"].header["RA"]
                        # gr.attrs["Dec"] = f["LCDAT_PIBAND4"].header["DEC"]

            print("Done")
        else:
            print("The point source database will NOT be (re)created.")

    if using_mpi:
        comm.barrier()

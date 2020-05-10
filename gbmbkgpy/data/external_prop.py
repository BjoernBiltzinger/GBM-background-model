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
from gbmbkgpy.io.package_data import (
    get_path_of_data_file,
    get_path_of_external_data_file,
)
from gbmbkgpy.modeling.point_source import PointSrc_fixed, PointSrc_free
import csv

from gbmbkgpy.io.downloading import download_files
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.binner import Rebinner

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


class ExternalProps(object):
    def __init__(self, dates, detectors, bgo_cr_approximation=False):
        """
        Build the external properties for a given day
        :param detectors: list
        :param dates: [YYMMDD, YYMMDD,]
        """

        assert len(dates[0]) == 6, "Day must be in format YYMMDD"
        self._detectors = detectors

        self._side_0 = ["n0", "n1", "n2", "n3", "n4", "n5"]
        self._side_1 = ["n6", "n7", "n8", "n9", "na", "nb"]

        # Global list which weeks where already added to the lat data (to prevent double entries later)
        self._weeks = np.array([])
        if not bgo_cr_approximation:
            for i, date in enumerate(dates):
                (
                    mc_l,
                    mc_b,
                    lat_time,
                    lat_geo,
                    lon_geo,
                ) = self._one_day_build_lat_spacecraft(date)
                if i == 0:
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
        else:
            self._build_bgo_cr_approximation(dates, detectors)

    def build_point_sources(
        self,
        det_responses,
        det_geometries,
        echans,
        include_all_ps=False,
        point_source_list=[],
        free_spectrum=[],
    ):
        """
        This function reads the point_sources.dat file and builds the point sources
        :param echans:
        :param include_all_ps:
        :param source_list:
        :param free_spectrum:
        :return:
        """

        file_path = get_path_of_data_file(
            "background_point_sources/", "point_sources.dat"
        )

        self._ps_df = pd.read_table(file_path, names=["name", "ra", "dec"])

        if include_all_ps:
            point_source_list = list(self._ps_df["name"])

        # instantiate dic of point source objects
        self._point_sources_dic = {}

        ### Single core calc ###
        for row in self._ps_df.itertuples():
            for i, element in enumerate(point_source_list):
                if row[1] == element:
                    if len(free_spectrum) > 0 and free_spectrum[i]:
                        self._point_sources_dic[row[1]] = PointSrc_free(
                            name=row[1],
                            ra=row[2],
                            dec=row[3],
                            det_responses=det_responses,
                            det_geometries=det_geometries,
                            echans=echans,
                        )

                    else:
                        self._point_sources_dic[row[1]] = PointSrc_fixed(
                            name=row[1],
                            ra=row[2],
                            dec=row[3],
                            det_responses=det_responses,
                            det_geometries=det_geometries,
                            echans=echans,
                            spectral_index=2.114,
                        )

    def bgo_cr_approximation(self, met):

        bgo_cr_rates = np.zeros((len(met), len(self._detectors), 2))

        for det_idx, det in enumerate(self._detectors):
            if det in self._side_0:
                bgo_cr_rates[:, det_idx, :] = self._bgo_0_rate_interp(met)
            elif det in self._side_1:
                bgo_cr_rates[:, det_idx, :] = self._bgo_1_rate_interp(met)
            else:
                raise AssertionError("Use a valid NaI det name to use this function.")

        return bgo_cr_rates

    def mc_l_rates(self, met):

        return np.tile(self.mc_l(met), (len(self._detectors), 1))

    def mc_l(self, met):
        """
        Get MC L for a given MET
        :param met: 
        :return: 
        """

        return self._mc_l_interp(met)

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
        year = "20%s" % date[:2]
        month = date[2:-2]
        dd = date[-2:]

        day = astro_time.Time("%s-%s-%s" % (year, month, dd))

        min_met = GBMTime(day).met

        max_met = GBMTime(day + u.Quantity(1, u.day)).met

        gbm_time = GBMTime(day)

        mission_week = np.floor(gbm_time.mission_week.value)

        filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % mission_week
        filepath = get_path_of_external_data_file("lat", filename)
        if using_mpi:
            if not file_existing_and_readable(filepath):
                if rank == 0:
                    download_lat_spacecraft(mission_week)
                comm.Barrier()
        else:
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

            if f["PRIMARY"].header["TSTART"] >= min_met:

                # we need to get week before

                week_before = True

                before_filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % (
                    mission_week - 1
                )
                before_filepath = get_path_of_external_data_file("lat", before_filename)
                if using_mpi:
                    if not file_existing_and_readable(before_filepath):
                        if rank == 0:
                            download_lat_spacecraft(mission_week - 1)
                        comm.Barrier()
                else:
                    if not file_existing_and_readable(before_filepath):
                        download_lat_spacecraft(mission_week - 1)

            if f["PRIMARY"].header["TSTOP"] <= max_met:

                # we need to get week after

                week_after = True

                after_filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % (
                    mission_week + 1
                )
                after_filepath = get_path_of_external_data_file("lat", after_filename)
                if using_mpi:
                    if not file_existing_and_readable(after_filepath):
                        if rank == 0:
                            download_lat_spacecraft(mission_week + 1)
                        comm.Barrier()
                else:
                    if not file_existing_and_readable(after_filepath):
                        download_lat_spacecraft(mission_week + 1)

            # first lets get the primary file
            if mission_week not in self._weeks:
                lat_time = np.mean(
                    np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                    axis=0,
                )
                mc_l = f["SC_DATA"].data["L_MCILWAIN"]
                mc_b = f["SC_DATA"].data["B_MCILWAIN"]
                lon_geo = f["SC_DATA"].data["LON_GEO"]
                lat_geo = f["SC_DATA"].data["LAT_GEO"]

                self._weeks = np.append(self._weeks, mission_week)
        # if we need to append anything to make up for the
        # dates not being included in the files
        # do it here... thanks Fermi!
        if week_before and (mission_week - 1) not in self._weeks:
            with fits.open(before_filepath) as f:
                lat_time_before = np.mean(
                    np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                    axis=0,
                )
                mc_l_before = f["SC_DATA"].data["L_MCILWAIN"]
                mc_b_before = f["SC_DATA"].data["B_MCILWAIN"]
                lon_geo_before = f["SC_DATA"].data["LON_GEO"]
                lat_geo_before = f["SC_DATA"].data["LAT_GEO"]

            mc_b = np.append(mc_b_before, mc_b)
            mc_l = np.append(mc_l_before, mc_l)
            lon_geo = np.append(lon_geo_before, lon_geo)
            lat_geo = np.append(lat_geo_before, lat_geo)
            lat_time = np.append(lat_time_before, lat_time)

            self._weeks = np.append(self._weeks, mission_week - 1)

        if week_after and (mission_week + 1) not in self._weeks:
            with fits.open(after_filepath) as f:
                lat_time_after = np.mean(
                    np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                    axis=0,
                )
                mc_l_after = f["SC_DATA"].data["L_MCILWAIN"]
                mc_b_after = f["SC_DATA"].data["B_MCILWAIN"]
                lon_geo_after = f["SC_DATA"].data["LON_GEO"]
                lat_geo_after = f["SC_DATA"].data["LAT_GEO"]
            mc_b = np.append(mc_b, mc_b_after)
            mc_l = np.append(mc_l, mc_l_after)
            lon_geo = np.append(lon_geo, lon_geo_after)
            lat_geo = np.append(lat_geo, lat_geo_after)
            lat_time = np.append(lat_time, lat_time_after)
            self._weeks = np.append(self._weeks, mission_week + 1)

        return mc_l, mc_b, lat_time, lat_geo, lon_geo

    def _calc_bgo_rates(self, date, bgo_det):

        data_type = "cspec"
        echans = np.arange(85, 105, 1)

        if using_mpi:
            if rank == 0:
                download_files(data_type, bgo_det, date)
            comm.barrier()
        else:
            download_files(data_type, bgo_det, date)

        datafile_name = "glg_{0}_{1}_{2}_v00.pha".format(data_type, bgo_det, date)
        datafile_path = os.path.join(
            get_path_of_external_data_dir(), data_type, date, datafile_name
        )

        with fits.open(datafile_path) as f:
            counts = f["SPECTRUM"].data["COUNTS"][:, echans[0]]
            for echan in echans[1:]:
                counts += f["SPECTRUM"].data["COUNTS"][:, echan]
            bin_start = f["SPECTRUM"].data["TIME"]
            bin_stop = f["SPECTRUM"].data["ENDTIME"]

        total_time_bins = np.vstack((bin_start, bin_stop)).T
        min_bin_width = 100

        this_rebinner = Rebinner(total_time_bins, min_bin_width)
        rebinned_time_bins = this_rebinner.time_rebinned
        (rebinned_counts,) = this_rebinner.rebin(counts)

        rates = rebinned_counts / (rebinned_time_bins[:, 1] - rebinned_time_bins[:, 0])

        # Add first time and last time with corresponding rate to rate_list

        rates = np.concatenate((rates[:1], rates, rates[-1:]))

        times = np.concatenate(
            (bin_start[:1], np.mean(rebinned_time_bins, axis=1), bin_stop[-1:])
        )

        return times, rates

    def _build_bgo_cr_approximation(self, dates, detectors):
        """
        Function that gets the count rate of the 85-105th energy channel of the BGO
        of the correct side and uses this as function proportional to the CR influence in
        the NaI energy channels. Makes less sense when SAA is included!
        :param date: Date
        :param det: NaI detector
        """
        get_b0 = False
        get_b1 = False

        for det_idx, det in enumerate(detectors):
            if det in self._side_0:
                get_b0 = True
            elif det in self._side_1:
                get_b1 = True
            else:
                raise AssertionError("Use a valid NaI det name to use this function.")

        if get_b0:

            for date_idx, date in enumerate(dates):

                bgo_0_times, bgo_0_rates = self._calc_bgo_rates(date, "b0")

                if date_idx == 0:
                    self._bgo_0_times = bgo_0_times
                    self._bgo_0_rates = bgo_0_rates
                else:
                    self._bgo_0_times = np.append(self._bgo_0_times, bgo_0_times)
                    self._bgo_0_rates = np.append(self._bgo_0_rates, bgo_0_rates)

            self._bgo_0_rate_interp = interpolate.UnivariateSpline(
                self._bgo_0_times, self._bgo_0_rates, s=1000, k=3
            )

        if get_b1:

            for date_idx, date in enumerate(dates):

                bgo_1_times, bgo_1_rates = self._calc_bgo_rates(date, "b1")

                if date_idx == 0:
                    self._bgo_1_times = bgo_1_times
                    self._bgo_1_rates = bgo_1_rates
                else:
                    self._bgo_1_times = np.append(self._bgo_1_times, bgo_1_times)
                    self._bgo_1_rates = np.append(self._bgo_1_rates, bgo_1_rates)

            self._bgo_1_rate_interp = interpolate.UnivariateSpline(
                self._bgo_1_times, self._bgo_1_rates, s=1000, k=3
            )

    def lat_acd(self, time_bins, use_side):

        data_path = "/home/bbiltzing/output_acd.csv"

        with open(data_path, "r") as f:
            lines = csv.reader(f)
            lines_final = []
            for line in lines:
                lines_final.append(",".join(line))

        timestamps = []
        dets = []
        counts = []
        delta_times = []
        sides = []

        for line in lines_final[1:]:
            timestamp, det, count, delta_time, side = line.split(",")
            timestamps.append(float(timestamp))
            dets.append(int(det))
            counts.append(int(count))
            delta_times.append(float(delta_time))
            sides.append(side)
        timestamps = np.array(timestamps)
        dets = np.array(dets)
        counts = np.array(counts)
        delta_times = np.array(delta_times)

        counts_A = []
        delta_times_A = []
        counts_B = []
        delta_times_B = []
        counts_C = []
        delta_times_C = []
        counts_D = []
        delta_times_D = []
        for i, time in enumerate(timestamps[::108]):
            counts_A.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "A"
                    ]
                )
            )
            delta_times_A.append(
                np.array(
                    [
                        delta_times[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "A"
                    ]
                )
            )
            counts_B.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "B"
                    ]
                )
            )
            delta_times_B.append(
                np.array(
                    [
                        delta_times[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "B"
                    ]
                )
            )
            counts_C.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "C"
                    ]
                )
            )
            delta_times_C.append(
                np.array(
                    [
                        delta_times[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "C"
                    ]
                )
            )
            counts_D.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "D"
                    ]
                )
            )
            delta_times_D.append(
                np.array(
                    [
                        delta_times[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "D"
                    ]
                )
            )
        counts_A = np.array(counts_A)
        counts_B = np.array(counts_B)
        counts_C = np.array(counts_C)
        counts_D = np.array(counts_D)
        delta_times_A = np.array(delta_times_A)
        delta_times_B = np.array(delta_times_B)
        delta_times_C = np.array(delta_times_C)
        delta_times_D = np.array(delta_times_D)

        rate_A = np.sum(counts_A, axis=1) / np.sum(delta_times_A, axis=1)
        rate_B = np.sum(counts_B, axis=1) / np.sum(delta_times_B, axis=1)
        rate_C = np.sum(counts_C, axis=1) / np.sum(delta_times_C, axis=1)
        rate_D = np.sum(counts_D, axis=1) / np.sum(delta_times_D, axis=1)

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
        for i in range(len(timestamps[::108]) / sum_timestamps):
            binned_timestamps.append(
                (
                    timestamps[::108][(i + 1) * sum_timestamps - 1]
                    + timestamps[::108][i * sum_timestamps]
                )
                / 2
            )
            rate_A_binned.append(
                np.sum(counts_A_all[i * sum_timestamps : (i + 1) * sum_timestamps])
                / np.sum(
                    time_delta_A_all[i * sum_timestamps : (i + 1) * sum_timestamps]
                )
            )
            rate_B_binned.append(
                np.sum(counts_B_all[i * sum_timestamps : (i + 1) * sum_timestamps])
                / np.sum(
                    time_delta_B_all[i * sum_timestamps : (i + 1) * sum_timestamps]
                )
            )
            rate_C_binned.append(
                np.sum(counts_C_all[i * sum_timestamps : (i + 1) * sum_timestamps])
                / np.sum(
                    time_delta_C_all[i * sum_timestamps : (i + 1) * sum_timestamps]
                )
            )
            rate_D_binned.append(
                np.sum(counts_D_all[i * sum_timestamps : (i + 1) * sum_timestamps])
                / np.sum(
                    time_delta_D_all[i * sum_timestamps : (i + 1) * sum_timestamps]
                )
            )
        rate_A_binned = np.array(rate_A_binned)
        rate_B_binned = np.array(rate_B_binned)
        rate_C_binned = np.array(rate_C_binned)
        rate_D_binned = np.array(rate_D_binned)
        binned_timestamps = np.array(binned_timestamps)

        if use_side == "A":
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_osubplot(111)
            # ax.scatter(timestamps[::108], rate_A, s=0.3)
            # fig.savefig('ext_prop_lat_acd.pdf')

            rate_A_binned[rate_A_binned > 210] = (
                rate_A_binned[rate_A_binned > 210]
                - (rate_A_binned[rate_A_binned > 210]).min()
            )
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_A_binned)
        elif use_side == "B":
            # rate_B_binned[rate_B_binned>100] = rate_B_binned[rate_B_binned>100] - (rate_B_binned[rate_B_binned>100]).min()
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_B_binned)
        elif use_side == "C":
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(binned_timestamps, rate_C_binned)
            fig.savefig("ext_prop_lat_acd.pdf")

            rate_C_binned[rate_C_binned > 210] = (
                rate_C_binned[rate_C_binned > 210]
                - (rate_C_binned[rate_C_binned > 210]).min()
            )

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(binned_timestamps, rate_C_binned)
            fig.savefig("ext_prop_lat_acd_2.pdf")

            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_C_binned)
        elif use_side == "D":
            interpolate_acd = interpolate.interp1d(binned_timestamps, rate_D_binned)

        return interpolate_acd(time_bins)

    def acd_saa_mask(self, time_bins):

        data_path = "/home/bbiltzing/output_acd.csv"

        with open(data_path, "r") as f:
            lines = csv.reader(f)
            lines_final = []
            for line in lines:
                lines_final.append(",".join(line))

        timestamps = []
        dets = []
        counts = []
        delta_times = []
        sides = []

        for line in lines_final[1:]:
            timestamp, det, count, delta_time, side = line.split(",")
            timestamps.append(float(timestamp))
            dets.append(int(det))
            counts.append(int(count))
            delta_times.append(float(delta_time))
            sides.append(side)
        timestamps = np.array(timestamps)
        dets = np.array(dets)
        counts = np.array(counts)
        delta_times = np.array(delta_times)

        counts_A = []
        counts_B = []
        counts_C = []
        counts_D = []

        for i, time in enumerate(timestamps[::108]):
            counts_A.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "A"
                    ]
                )
            )
            counts_B.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "B"
                    ]
                )
            )
            counts_C.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "C"
                    ]
                )
            )
            counts_D.append(
                np.array(
                    [
                        counts[j]
                        for j in range(i * 108, (i + 1) * 108)
                        if sides[j] == "D"
                    ]
                )
            )
        counts_A_all = np.sum(counts_A, axis=1)
        counts_B_all = np.sum(counts_B, axis=1)
        counts_C_all = np.sum(counts_C, axis=1)
        counts_D_all = np.sum(counts_D, axis=1)

        timestamp_zero = (
            np.sum(
                np.array([counts_A_all, counts_B_all, counts_C_all, counts_D_all]),
                axis=0,
            )
            > 1
        )

        # set last timestamp before and after SAA also to False
        i = 0
        while i < len(timestamp_zero) - 1:
            if not timestamp_zero[i + 1]:
                timestamp_zero[i] = False
            elif not timestamp_zero[i] and timestamp_zero[i + 1]:
                timestamp_zero[i + 1] = False
                # jump next list value
                i += 1
            i += 1

        timestamp_index = []
        i = 0
        for time_bin in time_bins:
            while i < len(timestamps):
                if time_bin[0] > timestamps[::108][i]:
                    timestamp_index.append(i)
                    break
                else:
                    i += 1

        mask = timestamp_zero[timestamp_index]

        return mask

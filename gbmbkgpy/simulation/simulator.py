import os
import h5py
import numpy as np
import scipy.interpolate as interpolate
import yaml
from scipy.integrate import trapz
from scipy.interpolate import interp1d

import astropy.io.fits as fits
import astropy.time as astro_time
import astropy.units as u
from gbm_drm_gen.drmgen import DRMGen
from gbmbkgpy.io.downloading import (
    download_data_file,
    download_flares,
    download_lat_spacecraft,
)
from gbmbkgpy.io.file_utils import (
    file_existing_and_readable,
    if_dir_containing_file_not_existing_then_make,
)
from gbmbkgpy.io.package_data import (
    get_path_of_data_file,
    get_path_of_external_data_dir,
    get_path_of_external_data_file,
)
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmgeometry import GBMTime, PositionInterpolator, gbm_detector_list

from gbmbkgpy.utils.binner import Rebinner


class BackgroundSimulator(object):
    """
    Simulates the background of GBM and writes fits files that can be read by the Background model
    A real data file is used to get the ebounds and time bins as well as a real poshist file to
    get the spacecraft position and pointings.

    The simulation output folder is created in the data dir set by the env var GBMDATA

    The simulation parameters have to be passes as a config dictionary or as yaml file.

    :Example Usage:
    simulator = BackgroundSimulator.from_config_file("config.yml")
    simulator.simulate_background()
    simulator.save_to_fits()
    """

    def __init__(self, config):

        self._config = config

        self._data_type = config["data_type"]

        self._day = config["day"]

        self._valid_det_names = [
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "na",
            "nb",
        ]

        self._download_data()

        self._setup_simulation()

    @classmethod
    def from_config_file(cls, config_yaml):

        with open(config_yaml, "r") as f:

            config = yaml.safe_load(f)

        return cls(config)

    def _download_data(self):
        # The datafile is only used to get the time bins and ebins
        # The used detector does not matter
        version = "v00"
        det = self._valid_det_names[0]

        # Download data-file and poshist file if not existing:
        datafile_name = "glg_{0}_{1}_{2}_{3}.pha".format(
            self._data_type, det, self._day, version
        )
        datafile_path = os.path.join(
            get_path_of_external_data_dir(), self._data_type, self._day, datafile_name
        )

        poshistfile_name = "glg_{0}_all_{1}_v00.fit".format("poshist", self._day)
        poshistfile_path = os.path.join(
            get_path_of_external_data_dir(), "poshist", poshistfile_name
        )

        if not file_existing_and_readable(datafile_path):
            download_data_file(self._day, self._data_type, det)

        if not file_existing_and_readable(poshistfile_path):
            download_data_file(self._day, "poshist")

        # save datafile_path for later use
        self._data_file = datafile_path

        # Save poshistfile_path for later usage
        self._poshist_file = poshistfile_path

    def _setup_simulation(self):

        # Setup geometry calculation
        position_interp = PositionInterpolator.from_poshist(self._poshist_file)

        step_size = np.floor(
            (position_interp.time).size / self._config["interp_steps"]
        ).astype(int)

        times_interp = np.append(
            position_interp.time[::step_size], position_interp.time[-1]
        )

        quaternions = position_interp.quaternion(times_interp)
        sc_positions = position_interp.sc_pos(times_interp)

        fermi_active_intervals = np.array(position_interp._on_times)

        # Init all lists
        earth_az_zen = []  # azimuth and zenith angle of earth in sat. frame

        # Give some output how much of the geometry is already calculated (progress_bar)
        with progress_bar(len(times_interp), title="Calculating earth position") as p:

            # Calculate the geometry for all times
            for step_idx, mean_time in enumerate(times_interp):

                det = gbm_detector_list[self._valid_det_names[0]](
                    quaternion=quaternions[step_idx],
                    sc_pos=sc_positions[step_idx],
                    time=astro_time.Time(position_interp.utc(mean_time)),
                )

                earth_az_zen.append(det.earth_az_zen_sat)

                p.increase()

        # Make the list numpy arrays
        earth_az_zen = np.array(earth_az_zen)

        # Calculate the earth position in cartesian coordinates
        earth_rad = np.deg2rad(earth_az_zen)

        earth_cart = np.zeros((len(earth_az_zen), 3))

        earth_cart[:, 0] = np.cos(earth_rad[:, 1]) * np.cos(earth_rad[:, 0])
        earth_cart[:, 1] = np.cos(earth_rad[:, 1]) * np.sin(earth_rad[:, 0])
        earth_cart[:, 2] = np.sin(earth_rad[:, 1])

        # Get the time bins and ebin edges of the data file
        ebin_in_edge = np.array(np.logspace(0.5, 3.7, 301), dtype=np.float32)

        with fits.open(self._data_file) as f:
            edge_start = f["EBOUNDS"].data["E_MIN"]
            edge_stop = f["EBOUNDS"].data["E_MAX"]

            bin_start = f["SPECTRUM"].data["TIME"]
            bin_stop = f["SPECTRUM"].data["ENDTIME"]

        # Sometimes there are corrupt time bins where the time bin start = time bin stop
        # So we have to delete these times bins
        idx_zero_bins = np.where(bin_start == bin_stop)[0]

        bin_start = np.delete(bin_start, idx_zero_bins)
        bin_stop = np.delete(bin_stop, idx_zero_bins)

        min_time_pos = min(position_interp.time)
        max_time_pos = max(position_interp.time)

        idx_below_min_time = np.where(bin_start < min_time_pos)
        idx_above_max_time = np.where(bin_stop > max_time_pos)
        idx_out_of_bounds = np.unique(
            np.hstack((idx_below_min_time, idx_above_max_time))
        )

        bin_start = np.delete(bin_start, idx_out_of_bounds)
        bin_stop = np.delete(bin_stop, idx_out_of_bounds)

        time_bins = np.vstack((bin_start, bin_stop)).T

        # Rebinn the time bins if a min_bin_width is passed
        if self._config.get("min_bin_width", 1e-9) > 1e-9:

            # Build SAA mask
            saa_mask = np.zeros(len(time_bins), dtype=bool)
            for active_interval in fermi_active_intervals:
                int_mask = np.logical_and(
                    time_bins[:, 0] > active_interval[0] + 10,
                    time_bins[:, 1] < active_interval[1] - 10,
                )

                saa_mask[int_mask] = True

            data_rebinner = Rebinner(
                time_bins, min_bin_width=self._config["min_bin_width"], mask=saa_mask
            )

            time_bins = data_rebinner.time_rebinned

        ebin_out_edge = np.append(edge_start, edge_stop[-1])

        self._edge_start = edge_start
        self._edge_stop = edge_stop

        self._bin_start = time_bins[:, 0]
        self._bin_stop = time_bins[:, 1]
        self._time_bins = time_bins
        self._ebin_in_edge = ebin_in_edge
        self._ebin_out_edge = ebin_out_edge

        self._echans = np.arange(len(self._ebin_out_edge) - 1)

        self._times_interp = times_interp
        self._quaternions = quaternions
        self._sc_positions = sc_positions
        self._earth_cartesian = earth_cart
        self._fermi_active_intervals = fermi_active_intervals

    def run(self):

        self._simulate_background()

        self._counts_detectors = self._counts_background

    def _simulate_background(self):

        counts_background = {}

        with progress_bar(
            len(self._valid_det_names),
            title="Simulating background for all 12 NaI detectors:",
        ) as p:
            for det_idx, det in enumerate(self._valid_det_names):

                counts_sum = np.zeros((len(self._time_bins), len(self._echans)))

                # Simulate CGB and Albedo
                if self._config.get("use_earth", False) or self._config.get(
                    "use_cgb", False
                ):

                    counts_earth, counts_cgb = self._simulate_albedo_cgb(
                        det_idx=det_idx,
                        spectrum_earth=self._config["sources"]["earth"]["spectrum"],
                        spectrum_cgb=self._config["sources"]["cgb"]["spectrum"],
                        n_grid=self._config["response"]["n_grid"],
                    )

                    if self._config.get("use_earth", False):

                        counts_sum += counts_earth

                    if self._config.get("use_cgb", False):

                        counts_sum += counts_cgb

                # Simulate Point Sources
                if self._config.get("use_ps", False):

                    for point_source in self._config["sources"]["point_sources"]:

                        counts_sum += self._simulate_pointsource(
                            det_idx=det_idx,
                            ra=point_source["ra"],
                            dec=point_source["dec"],
                            spectrum=point_source["spectrum"],
                        )

                # Simulate Constant Source
                if self._config.get("use_const", False):

                    counts_sum += self._simulate_constant(
                        norm=self._config["sources"]["constant"]["norm"]
                    )

                # Simulate Cosmic Rays
                if self._config.get("use_cr", False):

                    counts_sum += self._simulate_cosmic_rays(
                        norm=self._config["sources"]["cosmic_rays"]["norm"]
                    )

                # Simulate SAA
                if self._config.get("use_saa", False):

                    for saa_exit in self._fermi_active_intervals[:, 0]:

                        for exp_decay in self._config["sources"]["saa_decay"]:

                            counts_sum += self._simulate_saa(
                                t0=saa_exit,
                                norm=exp_decay["norm"],
                                decay_constant=exp_decay["decay_constant"],
                            )

                counts_background[det] = counts_sum

                p.increase()

        self._counts_background = counts_background

    def _simulate_albedo_cgb(self, det_idx, spectrum_earth, spectrum_cgb, n_grid):

        if spectrum_earth["model"] == "powerlaw":

            true_flux_earth = self._spectrum_int_pl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                e_norm=spectrum_earth["e_norm"],
                norm=spectrum_earth["norm"],
                index=spectrum_earth["index"],
            )

        elif spectrum_earth["model"] == "broken_powerlaw":

            true_flux_earth = self._spec_integral_bpl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                norm=spectrum_earth["norm"],
                break_energy=spectrum_earth["break_energy"],
                index1=spectrum_earth["index1"],
                index2=spectrum_earth["index2"],
            )

        else:
            raise NotImplementedError(
                "The selected spectral model for the Earth is not implemented"
            )

        if spectrum_cgb["model"] == "powerlaw":

            true_flux_cgb = self._spectrum_int_pl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                e_norm=spectrum_cgb["e_norm"],
                norm=spectrum_cgb["norm"],
                index=spectrum_cgb["index"],
            )

        elif spectrum_cgb["model"] == "broken_powerlaw":

            true_flux_cgb = self._spec_integral_bpl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                norm=spectrum_cgb["norm"],
                break_energy=spectrum_cgb["break_energy"],
                index1=spectrum_cgb["index1"],
                index2=spectrum_cgb["index2"],
            )

        else:
            raise NotImplementedError(
                "The selected spectral model for the CGB is not implemented"
            )

        (
            effective_response_earth,
            effective_response_cgb,
        ) = self._get_effective_response_albedo_cgb(det_idx=det_idx, n_grid=n_grid)

        folded_flux_earth = np.dot(true_flux_earth, effective_response_earth)

        folded_flux_cgb = np.dot(true_flux_cgb, effective_response_cgb)

        # Interpolate to time bins of real data file
        earth_rates_interpolator = interp1d(
            self._times_interp, folded_flux_earth, axis=0
        )
        cgb_rates_interpolator = interp1d(self._times_interp, folded_flux_cgb, axis=0)

        earth_rates = earth_rates_interpolator(self._time_bins)
        cgb_rates = cgb_rates_interpolator(self._time_bins)

        # Swap the axes for the integration
        earth_rates = np.swapaxes(earth_rates, 0, 2)
        earth_rates = np.swapaxes(earth_rates, 1, 2)

        cgb_rates = np.swapaxes(cgb_rates, 0, 2)
        cgb_rates = np.swapaxes(cgb_rates, 1, 2)

        # Integrate over time bins
        earth_counts = trapz(earth_rates, self._time_bins).T
        cgb_counts = trapz(cgb_rates, self._time_bins).T

        return earth_counts, cgb_counts

    def _simulate_pointsource(self, det_idx, ra, dec, spectrum):

        if spectrum["model"] == "powerlaw":

            flux = self._spectrum_int_pl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                e_norm=spectrum["e_norm"],
                norm=spectrum["norm"],
                index=spectrum["index"],
            )

        elif spectrum["model"] == "broken_powerlaw":

            flux = self._spec_integral_bpl(
                e1=self._ebin_in_edge[:-1],
                e2=self._ebin_in_edge[1:],
                norm=spectrum["norm"],
                break_energy=spectrum["break_energy"],
                index1=spectrum["index1"],
                index2=spectrum["index2"],
            )

        else:
            raise NotImplementedError(
                "The selected spectral model for this pointsource is not implemented"
            )

        response_matrix = []

        for j in range(len(self._quaternions)):
            response_step = (
                DRMGen(
                    self._quaternions[j],
                    self._sc_positions[j],
                    det_idx,
                    self._ebin_in_edge,
                    mat_type=0,
                    ebin_edge_out=self._ebin_out_edge,
                    occult=True,
                )
                .to_3ML_response(ra, dec)
                .matrix.T
            )

            response_matrix.append(response_step)

        response_matrix = np.array(response_matrix)

        count_rates = np.dot(flux, response_matrix)

        # Interpolate to time bins of real data file
        count_rates_interpolator = interp1d(self._times_interp, count_rates, axis=0)

        count_rates_interp = count_rates_interpolator(self._time_bins)

        # Swap the axes for the integration
        count_rates_interp = np.swapaxes(count_rates_interp, 0, 2)
        count_rates_interp = np.swapaxes(count_rates_interp, 1, 2)

        # Integrate over time bins
        counts = trapz(count_rates_interp, self._time_bins).T

        return counts

    def _simulate_constant(self, norm):

        count_rates = np.ones((len(self._echans), len(self._time_bins), 2))

        counts = trapz(count_rates, self._time_bins).T

        return norm * counts

    def _simulate_saa(self, t0, norm, decay_constant):

        idx_start = self._time_bins[:, 0] < t0

        tstart = self._time_bins[:, 0][~idx_start]
        tstop = self._time_bins[:, 1][~idx_start]

        counts = np.zeros((len(self._time_bins)))

        counts[~idx_start] = (
            -norm
            / decay_constant
            * (
                np.exp((t0 - tstop) * np.abs(decay_constant))
                - np.exp((t0 - tstart) * np.abs(decay_constant))
            )
        )

        counts_all_echans = np.tile(counts, (len(self._echans), 1)).T

        return counts_all_echans

    def _simulate_cosmic_rays(self, norm):

        lat_time, mc_l = self._get_mcl_from_lat_file()

        mc_l_interp = interpolate.interp1d(lat_time, mc_l)

        cr_count_rate = mc_l_interp(self._time_bins)

        # Remove vertical movement
        cr_count_rate[cr_count_rate > 0] = cr_count_rate[cr_count_rate > 0] - np.min(
            cr_count_rate[cr_count_rate > 0]
        )

        cr_counts = norm * trapz(cr_count_rate, self._time_bins).T

        counts_all_echans = np.tile(cr_counts, (len(self._echans), 1)).T

        return counts_all_echans

    def save_to_fits(self, overwrite=False):

        with progress_bar(
            12, title="Exporting simulation to fits for all 12 NaI detectors:"
        ) as p:
            for det_idx, det in enumerate(self._valid_det_names):

                # Add Poisson noise
                np.random.seed(self._config["random_seed"])

                counts_poisson = np.random.poisson(self._counts_detectors[det])

                # Write to fits file
                primary_hdu = fits.PrimaryHDU()

                # Ebounds
                c1 = fits.Column(name="E_MIN", array=self._edge_start, format="1E")

                c2 = fits.Column(name="E_MAX", array=self._edge_stop, format="1E")

                ebounds = fits.BinTableHDU.from_columns([c1, c2], name="EBOUNDS")

                # SPECTRUM
                if self._data_type == "ctime":

                    c3 = fits.Column(name="COUNTS", array=counts_poisson, format="8I")

                else:

                    c3 = fits.Column(name="COUNTS", array=counts_poisson, format="128I")

                c4 = fits.Column(name="TIME", array=self._bin_start, format="1D")

                c5 = fits.Column(name="ENDTIME", array=self._bin_stop, format="1D")

                data = fits.BinTableHDU.from_columns([c3, c4, c5], name="SPECTRUM")

                hdul = fits.HDUList([primary_hdu, ebounds, data])

                result_dir = os.path.join(
                    get_path_of_external_data_dir(),
                    "simulation",
                    self._data_type,
                    self._day,
                )

                if not os.path.exists(result_dir):

                    os.makedirs(result_dir)

                hdul.writeto(
                    os.path.join(
                        result_dir, f"glg_{self._data_type}_{det}_{self._day}_v00.pha"
                    ),
                    overwrite=overwrite,
                )

                p.increase()

    def _get_effective_response_albedo_cgb(self, det_idx, n_grid=4000):

        resp_grid_points, response_array = self._get_detector_responses(det_idx, n_grid)

        # Factor to multiply the responses with. Needed as the later spectra are given in units of
        # 1/sr. The sr_points gives the area of the sphere occulted by one point
        sr_points = 4 * np.pi / len(resp_grid_points)

        # Now lets seperate the parts of the Earth Albedo and the CGB
        # define the opening angle of the earth in degree
        earth_radius = 6371.0
        fermi_radius = np.sqrt(np.sum(self._sc_positions ** 2, axis=1))
        horizon_angle = 90 - np.rad2deg(np.arccos(earth_radius / fermi_radius))

        min_vis = np.deg2rad(horizon_angle)

        # Calculate the normalization of the spacecraft position vectors
        earth_cart_norm = np.sqrt(
            np.sum(self._earth_cartesian * self._earth_cartesian, axis=1)
        ).reshape((len(self._earth_cartesian), 1))

        # Calculate the normalization of the grid points of the response precalculation
        resp_grid_points_norm = np.sqrt(
            np.sum(resp_grid_points * resp_grid_points, axis=1)
        ).reshape((len(resp_grid_points), 1))

        tmp = np.clip(
            np.dot(
                self._earth_cartesian / earth_cart_norm,
                resp_grid_points.T / resp_grid_points_norm.T,
            ),
            -1,
            1,
        )

        # Calculate separation angle between
        # spacecraft and earth horizon
        ang_sep = np.arccos(tmp)

        # Create a mask with True when the Earth is in the FOV
        # and False when its CGB
        earth_occultion_idx = np.less(ang_sep.T, min_vis).T

        # Sum up the responses that are occulted by earth in earth_effective_response
        # and the others in cgb_effective_response
        # then mulitiply by the sr_points factor which is the area
        # of the unit sphere covered by every point
        effective_response_earth = (
            np.tensordot(earth_occultion_idx, response_array, [(1,), (0,)]) * sr_points
        )

        effective_response_cgb = (
            np.tensordot(~earth_occultion_idx, response_array, [(1,), (0,)]) * sr_points
        )

        return effective_response_earth, effective_response_cgb

    def _get_detector_responses(self, det_idx, n_grid):
        response_cache_file = os.path.join(
            get_path_of_external_data_dir(),
            "simulation",
            "response",
            self._day,
            self._data_type,
            f"effective_response_{n_grid}_{self._valid_det_names[det_idx]}.hd5",
        )

        if file_existing_and_readable(response_cache_file):

            print(f"Load response cache for detector {self._valid_det_names[det_idx]}")

            with h5py.File(response_cache_file, "r") as f:
                det = f.attrs["det"]
                data_type = f.attrs["data_type"]
                ngrid = f.attrs["n_grid"]

                ebin_in_edge = f["ebin_in_edge"][()]
                ebin_out_edge = f["ebin_out_edge"][()]
                resp_grid_points = f["points"][()]
                response_array = f["response_array"][()]

            # Assert that the loaded information is correct
            assert det == det_idx
            assert data_type == self._data_type
            assert n_grid == ngrid

            assert np.array_equal(ebin_in_edge, self._ebin_in_edge)
            assert np.array_equal(ebin_out_edge, self._ebin_out_edge)

        else:

            print(
                f"No response cache existing for detector {self._valid_det_names[det_idx]}. We will build it from scratch!"
            )
            # Create the points on the unit sphere
            resp_grid_points = self._fibonacci_sphere(samples=n_grid)

            # Initialize response list
            responses = []

            try:
                from pathos.multiprocessing import cpu_count
                from pathos.pools import ProcessPool as Pool

                using_multiprocessing = True

                multiprocessing_n_cores = int(
                    os.environ.get("gbm_bkg_multiprocessing_n_cores", cpu_count())
                )

            except:
                using_multiprocessing = False

            if not using_multiprocessing:

                # Create the DRM object (quaternions and sc_pos are dummy values, not important
                # as we calculate everything in the sat frame
                DRM = DRMGen(
                    np.array([0.0745, -0.105, 0.0939, 0.987]),
                    np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]),
                    det_idx,
                    self._ebin_in_edge,
                    mat_type=0,
                    ebin_edge_out=self._ebin_out_edge,
                    occult=True,
                )

                with progress_bar(
                    len(resp_grid_points),
                    title=f"Calculating response on a grid around detector {self._valid_det_names[det_idx]}",
                ) as p:

                    for point in resp_grid_points:

                        az = np.arctan2(point[1], point[0]) * 180 / np.pi
                        zen = np.arcsin(point[2]) * 180 / np.pi

                        matrix = DRM.to_3ML_response_direct_sat_coord(az, zen).matrix

                        responses.append(matrix.T)

                        p.increase()

            else:

                def get_response(point):
                    x, y, z = point[0], point[1], point[2]

                    zen = np.arcsin(z) * 180 / np.pi
                    az = np.arctan2(y, x) * 180 / np.pi

                    drm = DRMGen(
                        np.array([0.0745, -0.105, 0.0939, 0.987]),
                        np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]),
                        det_idx,
                        self._ebin_in_edge,
                        mat_type=0,
                        ebin_edge_out=self._ebin_out_edge,
                        occult=True,
                    )
                    matrix = drm.to_3ML_response_direct_sat_coord(az, zen).matrix

                    return matrix.T

                with Pool(multiprocessing_n_cores) as pool:
                    responses = pool.map(get_response, resp_grid_points)

            response_array = np.array(responses)

            if_dir_containing_file_not_existing_then_make(response_cache_file)

            with h5py.File(response_cache_file, "w") as f:
                f.attrs["det"] = det_idx
                f.attrs["data_type"] = self._data_type
                f.attrs["n_grid"] = n_grid

                f.create_dataset(
                    "ebin_in_edge", data=self._ebin_in_edge, compression="lzf"
                )

                f.create_dataset(
                    "ebin_out_edge", data=self._ebin_out_edge, compression="lzf"
                )

                f.create_dataset("points", data=resp_grid_points, compression="lzf")

                f.create_dataset(
                    "response_array", data=response_array, compression="lzf"
                )

        return resp_grid_points, response_array

    def _get_mcl_from_lat_file(self):
        # read the file
        day = astro_time.Time(f"20{self._day[:2]}-{self._day[2:-2]}-{self._day[-2:]}")

        min_met = GBMTime(day).met

        max_met = GBMTime(day + u.Quantity(1, u.day)).met

        gbm_time = GBMTime(day)

        mission_week = np.floor(gbm_time.mission_week.value)

        filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % mission_week
        filepath = get_path_of_external_data_file("lat", filename)

        if not file_existing_and_readable(filepath):
            download_lat_spacecraft(mission_week)

        # Init all arrays as empty arrays
        lat_time = np.array([])
        mc_l = np.array([])

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
                if not file_existing_and_readable(before_filepath):
                    download_lat_spacecraft(mission_week - 1)

            if f["PRIMARY"].header["TSTOP"] <= max_met:

                # we need to get week after

                week_after = True

                after_filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % (
                    mission_week + 1
                )
                after_filepath = get_path_of_external_data_file("lat", after_filename)
                if not file_existing_and_readable(after_filepath):
                    download_lat_spacecraft(mission_week + 1)

            # first lets get the primary file
            lat_time = np.mean(
                np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                axis=0,
            )
            mc_l = f["SC_DATA"].data["L_MCILWAIN"]

        # if we need to append anything to make up for the
        # dates not being included in the files
        # do it here... thanks Fermi!
        if week_before:
            with fits.open(before_filepath) as f:
                lat_time_before = np.mean(
                    np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                    axis=0,
                )
                mc_l_before = f["SC_DATA"].data["L_MCILWAIN"]

            mc_l = np.append(mc_l_before, mc_l)
            lat_time = np.append(lat_time_before, lat_time)

        if week_after:
            with fits.open(after_filepath) as f:
                lat_time_after = np.mean(
                    np.vstack((f["SC_DATA"].data["START"], f["SC_DATA"].data["STOP"])),
                    axis=0,
                )
                mc_l_after = f["SC_DATA"].data["L_MCILWAIN"]
            mc_l = np.append(mc_l, mc_l_after)
            lat_time = np.append(lat_time, lat_time_after)

        return lat_time, mc_l

    def _spectrum_pl(self, energy, e_norm, norm, index):
        return norm / (energy / e_norm) ** index

    def _spectrum_int_pl(self, e1, e2, e_norm, norm, index):

        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return:
        """
        return (
            (e2 - e1)
            / 6.0
            * (
                self._spectrum_pl(e1, e_norm, norm, index)
                + 4 * self._spectrum_pl((e1 + e2) / 2.0, e_norm, norm, index)
                + self._spectrum_pl(e2, e_norm, norm, index)
            )
        )

    def _spectrum_bpl(self, energy, norm, break_energy, index1, index2):

        return norm / (
            (energy / break_energy) ** index1 + (energy / break_energy) ** index2
        )

    def _spec_integral_bpl(self, e1, e2, norm, break_energy, index1, index2):
        """
        Calculates the flux of photons between two energies
        :param e1: lower e bound
        :param e2: upper e bound
        :return:
        """
        return (
            (e2 - e1)
            / 6.0
            * (
                self._spectrum_bpl(e1, norm, break_energy, index1, index2)
                + 4
                * self._spectrum_bpl(
                    (e1 + e2) / 2.0, norm, break_energy, index1, index2
                )
                + self._spectrum_bpl(e2, norm, break_energy, index1, index2)
            )
        )

    def _fibonacci_sphere(self, samples=1):
        """
        Calculate equally distributed points on a unit sphere using fibonacci
        :params samples: number of points
        """
        rnd = 1.0

        points = []
        offset = 2.0 / samples
        increment = np.pi * (3.0 - np.sqrt(5.0))

        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % samples) * increment

            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x, y, z])

        return np.array(points)

    @property
    def counts_detectors(self):
        return self._counts_detectors

    @property
    def time_bins(self):
        return self._time_bins

    @property
    def time_bin_width(self):
        return np.diff(self._time_bins, axis=1)

    @property
    def config(self):
        return self._config

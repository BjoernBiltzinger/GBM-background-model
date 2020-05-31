import numpy as np
from gbm_drm_gen.drmgen import DRMGen
import os
import h5py
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.file_utils import (
    file_existing_and_readable,
    if_dir_containing_file_not_existing_then_make,
)
import astropy.io.fits as fits
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

valid_det_names = [
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


class Response_Precalculation(object):
    def __init__(
        self,
        detectors,
        dates,
        echans,
        Ngrid=40000,
        Ebin_edge_incoming=None,
        data_type="ctime",
        trigger=None,
        simulation=False
    ):
        self._echans = echans

        responses = {}

        for det in detectors:
            responses[det] = Det_Response_Precalculation(
                det, dates, echans, Ngrid, Ebin_edge_incoming, data_type, trigger, simulation
            )

        self._responses = responses

    @property
    def responses(self):
        return self._responses

    @property
    def echans(self):
        return self._echans


class Det_Response_Precalculation(object):
    """
    With this class one can precalculate the response on a equally distributed point grid
    around the detector. Is used later to calculate the rates of spectral sources
    like the earth or the CGB
    """

    def __init__(
        self,
        det,
        dates,
        echans,
        Ngrid=40000,
        Ebin_edge_incoming=None,
        data_type="ctime",
        trigger=None,
        simulation=False,
    ):
        """
        initialize the grid around the detector and set the values for the Ebins of incoming and detected photons
        :param det: which detector is used
        :param Ngrid: Number of Gridpoints for Grid around the detector
        :param Ebin_edge_incoming: Ebins edges of incomming photons
        :param Ebin_edge_detector: Ebins edges of detector
        """
        assert (
            det in valid_det_names
        ), "Invalid det name. Must be one of these {} but is {}.".format(
            valid_det_names, det
        )

        assert (
            type(dates[0]) == str and len(dates[0]) == 6
        ), "Day must be a string of the format YYMMDD, but is {}".format(dates[0])

        assert type(Ngrid) == int, "Ngrid has to be an integer, but is a {}.".format(
            type(Ngrid)
        )

        if Ebin_edge_incoming is not None:
            assert (
                type(Ebin_edge_incoming) == np.ndarray
            ), "Invalid type for mean_time. Must be an array but is {}.".format(
                type(Ebin_edge_incoming)
            )

        assert (
            data_type == "ctime" or data_type == "cspec " or data_type == "trigdat"
        ), "Please use a valid data_type (ctime, cspec or trigdat). Your input is {}.".format(
            data_type
        )

        if data_type == "ctime":
            assert (
                type(echans)
                and max(echans) <= 7
                and min(echans) >= 0
                and all(isinstance(x, int) for x in echans)
            ), "Echan_list variable must be a list and can only have integer entries between 0 and 7"

        if data_type == "cspec":
            assert (
                type(echans)
                and max(echans) <= 127
                and min(echans) >= 0
                and all(isinstance(x, int) for x in echans)
            ), "Echan_list variable must be a list and can only have integer entries between 0 and 7"

        if data_type == "trigdat":
            assert (
                type(echans)
                and max(echans) <= 7
                and min(echans) >= 0
                and all(isinstance(x, int) for x in echans)
            ), "Echan_list variable must be a list and can only have integer entries between 0 and 7"

            assert (
                trigger is not None
            ), "If you use trigdat data you have to provide a trigger."

        self._data_type = data_type

        self._echans = echans

        self._Ngrid = Ngrid

        self._detector = det

        # Translate the n0-nb and b0,b1 notation to the detector 0-14 notation that is used
        # by the response generator
        self._det = valid_det_names.index(det)

        if self._data_type == "ctime" or self._data_type == "trigdat":
            self._echan_mask = np.zeros(8, dtype=bool)
            self._echan_mask[self._echans] = True

        elif self._data_type == "cspec":
            self._echan_mask = np.zeros(128, dtype=bool)
            self._echan_mask[self._echans] = True

        if Ebin_edge_incoming is None:
            # Incoming spectrum between ~3 and ~5000 keV in 300 bins
            self._Ebin_in_edge = np.array(np.logspace(0.5, 3.7, 301), dtype=np.float64)
        else:
            # Use the user defined incoming energy bin
            self._Ebin_in_edge = Ebin_edge_incoming.astype(np.float64)

        # TODO: If we use multiple days then the Edges of the energy bins are not correct,
        # TODO: should we calculate a seperate response for each day?

        # Read in the datafile to get the energy boundaries
        if data_type == "trigdat":
            self._Ebin_out_edge = np.array(
                [3.4, 10.0, 22.0, 44.0, 95.0, 300.0, 500.0, 800.0, 2000.0],
                dtype=np.float64,
            )

            response_cache_file = os.path.join(
                get_path_of_external_data_dir(),
                "response",
                "trigdat",
                f"effective_response_{Ngrid}_{det}.hd5",
            )

            if file_existing_and_readable(response_cache_file):

                print(f"Load response cache for detector {det}")

                self._load_response_cache(response_cache_file)

            else:

                print(
                    f"No response cache existing for detector {det}. We will build it from scratch!"
                )

                # Create the points on the unit sphere
                self._points = np.array(self._fibonacci_sphere(samples=Ngrid))

                # Calculate the reponse for all points on the unit sphere
                self._calculate_responses()

                if using_mpi:

                    if rank == 0:

                        self._save_response_cache(response_cache_file)

                else:

                    self._save_response_cache(response_cache_file)

        else:
            datafile_name = "glg_{0}_{1}_{2}_v00.pha".format(data_type, det, dates[0])

            if simulation:

                datafile_path = os.path.join(
                    get_path_of_external_data_dir(), "simulation", data_type, dates[0], datafile_name
                )

            else:

                datafile_path = os.path.join(
                    get_path_of_external_data_dir(), data_type, dates[0], datafile_name
                )

            with fits.open(datafile_path) as f:
                edge_start = f["EBOUNDS"].data["E_MIN"]
                edge_stop = f["EBOUNDS"].data["E_MAX"]

            self._Ebin_out_edge = np.append(edge_start, edge_stop[-1])

            # Create the points on the unit sphere
            self._points = self._fibonacci_sphere(samples=Ngrid)

            # Calculate the reponse for all points on the unit sphere
            self._calculate_responses()

        self._get_needed_responses()

    @property
    def points(self):
        return self._points

    @property
    def Ngrid(self):
        return self._Ngrid

    @property
    def all_response_array(self):
        return self._all_response_array

    @property
    def response_array(self):
        return self._response_array

    @property
    def det(self):
        return self._det

    @property
    def detector(self):
        return self._detector

    @property
    def Ebin_in_edge(self):
        return self._Ebin_in_edge

    @property
    def Ebin_out_edge(self):
        return self._Ebin_out_edge

    @property
    def data_type(self):
        return self._data_type

    def set_Ebin_edge_incoming(self, Ebin_edge_incoming):
        """
        Set new Ebins for the incoming photons
        :param Ebin_edge_incoming:
        :return:
        """
        self._Ebin_in_edge = Ebin_edge_incoming

    def set_Ebin_edge_outcoming(self, Ebin_edge_outcoming):
        """
        set new Ebins for the detector
        :param Ebin_edge_outcoming:
        :return:
        """
        self._Ebin_out_edge = Ebin_edge_outcoming

    def _load_response_cache(self, cache_file):
        with h5py.File(cache_file, "r") as f:
            detector = f.attrs["detector"]
            det = f.attrs["det"]
            data_type = f.attrs["data_type"]
            n_grid = f.attrs["ngrid"]

            ebin_in_edge = f["ebin_in_edge"][()]
            ebin_out_edge = f["ebin_out_edge"][()]
            points = f["points"][()]
            all_response_array = f["response_array"][()]

        assert detector == self.detector
        assert det == self.det
        assert data_type == self.data_type
        assert n_grid == self.Ngrid

        assert np.array_equal(ebin_in_edge, self.Ebin_in_edge)
        assert np.array_equal(ebin_out_edge, self.Ebin_out_edge)

        self._points = points
        self._all_response_array = all_response_array

    def _save_response_cache(self, cache_file):
        if_dir_containing_file_not_existing_then_make(cache_file)

        with h5py.File(cache_file, "w") as f:
            f.attrs["detector"] = self.detector
            f.attrs["det"] = self.det
            f.attrs["data_type"] = self.data_type
            f.attrs["ngrid"] = self.Ngrid

            f.create_dataset("ebin_in_edge", data=self.Ebin_in_edge, compression="lzf")

            f.create_dataset(
                "ebin_out_edge", data=self.Ebin_out_edge, compression="lzf"
            )

            f.create_dataset("points", data=self.points, compression="lzf")

            f.create_dataset(
                "response_array", data=self.all_response_array, compression="lzf"
            )

    def _get_needed_responses(self):
        """
        Get the needed reponses for this run
        """
        self._response_array = self.all_response_array[:,:,self._echan_mask]
        # We do not need this anymore
        del self._all_response_array

    def _response(self, x, y, z, DRM):
        """
        Gives the instrument response for a certain point on the unit sphere around the detector

        :param x: x-positon on unit sphere in sat. coord
        :param y: y-positon on unit sphere in sat. coord
        :param z: z-positon on unit sphere in sat. coord
        :param DRM: The DRM object
        :return: response object
        """
        zen = np.arcsin(z) * 180 / np.pi
        az = np.arctan2(y, x) * 180 / np.pi
        return DRM.to_3ML_response_direct_sat_coord(az, zen)

    def _calculate_responses(self):
        """
        Function to calculate the responses from all the points on the unit sphere.
        """
        # Initialize response list
        responses = []
        # Create the DRM object (quaternions and sc_pos are dummy values, not important
        # as we calculate everything in the sat frame

        DRM = DRMGen(
            np.array([0.0745, -0.105, 0.0939, 0.987]),
            np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]),
            self._det,
            self.Ebin_in_edge,
            mat_type=0,
            ebin_edge_out=self._Ebin_out_edge,
        )

        # If MPI is used split up the points among the used cores to speed up
        if using_mpi:

            # This builds some really large arrays which mpi can sometimes not handle anymore
            # Therefore we have to separate the calculation and broadcasting in several runs of smaller arrays

            # If Ngrid is smaller than 5000 or we are using ctime data everything is fine and mpi works in a single run
            if (
                self._Ngrid <= 5000
                or self._data_type == "ctime"
                or self._data_type == "trigdat"
            ):

                points_per_rank = float(self._Ngrid) / float(size)
                points_lower_index = int(np.floor(points_per_rank * rank))
                points_upper_index = int(np.floor(points_per_rank * (rank + 1)))

                if rank == 0:

                    with progress_bar(
                        points_upper_index - points_lower_index,
                        title="Calculating response on a grid around detector {}. "
                        "This shows the progress of rank 0. All other should be about the same.".format(
                            self.detector
                        ),
                    ) as p:

                        for point in self._points[
                            points_lower_index:points_upper_index
                        ]:
                            # get the response of every point
                            matrix = self._response(
                                point[0], point[1], point[2], DRM
                            ).matrix
                            responses.append(matrix.T)

                            p.increase()

                else:

                    for point in self._points[points_lower_index:points_upper_index]:
                        # get the response of every point
                        matrix = self._response(
                            point[0], point[1], point[2], DRM
                        ).matrix
                        responses.append(matrix.T)

                # Collect all results in rank=0 and broadcast the final array to all ranks in the end
                responses = np.array(responses)
                responses_g = comm.gather(responses, root=0)
                if rank == 0:
                    responses_g = np.concatenate(responses_g)

                # broadcast the resulting list to all ranks
                responses = comm.bcast(responses_g, root=0)

            else:
                # Split the grid points in runs with 4000 points each
                num_split = int(np.ceil(self._Ngrid / 4000.0))

                # Save start and stop index of every run
                N_grid_start = np.arange(0, num_split * 4000, 4000)
                N_grid_stop = np.array([])
                for i in range(num_split):
                    if i == num_split - 1:
                        N_grid_stop = np.append(N_grid_stop, self._Ngrid)
                    else:
                        N_grid_stop = np.append(N_grid_stop, (i + 1) * 4000)

                # Calcualte the response for all runs and save them as separate arrays in one big array
                responses_all_split = []
                for j in range(num_split):
                    responses_split = []
                    n_start = N_grid_start[j]
                    n_stop = N_grid_stop[j]

                    # Split up the points of this run among the mpi ranks
                    points_per_rank = float(n_stop - n_start) / float(size)
                    points_lower_index = int(np.floor(points_per_rank * rank) + n_start)
                    points_upper_index = int(
                        np.floor(points_per_rank * (rank + 1)) + n_start
                    )

                    if rank == 0:
                        print(
                            "We have to split up the response precalculation in {} runs."
                            " MPI can not handle everything at once.".format(num_split)
                        )

                        with progress_bar(
                            points_upper_index - points_lower_index,
                            title="Calculating response on a grid around detector {}, run {} of {}."
                            " This shows the progress of rank 0. All other should be about the same.".format(
                                self.detector, j + 1, num_split
                            ),
                        ) as p:

                            for point in self._points[
                                points_lower_index:points_upper_index
                            ]:
                                # get the response of every point
                                matrix = self._response(
                                    point[0], point[1], point[2], DRM
                                ).matrix
                                responses_split.append(matrix.T)
                                p.increase()

                    else:

                        for point in self._points[
                            points_lower_index:points_upper_index
                        ]:
                            # get the response of every point
                            matrix = self._response(
                                point[0], point[1], point[2], DRM
                            ).matrix
                            responses_split.append(matrix.T)

                    # Collect all results in rank=0 and broadcast the final array to all ranks in the end
                    responses_split = np.array(responses_split)
                    responses_split_g = comm.gather(responses_split, root=0)
                    if rank == 0:
                        responses_split_g = np.concatenate(responses_split_g)

                    responses_split = comm.bcast(responses_split_g, root=0)

                    # Add results of this run to the big array
                    responses_all_split.append(responses_split)

                # Concatenate the big array to get one array with length Ngrid where the entries are the responses
                # of the points

                responses = np.concatenate(responses_all_split)

        else:
            with progress_bar(
                len(self._points),
                title="Calculating response on a grid around detector {}. "
                "This shows the progress of rank 0. All other should be about the same.".format(
                    self.detector
                ),
            ) as p:
                for point in self._points:
                    # get the response of every point
                    matrix = self._response(point[0],
                                            point[1],
                                            point[2], DRM).matrix
                    responses.append(matrix.T)
                    p.increase()

        self._all_response_array = np.array(responses)

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

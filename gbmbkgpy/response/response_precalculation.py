import numpy as np

from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.utils.mpi import check_mpi

using_mpi, rank, size, comm = check_mpi()


def fibonacci_sphere(samples=1):
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


class ResponsePrecalculation:

    def __init__(self, response_generator, Ngrid=40000):
        self._response_generator = response_generator
        self._Ngrid = Ngrid

        self._points = fibonacci_sphere(samples=Ngrid)
        self._calculate_responses()

    def _calculate_responses(self):
        """
        Function to calculate the responses from all the points on the unit sphere.
        """
        # Initialize response list
        responses = []

        if self._Ngrid > 5000:
            # we have to split the calc in several parts in case we are using mpi
            endpoint_per_run = np.arange(4000, self._Ngrid, 4000, dtype=int)
            endpoint_per_run = np.append(endpoint_per_run, self._Ngrid)
        else:
            endpoint_per_run = np.array([self._Ngrid])

        num_calcs = len(endpoint_per_run)
        for i, endpoint in enumerate(endpoint_per_run):
            if i == 0:
                startpoint = 0
            else:
                startpoint = endpoint_per_run[i-1]

            points_per_rank = float(endpoint-startpoint) / float(size)
            points_lower_index = (int(np.floor(points_per_rank * rank)) +
                                  startpoint)
            points_upper_index = (int(np.floor(points_per_rank * (rank + 1))) +
                                  startpoint)

            # Only rank==0 gives some output how much of the geometry is
            # already calculated (progress_bar)
            hidden = False if rank == 0 else True

            with progress_bar(
                    points_upper_index-points_lower_index,
                    title=f"Calculating response calc {i} out of {num_calcs}."
                    "This shows the progress of rank 0. "
                    "All other should be about the same.",
                    hidden=hidden,
            ) as p:
                for point in self._points[
                        points_lower_index:points_upper_index
                ]:
                    # get the response of every point
                    matrix = self._response_generator.calc_response_xyz(
                        point[0], point[1], point[2]
                    )

                    responses.append(matrix)

                    p.increase()

        responses = np.array(responses)
        if using_mpi:
            responses_g = comm.gather(responses, root=0)
            if rank == 0:
                responses_g = np.concatenate(responses_g)

            # broadcast the resulting list to all ranks
            responses = comm.bcast(responses_g, root=0)

        # mult with area per point
        self._response_array = np.array(responses)*(4*np.pi/self._Ngrid)

    @property
    def response_grid(self):
        return self._response_array

    @property
    def drm_gen(self):
        return self._response_generator

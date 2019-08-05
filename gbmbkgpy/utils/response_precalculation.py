import numpy as np
from gbm_drm_gen.drmgen import DRMGen
import os
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
import astropy.io.fits as fits

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


valid_det_names = ['n0','n1' ,'n2' ,'n3' ,'n4' ,'n5' ,'n6' ,'n7' ,'n8' ,'n9' ,'na' ,'nb']

class Response_Precalculation(object):
    """
    With this class one can precalculate the response on a equally distributed point grid
    around the detector. Is used later to calculate the rates of spectral sources
    like the earth or the CGB
    """
    def __init__(self, det, day, Ngrid=40000, Ebin_edge_incoming=None, data_type='ctime'):
        """
        initialize the grid around the detector and set the values for the Ebins of incoming and detected photons
        :param det: which detector is used
        :param Ngrid: Number of Gridpoints for Grid around the detector
        :param Ebin_edge_incoming: Ebins edges of incomming photons
        :param Ebin_edge_detector: Ebins edges of detector
        """
        
        assert det in valid_det_names, 'Invalid det name. Must be one of these {} but is {}.'.format(valid_det_names, det)
        assert type(day[0])==str and len(day[0])==6, 'Day must be a string of the format YYMMDD, but is {}'.format(day)
        assert type(Ngrid) == int, 'Ngrid has to be an integer, but is a {}.'.format(type(Ngrid))
        if Ebin_edge_incoming!=None:
            assert type(Ebin_edge_incoming)==np.ndarray, 'Invalid type for mean_time. Must be an array but is {}.'.format(type(Ebin_edge_incoming))
        assert data_type=='ctime' or data_type=='cspec', 'Please use a valid data_type (ctime or cspec). Your input is {}.'.format(data_type)

        
        self._data_type = data_type

        # If no values for Ngrid or Ebin_incoming are given we use the standard values

        self._Ngrid = Ngrid
        
        if Ebin_edge_incoming==None:
            # Incoming spectrum between ~3 and ~5000 keV in 300 bins
            self._Ebin_in_edge = np.array(np.logspace(0.5, 3.7, 301), dtype=np.float32)
        else:
            # Use the user defined incoming energy bins
            self._Ebin_in_edge = Ebin_edge_incoming
            
        # Read in the datafile to get the energy boundaries
        datafile_name = 'glg_{0}_{1}_{2}_v00.pha'.format(data_type, det, day[0])
        datafile_path = os.path.join(get_path_of_external_data_dir(), data_type, day[0], datafile_name)
        with fits.open(datafile_path) as f:
            edge_start = f['EBOUNDS'].data['E_MIN']
            edge_stop = f['EBOUNDS'].data['E_MAX']

        self._Ebin_out_edge = np.append(edge_start, edge_stop[-1])

        # Create the points on the unit sphere
        self._points = np.array(self._fibonacci_sphere(samples=Ngrid))

        # Translate the n0-nb and b0,b1 notation to the detector 0-14 notation that is used
        # by the response generator
        if det[0]=='n':
            if det[1]=='a':
                self._det=10
            elif det[1]=='b':
                self._det=11
            else:
                self._det=int(det[1])
        elif det[0]=='b':
            if det[1]=='0':
                self._det=12
            elif det[1]=='1':
                self._det=13

        # Calculate the reponse for all points on the unit sphere
        self._calculate_responses()

    @property
    def points(self):
        return self._points

    @property
    def Ngrid(self):
        return self._Ngrid

    @property
    def responses(self):
        return self._responses

    @property
    def det(self):
        return self._det
    
    @property
    def Ebin_in_edge(self):
        return self._Ebin_in_edge

    @property
    def Ebin_out_edge(self):
        return self._Ebin_out_edge

    def set_Ebin_edge_incoming(self, Ebin_edge_incoming):
        """
        Set new Ebins for the incoming photons
        :param Ebin_edge_incoming:
        :return:
        """
        self._Ebin_in_edge=Ebin_edge_incoming

    def set_Ebin_edge_outcoming(self, Ebin_edge_outcoming):
        """
        set new Ebins for the detector
        :param Ebin_edge_outcoming:
        :return:
        """
        self._Ebin_out_edge=Ebin_edge_outcoming

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
        
        DRM = DRMGen(np.array([0.0745, -0.105, 0.0939, 0.987]),
                     np.array([-5.88 * 10 ** 6, -2.08 * 10 ** 6, 2.97 * 10 ** 6]), self._det,
                     self.Ebin_in_edge, mat_type=0, ebin_edge_out=self._Ebin_out_edge)

        # If MPI is used split up the points among the used cores to speed up
        if using_mpi:
            points_per_rank = float(self._Ngrid) / float(size)
            points_lower_index = int(np.floor(points_per_rank * rank))
            points_upper_index = int(np.floor(points_per_rank * (rank + 1)))
            for point in self._points[points_lower_index:points_upper_index]:
                # get the response of every point
                rsp = self._response(point[0], point[1], point[2], DRM)
                responses.append(rsp.matrix.T)

            # Collect all results in rank=0 and broadcast the final array to all ranks in the end
            responses = np.array(responses)
            responses_g = comm.gather(responses, root=0)
            if rank == 0:
                responses_g = np.concatenate(responses_g)

            # broadcast the resulting list to all ranks
            responses = comm.bcast(responses_g, root=0)
                        
        else:
            for point in self._points:
                # get the response of every point
                rsp = self._response(point[0], point[1], point[2], DRM)
                responses.append(rsp.matrix.T)

        self._responses = np.array(responses)

    def _fibonacci_sphere(self, samples=1):
        """
        Calculate equally distributed points on a unit sphere using fibonacci
        :params samples: number of points
        """
        rnd = 1.

        points = []
        offset = 2. / samples
        increment = np.pi * (3. - np.sqrt(5.));

        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2);
            r = np.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % samples) * increment

            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x, y, z])

        return points

from gbmgeometry import GBMFrame
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from gbmbkgpy.utils.progress_bar import progress_bar
from scipy.interpolate import interpolate



class GC_fixed:
    def __init__(self, det_responses, geometry):
        """
        Initialize galactic center (GC) and precalculate count rates attributable to GC.
        
        :params response_object: response_precalculation object
        :params geometry_object: geometry precalculation object
        """
        
        self._detectors = list(det_responses.responses.keys())

        self._rsp = det_responses.responses
        self._geom = geometry

        self._data_type = self._rsp[self._detectors[0]].data_type
        
        self._echans = det_responses.echans
        self._Ntime = self._geom.geometry_times.shape[0]
        self._Ngrid = self._rsp[list(self._rsp.keys())[0]].Ngrid
        
        
        # add function calls here that conduct all calculations
        self._calc_gc_rates()
        
        self._interpolate_gc_rates()
        
    @property
    def responses(self):
        return self._rsp
   

    def get_gc_rates(self, met):
        """
        Returns an array with the predicted count rates for the times for which the geometry 
        was calculated for all energy channels.
        """
        
        gc_rates = self._interp_rate_gc(met)
        

        # The interpolated rate has the dimensions (len(time_bins), 2, len(detectors), len(echans))
        # We want (len(time_bins), len(detectors), len(echans), 2) so we need to swap axes.
        # The 2 is the start stop in the time_bins
        
        gc_rates = np.swapaxes(gc_rates, 1, 2)
        gc_rates = np.swapaxes(gc_rates, 2, 3)

        return gc_rates
    
    
    def _calc_gc_rates(self):
        """
        Returns a dictionary {det: reco_cts} where reco_cts is an 1D-array with the reconstructed count rates 
        (for the specified detector det) in all the reconstructed energy bins specified by the 
        response_precalculation object
        """
        
        folded_flux_gc = np.zeros(
            (self._Ntime, len(self._detectors), len(self._echans))
        )
        
        for det_idx, det in enumerate(self._detectors):
            self._current_det = det
            
            folded_flux_gc[:, det_idx, :] = self._calc_gc_rates_one_det(det_response=self._rsp[det])
            
        self._folded_flux_gc = folded_flux_gc
        
    def _calc_gc_rates_one_det(self, det_response):
        """
        Returns an array of shape (len(Ntime), len(echans)) with the reconstructed count rates 
        (for the specified detector det) in all the reconstructed energy bins specified by the 
        response_precalculation object for all the Ntime times.
        :param det_response: response matrix for detector of interest
        """
        
        reconstr_counts = np.zeros((self._Ntime, len(self._echans))) 
        
        # implement progress bar to monitor progress
        with progress_bar(self._Ntime, 
                          title=f"Calculating reconstructed count rates of galactic center for detector {self._current_det}"
                         ) as p:

            occ_mask = self._calc_earth_occultation_mask(det_response)
            weights = self._calc_gc_weights(occ_mask, det_response)

            # get flux vector (array of integrated spectrum in specified input bins):
            # use trapezoidal rule to integrate spectrum
            # (needs to be adapted if a small number of input bins is used)
            flux_vec = np.zeros(len(det_response.Ebin_in_edge)-1)
            for i in range(len(det_response.Ebin_in_edge)-1):
                x1 = np.array([det_response.Ebin_in_edge[i]])
                x2 = np.array([det_response.Ebin_in_edge[i+1]])
                flux_vec[i] = 0.5 * (x2 - x1) * (self._spectrum(x1) + self._spectrum(x2))
            
            
            for time in range(self._Ntime):
                # transposing is needed due to numpy broadcasting rules
                total_resp_matr = np.sum((det_response.response_array.T * weights[time, :]).T, axis=0)
                reconstr_counts[time, :] = total_resp_matr.T @ flux_vec
                # multiplication with sr_points has already been done in calc_gc_weights function
                
                p.increase()

        return reconstr_counts
    
    def _calc_gc_weights(self, occ_mask, det_response):
        """
        Returns an array of shape (Ntime, Ngrid) that characterizes how well each gridpoint hits the GC
        at the specific time
        """
        
        # get list of GBMFrame objects for all relevant times
        gbm_frames = []
        for time_met in range(len(self._geom.geometry_times)):
            # Get sc pos and quaternions
            q1, q2, q3, q4 = self._geom.quaternion[time_met, :]
            scx, scy, scz = self._geom.sc_pos[time_met, :]

            # Define GBMFrame for the given parameters
            gbm_frame = GBMFrame(quaternion_1=q1,
                                quaternion_2=q2,
                                quaternion_3=q3,
                                quaternion_4=q4,
                                sc_pos_X=scx,
                                sc_pos_Y=scy,
                                sc_pos_Z=scz,)

            # append transformed frame and Skycoord object to lists
            gbm_frames.append(gbm_frame)
        
        
        # get cartesian coordinates of gridpoints
        x_val = det_response.points[..., 0]
        y_val = det_response.points[..., 1]
        z_val = det_response.points[..., 2]
        cart_coords_grid = list(zip(x_val, y_val, z_val))
        
        weights = np.zeros((self._Ntime, self._Ngrid))

        for time in range(self._Ntime):
            # get gc in gbm frame and the vectors for (time-dependent) "longitude and latitude axes"
            gc_center = np.array(SkyCoord(l=0*u.degree, b=0*u.degree, frame = 'galactic').transform_to(gbm_frames[time]).cartesian.get_xyz())

            # Points on the unit sphere in gbm_frame
            gc_l = np.array(SkyCoord(l=45*u.degree, b=0*u.degree, frame = 'galactic').transform_to(gbm_frames[time]).cartesian.get_xyz())
            gc_b = np.array(SkyCoord(l=0*u.degree, b=45*u.degree, frame = 'galactic').transform_to(gbm_frames[time]).cartesian.get_xyz())
            # correct the length to get a right angle between the connection gbm_frame_center->gc_center
            # and the latitude vector
            # simple geometry with c=sqrt(b^2+a^2), with b=a=1.
            gc_l *= np.sqrt(2)
            gc_b *= np.sqrt(2)

            # get the new axis vectors
            lon_ax = gc_l-gc_center
            lat_ax = gc_b-gc_center
            y_ax = gc_center # vector pointing from center of gbm frame to galactic center

            # construct rotation matrix
            rot_mat = np.zeros((3,3))
            rot_mat[:,0]=lon_ax
            rot_mat[:,1]=y_ax
            rot_mat[:,2]=lat_ax

            points_transformed = np.zeros((len(cart_coords_grid), 3))
            # Transform all points
            points_transformed = np.dot(cart_coords_grid, rot_mat)

            # get l and b in the new coord system
            # set up in a way that b=l=0 points towards y-axis
            b = np.arcsin(points_transformed[:,2])
            l = np.arctan2(points_transformed[:,0],points_transformed[:,1])

            # get the weights
            weight = self._lorentzian(l, b)

            # populate weights array with weight
            weights[time, ...] = weight

        # set weight of occulted gridpoints to zero
        weights[occ_mask] = 0
        
        # every gridpoint needs to be multiplied with the sr_points factor which is the
        # area of the unit sphere covered by every point
        resp_grid_points = det_response.points        
        sr_points = 4 * np.pi / len(resp_grid_points)        
        weights = weights * sr_points
        
        return weights

    
    def _calc_earth_occultation_mask(self, det_response):
        """
        create a mask with True when Earth is in FOV and False when it's not;
        the code is mostly copied from albedo_cgb.py
        """

        earth_radius = 6371.0
        fermi_radius = np.sqrt(np.sum(self._geom.sc_pos ** 2, axis=1))
        horizon_angle = 90 - np.rad2deg(np.arccos(earth_radius / fermi_radius))

        min_vis = np.deg2rad(horizon_angle)
        
        resp_grid_points = det_response.points

        # Calculate the normalization of the spacecraft position vectors
        earth_position_cart_norm = np.sqrt(
            np.sum(self._geom.earth_position_cart * self._geom.earth_position_cart, axis=1)
        ).reshape((len(self._geom.earth_position_cart), 1))

        # Calculate the normalization of the grid points of the response precalculation
        resp_grid_points_norm = np.sqrt(np.sum(resp_grid_points * resp_grid_points, axis=1)
        ).reshape((len(resp_grid_points), 1))

        tmp = np.clip(np.dot(self._geom.earth_position_cart / earth_position_cart_norm,
                resp_grid_points.T / resp_grid_points_norm.T), -1, 1)

        # Calculate separation angle between spacecraft and earth horizon
        ang_sep = np.arccos(tmp)

        # Create a mask with True when the Earth is in the FOV and False if not
        earth_occultion_idx = np.less(ang_sep.T, min_vis).T

        return earth_occultion_idx
        
    def _spectrum(self, E, c_tot = 1): #x nparray in keV
        """
        Takes energies [keV] as numpy array as input and returns an array of the 
        photon flux at those energies.
        :param c_tot: Normalization that can be fitted later on. Standard value of 1 reproduces bouchet paper
        """
        
        # define three components of energy spectrum according to bouchet 2011:
        # interstellar emissions:
        pl_ind_ie = 1.45
        c_ie = 1.1e-04 * 100 ** pl_ind_ie
        flux_ie = c_ie * E**(-pl_ind_ie)
        
        # magn white dwarfs, relevant for E<50keV:
        exp_cutoff_mwd = 8
        c_mwd = 2e-04 * np.exp(50/exp_cutoff_mwd)
        flux_mwd = c_mwd * np.exp(-E/exp_cutoff_mwd)
        
        # point sources (dominant contribution):
        pl_ind_ps = 2.9
        c_ps = 4.0e-04 * 100 ** pl_ind_ps
        flux_ps = c_ps * E**(-pl_ind_ps)
        
        #scales whole spectrum, can be fitted later on
        return c_tot * (flux_ie + flux_mwd + flux_ps)
        
    
    def _lorentzian(self, l, b):
        # define lorentzian functions for GC according to M. Türler in Integral paper
        # FWHM_l is 21 degrees
        FWHM_l = np.pi * 7 / 60 #rad
        l_0 = 0
        gamma_l = FWHM_l / 2
        
        # FWHM_b is 1.2 degrees
        FWHM_b = np.pi / 150 #rad
        # b_0 is -0.15 degrees
        b_0 = - np.pi / 1200 #rad
        gamma_b = FWHM_b / 2
        
        # Normalization so that the integral of the resulting function over the unit sphere equals 1
        c = 1 / 0.0180403
        
        return c * (gamma_l**2 / ((l - l_0)**2 + gamma_l**2)) * (gamma_b**2 / ((b - b_0)**2 + gamma_b**2))
    

    def _interpolate_gc_rates(self):
        self._interp_rate_gc = interpolate.interp1d(
            self._geom.geometry_times, self._folded_flux_gc, axis=0)
        

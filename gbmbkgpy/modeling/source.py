from scipy import integrate
import numpy as np

from astromodels import Constant

from gbmbkgpy.modeling.new_astromodels import fix_all_params


class Source:
    def __init__(self, name, fit_model, spectral_model=None):
        self._name = name
        self._fit_model = fit_model
        self._spectral_model = spectral_model

    def __call__(self):
        assert hasattr(self, "_time_bins"),\
            "You first have to set the time-bins before evalulation"
        return self._evaluate()

    def set_time_bins(self, time_bins):
        self._precalculation(time_bins)

    def _precalculation(self, time_bins):
        self._time_bins = time_bins

    def get_counts(self, bin_mask=None, time_bins=None):
        """
        Calls the evaluation of the source to get the counts per bin. Uses a bin_mask to exclude some bins if needed.
        No need of integration here anymore! This is done in the function class of the sources!
        :param time_bins:
        :param echan:
        :param bin_mask:
        :return:
        """
        if time_bins is not None:
            # special time bins input
            return self._evaluate_at_time_bins(time_bins)

        if bin_mask is not None:
            return self._evaluate()[bin_mask]

        return self._evaluate()

    def _evaluate(self):
        # evaluate at the default time bins
        raise NotImplementedError("Has to be implemented in sub-class")

    def _evaluate_at_time_bins(self, time_bins):
        # evaluate at given time bins
        raise NotImplementedError("Has to be implemented in sub-class")

    def __repr__(self):

        info = (f"### {self.name} ### \n"
                f"{self.fit_model} \n"
                f"{self.spectral_model}")

        return info

    def change_name(self, new_name):
        self._name = new_name

    @property
    def name(self):
        return self._name

    @property
    def fit_model(self):
        return self._fit_model

    @property
    def spectral_model(self):
        return self._spectral_model

    @property
    def parameters(self):
        return self.fit_model.free_parameters


class SAASource(Source):

    def __init__(self, name, time, model):
        assert model.name in ["Line",
                              "Exponential_cutoff",
                              "AstromodelFunctionVector"]

        if model.name != "AstromodelFunctionVector":

            self._model_vec = False
            if model.name == "Line":

                self._model_type = 1
            else:

                self._model_type = 2
        else:

            self._model_vec = True
            if model.vector[0].name == "Line":

                self._model_type = 1
            elif model.vector[0].name == "Exponential_cutoff":

                self._model_type = 2
            else:

                raise AssertionError("Base function must be Line or Exponential_cutoff")

        self._t0 = time

        super().__init__(name, model)

    def _precalculation(self, time_bins):
        """
        precalulate which time bins are after the SAA exit.
        :return:
        """
        self._idx_start = time_bins[:, 0] < self._t0

        self._tstart = time_bins[:, 0][~self._idx_start]
        self._tstop = time_bins[:, 1][~self._idx_start]

        if self._model_vec:

            self._out = np.zeros((len(time_bins),
                                  self.fit_model.num_x,
                                  ))

        else:

            self._out = np.zeros_like(time_bins[:, 0])

        super()._precalculation(time_bins)

    def _evaluate(self):
        """
        Mult base array with norm
        """
        if self._model_type == 1:

            raise NotImplementedError("Not yet")

        if self._model_type == 2:

            # analyic integral solution
            if self._model_vec:

                xc = self.fit_model.xc[np.newaxis, ...]
                self._out[~self._idx_start] = xc*(self._fit_model(self._tstart-self._t0) -
                                                  self._fit_model(self._tstop-self._t0))

            else:

                xc = self.fit_model.xc.value
                self._out[~self._idx_start] = xc*(self._fit_model(self._tstart-self._t0) -
                                                  self._fit_model(self._tstop-self._t0))

        return self._out

    def _evaluate_at_time_bins(self, time_bins):
        # Stupid code duplication
        idx_start = time_bins[:, 0] < self._t0

        tstart = time_bins[:, 0][~idx_start]
        tstop = time_bins[:, 1][~idx_start]

        if self._model_vec:

            out = np.zeros((len(time_bins),
                            self.fit_model.num_x,
                            ))

        else:

            out = np.zeros_like(time_bins[:, 0])

        if self._model_type == 1:

            raise NotImplementedError("Not yet")

        if self._model_type == 2:

            # analyic integral solution
            if self._model_vec:

                xc = self.fit_model.xc[np.newaxis, ...]
                out[~idx_start] = xc*(self._fit_model(tstart-self._t0) -
                                      self._fit_model(tstop-self._t0))

            else:

                xc = self.fit_model.xc.value
                out[~idx_start] = xc*(self._fit_model(tstart-self._t0) -
                                      self._fit_model(tstop-self._t0))

        return out

class NormOnlySource(Source):

    def __init__(self,
                 name,
                 interp1d_rate_base_array,
                 const_model=None,
                 spectral_model=None
                 ):
        """
        :param const_model: A model with constants. Could be one constant
        if the echans are connected or num_echans constants if the echans
        are independent
        """

        self._interp1d_rate_base_array = interp1d_rate_base_array

        super().__init__(name, const_model, spectral_model)

    def _precalculation(self, time_bins):
        #self._base_array = self._integrate_base_array(time_bins)
        self._integrate_base_array(time_bins)
        super()._precalculation(time_bins)

    def _integrate_base_array(self, time_bins):
        """
        Integrate the base rate array over the time bins
        We can do this once here, because only the normalization is fitted
        """
        rates = self._interp1d_rate_base_array(time_bins)
        if len(rates.shape) == 3:
            time_bins = np.tile(time_bins, (rates.shape[2], 1, 1)).T
            time_bins = np.swapaxes(time_bins, 0, 1)

        self._base_array = np.trapz(rates, time_bins, axis=1)

        if len(self._base_array.shape) == 1:

            if self.fit_model.name == "AstromodelFunctionVector":

                self._base_array = np.tile(self._base_array,
                                           (self.fit_model.num_x, 1)).T

            else:

                self._base_array = np.tile(self._base_array, (1, 1))

    def _evaluate(self):
        """
        Mult base array with norm
        """
        # eval model at dummy value (is a constant model)
        return self._fit_model(1)*self._base_array

    def _evaluate_at_time_bins(self, time_bins):
        rates = self._interp1d_rate_base_array(time_bins)
        if len(rates.shape) == 3:
            time_bins = np.tile(time_bins, (rates.shape[2], 1, 1)).T
            time_bins = np.swapaxes(time_bins, 0, 1)

        base_array = np.trapz(rates, time_bins, axis=1)

        if len(self._base_array.shape) == 1:

            if self.fit_model.name == "AstromodelFunctionVector":

                base_array = np.tile(self._base_array,
                                     (self.fit_model.num_x, 1)).T

            else:

                base_array = np.tile(base_array, (1, 1))
        #base_array = self._integrate_base_array(time_bins)
        return self._fit_model(1)*base_array


class PhotonSourceFixed(NormOnlySource):

    def __init__(self,
                 name,
                 astro_model,
                 rsp_obj):
        """
        :param name: Name of this source
        :param astro_model: Astromodel Function for spectrum
        :param response_array: Array with response for different times
        """

        # fix all the params
        fix_all_params(astro_model)

        self._monte_carlo_energies = rsp_obj.Ebins_in_edge
        response_interpolation = rsp_obj.interp_effective_response
        self._num_ebins_out = rsp_obj.num_ebins_out

        interp1d_rate_base_array = self._construct_interp1d_rate_base_array(response_interpolation,
                                                                            astro_model,
                                                                            1.0)

        const = Constant()
        const.k.value = 1.0

        super().__init__(name, interp1d_rate_base_array,
                         const, astro_model) # const*astro_model

    def _construct_interp1d_rate_base_array(self, response_interpolation,
                                            model, norm_val):
        spec = 1/norm_val*model(self._monte_carlo_energies)

        # trapz integrate
        ee1 = self._monte_carlo_energies[:-1]
        ee2 = self._monte_carlo_energies[1:]
        binned_spec = np.trapz(np.array([spec[:-1], spec[1:]]).T,
                               np.array([ee1, ee2]).T)

        def interp1d_rate_base_array(time):
            return np.dot(binned_spec, response_interpolation(time))

        return interp1d_rate_base_array


class PhotonSourceFree(Source):

    def __init__(self,
                 name,
                 astro_model,
                 rsp_obj):

        assert len(astro_model.free_parameters) > 1,\
            "There should be more than one free parameter if the spectrum shape is free"

        self._monte_carlo_energies = rsp_obj.Ebins_in_edge
        self._response_interpolation = rsp_obj.interp_effective_response
        self._num_ebins_out = rsp_obj.num_ebins_out

        super().__init__(name, astro_model, astro_model)

    def _precalculation(self, time_bins):
        self._response_array = self._response_interpolation(time_bins)
        self._tile_time_bins = np.tile(time_bins,
                                       (self._num_ebins_out, 1, 1)).T
        self._tile_time_bins = np.swapaxes(self._tile_time_bins, 0, 1)

        super()._precalculation(time_bins)

    def _evaluate(self):

        # get flux at input edges
        spec = self._fit_model(self._monte_carlo_energies)

        # trapz integrate
        ee1 = self._monte_carlo_energies[:-1]
        ee2 = self._monte_carlo_energies[1:]
        binned_spec = np.trapz(np.array([spec[:-1], spec[1:]]).T,
                               np.array([ee1, ee2]).T)
        # fold with all the responses
        rates = np.dot(binned_spec, self._response_array)
        # integrate over the time bins
        return np.trapz(rates, self._tile_time_bins, axis=1)

    def _evaluate_at_time_bins(self, time_bins):
        response_array = self._response_interpolation(time_bins)
        tile_time_bins = np.tile(time_bins,
                                 (self._num_ebins_out, 1, 1)).T
        tile_time_bins = np.swapaxes(tile_time_bins, 0, 1)

        # get flux at input edges
        spec = self._fit_model(self._monte_carlo_energies)

        # trapz integrate
        ee1 = self._monte_carlo_energies[:-1]
        ee2 = self._monte_carlo_energies[1:]
        binned_spec = np.trapz(np.array([spec[:-1], spec[1:]]).T,
                               np.array([ee1, ee2]).T)
        # fold with all the responses
        rates = np.dot(binned_spec, response_array)
        # integrate over the time bins
        return np.trapz(rates, tile_time_bins, axis=1)

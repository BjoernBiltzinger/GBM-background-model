import numpy as np
import os, sys
import json
import collections
import math
import matplotlib.pyplot as plt
from datetime import datetime

from astromodels.functions.priors import (
    Uniform_prior,
    Log_uniform_prior,
    Log_normal,
    Truncated_gaussian,
    Gaussian,
)
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.statistics.stats_tools import compute_covariance_matrix

try:
    from mininest.integrator import ReactiveNestedSampler
    import pymultinest  # This is currently used for analysis_result

except:

    has_mininest = False

else:

    has_mininest = True

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


class MiniNestFit(object):
    def __init__(self, likelihood, parameters):

        self._likelihood = likelihood
        self.parameters = parameters

        self._det = self._likelihood._data._det
        self._echan_list = self._likelihood._echan_list
        self._n_dim = len(self._likelihood._free_parameters)
        self._day_list = self._likelihood._data.day

        self._day = ""
        for d in self._day_list:
            self._day = (
                d  # TODO change this; set maximum characters for multinestpath higher
            )

        self.cov_matrix = None
        self.best_fit_values = None

        if using_mpi:
            if rank == 0:
                self.output_dir = self._create_output_dir()
            else:
                self.output_dir = None

            self.output_dir = comm.bcast(self.output_dir, root=0)
        else:
            self.output_dir = self._create_output_dir()

        # We need to wrap the function, because multinest maximizes instead of minimizing
        def func_wrapper(values, ndim, nparams):
            # values is a wrapped C class. Extract from it the values in a python list
            values_list = [values[i] for i in range(ndim)]
            return self._likelihood(values_list) * (-1)

        # First build a uniform prior for each parameters
        self._build_priors()

        # declare local likelihood_wrapper object:
        self._loglike = func_wrapper

    @property
    def output_directory(self):
        return self.output_dir

    @property
    def samples(self):

        return self._samples

    def minimize_mininest(
        self,
        loglike=None,
        prior=None,
        n_dim=None,
        min_num_live_points=400,
        chain_name=None,
        resume=False,
        quiet=False,
        verbose=False,
        **kwargs
    ):

        assert has_mininest, "You need to have mininest installed to use this function"

        if loglike is None:
            loglike = self._loglike

        if prior is None:
            prior = self._construct_mininest_prior()

        if n_dim is None:
            n_dim = self._n_dim

        if chain_name is None:
            chain_name = self.output_dir
        # Run PyMultiNest

        min_ess = kwargs.pop("min_ess", 400)
        frac_remain = kwargs.pop("frac_remain", 0.01)
        dlogz = kwargs.pop("dlogz", 0.5)
        max_iter = kwargs.pop("max_iter", 0.0)
        dKL = kwargs.pop("dKL", 0.5)

        if not verbose:
            kwargs["viz_callback"] = False

        sampler = ReactiveNestedSampler(
            loglike=loglike,
            transform=prior,
            log_dir=chain_name,
            min_num_live_points=min_num_live_points,
            append_run_num=not resume,
            show_status=verbose,
            param_names=self.parameters.keys(),
            draw_multiple=False,
            **kwargs
        )

        sampler.run(
            dlogz=dlogz,
            max_iters=max_iter if max_iter > 0 else None,
            min_ess=min_ess,
            frac_remain=frac_remain,
            dKL=dKL,
        )

        # Store the sample for further use (if needed)
        self._sampler = sampler

        # if using mpi only analyze in rank=0
        if using_mpi:
            if rank == 0:
                _, _ = self.analyze_result()
        else:
            _, _ = self.analyze_result()

    def _construct_mininest_prior(self):
        """
        pymultinest becomes confused with the self pointer. We therefore ceate callbacks
        that pymultinest can understand.

        Here, we construct the prior.
        """
        ndim = len(self.parameters.items())

        def prior(params):

            out = np.zeros((len(params), ndim))

            for i, (parameter_name, parameter) in enumerate(self.parameters.items()):

                try:

                    out[:, i] = parameter.prior.from_unit_cube(params[:, i])

                except AttributeError:

                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with multinest" % parameter_name
                    )
            return out

        return prior

    def analyze_result(self, output_dir=None):
        """
        Analyze result of multinest fit, when a output directory of an old fit is passed the params.json
        will not be overwritten.
        :param output_dir:
        :return:
        """
        if output_dir is None:
            output_dir = self.output_dir

            # Save parameter names
            param_index = []
            for i, parameter in enumerate(self._likelihood._parameters.values()):
                param_index.append(parameter.name)

            self._param_names = param_index
            json.dump(self._param_names, open(output_dir + "params.json", "w"))

        ## Use PyMULTINEST analyzer to gather parameter info
        multinest_analyzer = pymultinest.analyse.Analyzer(
            n_params=self._n_dim, outputfiles_basename=output_dir
        )

        # Get the function value from the chain
        func_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

        # Get the samples from the sampler

        _raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]
        # print(_raw_samples)
        # print(_raw_samples[0])

        # Find the minimum of the function (i.e. the maximum of func_wrapper)

        idx = func_values.argmax()

        self.best_fit_values = _raw_samples[idx]

        self.minimum = func_values[idx] * (-1)
        self._samples = _raw_samples
        self.multinest_data = multinest_analyzer.get_data()

        # set parameters to best fit values
        self._likelihood.set_free_parameters(self.best_fit_values)
        return self.best_fit_values, self.minimum

    def _build_priors(self):
        # First build a uniform prior for each parameters
        self._param_priors = collections.OrderedDict()

        for parameter_name in self.parameters:

            min_value, max_value, mu, sigma = self.parameters[
                parameter_name
            ].get_prior_parameter
            prior_type = self.parameters[parameter_name].prior

            assert min_value is not None, (
                "Minimum value of parameter %s is None. In order to use the Multinest "
                "minimizer you need to define proper bounds for each "
                "free parameter" % parameter_name
            )

            assert max_value is not None, (
                "Maximum value of parameter %s is None. In order to use the Multinest "
                "minimizer you need to define proper bounds for each "
                "free parameter" % parameter_name
            )

            # Compute the difference in order of magnitudes between minimum and maximum

            if prior_type is not None:
                if prior_type == "uniform":
                    self._param_priors[parameter_name] = Uniform_prior(
                        lower_bound=min_value, upper_bound=max_value
                    )

                elif prior_type == "log_uniform":
                    self._param_priors[parameter_name] = Log_uniform_prior(
                        lower_bound=min_value, upper_bound=max_value
                    )
                elif prior_type == "gaussian":
                    self._param_priors[parameter_name] = Gaussian(mu=mu, sigma=sigma)
                elif prior_type == "truncated_gaussian":
                    self._param_priors[parameter_name] = Truncated_gaussian(
                        mu=mu, sigma=sigma, lower_bound=min_value, upper_bound=max_value
                    )
                elif prior_type == "log_normal":
                    self._param_priors[parameter_name] = Log_normal(mu=mu, sigma=sigma)
                else:
                    raise TypeError(
                        "Unknown prior! Please choose uniform or log_uniform prior"
                    )
            else:
                if min_value > 0:

                    orders_of_magnitude_span = math.log10(max_value) - math.log10(
                        min_value
                    )

                    if orders_of_magnitude_span > 2:

                        # Use a Log-uniform prior
                        self._param_priors[parameter_name] = Log_uniform_prior(
                            lower_bound=min_value, upper_bound=max_value
                        )

                    else:

                        # Use a uniform prior
                        self._param_priors[parameter_name] = Uniform_prior(
                            lower_bound=min_value, upper_bound=max_value
                        )

                else:

                    # Can only use a uniform prior
                    self._param_priors[parameter_name] = Uniform_prior(
                        lower_bound=min_value, upper_bound=max_value
                    )

    def _create_output_dir(self):
        current_time = datetime.now()
        fits_path = os.path.join(get_path_of_external_data_dir(), "fits/")
        multinest_out_dir = os.path.join(
            get_path_of_external_data_dir(), "fits", "mn_out/"
        )
        if len(self._echan_list) == 1:
            date_det_echan_dir = "{}_{}_{:d}/".format(
                self._day, self._det, self._echan_list[0]
            )
        else:
            date_det_echan_dir = "{}_{}_{:d}_ech_{:d}_to_{:d}/".format(
                self._day,
                self._det,
                len(self._echan_list),
                self._echan_list[0],
                self._echan_list[-1],
            )
        time_dir = current_time.strftime("%m-%d_%H-%M") + "/"

        output_dir = os.path.join(multinest_out_dir, date_det_echan_dir, time_dir)

        if not os.access(output_dir, os.F_OK):

            # create directory if it doesn't exist
            if not os.access(fits_path, os.F_OK):
                print("Making New Directory")
                os.mkdir(fits_path)

            # Create multinest_out if not existend
            if not os.access(multinest_out_dir, os.F_OK):
                print("Making New Directory")
                os.mkdir(multinest_out_dir)

            # Create date_det_echan_dir if not existend
            if not os.access(
                os.path.join(multinest_out_dir, date_det_echan_dir), os.F_OK
            ):
                print("Making New Directory")
                os.mkdir(os.path.join(multinest_out_dir, date_det_echan_dir))

            print("Making New Directory")
            os.mkdir(output_dir)

        return output_dir

    def comp_covariance_matrix(self):
        self.cov_matrix = compute_covariance_matrix(
            self._likelihood.cov_call, self.best_fit_values
        )

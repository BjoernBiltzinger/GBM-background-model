import numpy as np
import os
import sys
import json
import random
import collections
import math
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.statistics.stats_tools import compute_covariance_matrix
from astromodels.functions.priors import (
    Uniform_prior,
    Log_uniform_prior,
    Log_normal,
    Truncated_gaussian,
    Gaussian,
)

try:

    import pymultinest

except:

    has_pymultinest = False

else:

    has_pymultinest = True

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


class MultiNestFit(object):
    def __init__(self, likelihood, parameters):

        self._likelihood = likelihood
        self.parameters = parameters
        self._n_dim = len(self.parameters)

        self.cov_matrix = None
        self._best_fit_values = None
        self._sampler = None
        self._param_names = None
        self._minimum = None
        self._samples = None
        self.multinest_data = None

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
    def output_dir(self):
        return self._output_dir

    @property
    def samples(self):

        return self._samples

    @property
    def best_fit_values(self):
        return self._best_fit_values

    @property
    def minimum(self):
        return self._minimum

    def minimize_multinest(
        self,
        loglike=None,
        prior=None,
        n_dim=None,
        n_live_points=400,
        const_efficiency_mode=False,
        output_dir=None,
    ):

        assert (
            has_pymultinest
        ), "You need to have pymultinest installed to use this function"

        if loglike is None:
            loglike = self._loglike

        if prior is None:
            prior = self._construct_multinest_prior()

        if n_dim is None:
            n_dim = self._n_dim

        self._output_dir, tmp_output_dir = self._create_output_dir(output_dir)

        # Run PyMultiNest
        sampler = pymultinest.run(
            loglike,
            prior,
            n_dim,
            n_dim,
            n_live_points=n_live_points,
            outputfiles_basename=tmp_output_dir,
            multimodal=True,  # True was default
            resume=True,
            verbose=True,  # False was default
            importance_nested_sampling=False,
            const_efficiency_mode=const_efficiency_mode,
        )

        # Store the sample for further use (if needed)
        self._sampler = sampler

        # if using mpi only analyze in rank=0
        if using_mpi:

            if rank == 0:

                # If we used a temporary output dir then move it to the final destination
                if tmp_output_dir != self._output_dir:

                    shutil.move(tmp_output_dir, self._output_dir)

                self.analyze_result()

            # Cast the results to all ranks
            self._best_fit_values = comm.bcast(self._best_fit_values, root=0)
            self._minimum = comm.bcast(self._minimum, root=0)

        else:

            # If we used a temporary output dir then move it to the final destination

            if tmp_output_dir != self._output_dir:

                shutil.move(tmp_output_dir, self._output_dir)

            self.analyze_result()

    def _construct_multinest_prior(self):
        """
        pymultinest becomes confused with the self pointer. We therefore ceate callbacks
        that pymultinest can understand.

        Here, we construct the prior.
        """

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self.parameters.items()):

                try:

                    params[i] = self._param_priors[parameter_name].from_unit_cube(
                        params[i]
                    )

                except AttributeError:

                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with multinest" % parameter_name
                    )

                    # Give a test run to the prior to check that it is working. If it crashes while multinest is going

        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self.parameters.items())

        _ = prior([0.5] * n_dim, n_dim, [])

        return prior

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

    def analyze_result(self, output_dir=None):
        """
        Analyze result of multinest fit, when a output directory of an old fit is passed the params.json
        will not be overwritten.
        :param output_dir:
        :return:
        """
        if output_dir is None:
            output_dir = self._output_dir

        # Save parameter names
        self._param_names = [parameter.name for parameter in self.parameters.values()]

        if using_mpi:

            if rank == 0:

                json.dump(self._param_names, open(output_dir + "params.json", "w"))

        else:

            json.dump(self._param_names, open(output_dir + "params.json", "w"))

        ## Use PyMULTINEST analyzer to gather parameter info
        multinest_analyzer = pymultinest.analyse.Analyzer(
            n_params=self._n_dim, outputfiles_basename=output_dir
        )

        # Get the function value from the chain
        func_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

        # Get the samples from the sampler
        _raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]

        # Find the minimum of the function (i.e. the maximum of func_wrapper)
        idx = func_values.argmax()

        best_fit_values = _raw_samples[idx]

        minimum = func_values[idx] * (-1)

        self._samples = _raw_samples
        self.multinest_data = multinest_analyzer.get_data()

        # set parameters to best fit values
        self._likelihood.set_free_parameters(best_fit_values)

        self._best_fit_values = best_fit_values
        self._minimum = minimum

        print('The MAP of the model parameters:')
        print(dict(zip(self._param_names, best_fit_values)))

        return best_fit_values, minimum

    def comp_covariance_matrix(self):
        if rank == 0:
            self.cov_matrix = compute_covariance_matrix(
                self._likelihood.cov_call, self.best_fit_values
            )

        if using_mpi:
            self.cov_matrix = comm.bcast(self.cov_matrix, root=0)

    def _create_output_dir(self, output_dir):

        if output_dir is None:

            output_dir = os.path.join(
                get_path_of_external_data_dir(),
                "fits",
                "mn_out",
                datetime.now().strftime("%m-%d_%H-%M") + "/",
            )

        # If the output path is to long (MultiNest only supports 100 chars)
        # we will us a random directory name and move it when MultiNest finished.

        if len(output_dir) > 72:

            tmp_output_dir = os.path.join(
                get_path_of_external_data_dir(),
                "fits",
                "mn_out",
                str(random.getrandbits(16)) + "/",
            )

        else:

            tmp_output_dir = output_dir

        # Create output dir for multinest if not existing
        if using_mpi:

            if rank == 0:

                if not os.access(tmp_output_dir, os.F_OK):
                    print("Making New Directory")
                    os.makedirs(tmp_output_dir)

        else:

            if not os.access(tmp_output_dir, os.F_OK):
                print("Making New Directory")
                os.makedirs(tmp_output_dir)

        return output_dir, tmp_output_dir

    def create_corner_plot(self):

        if using_mpi:

            if rank == 0:

                create_plot = True

            else:

                create_plot = False

        else:

            create_plot = True

        if create_plot:

            safe_param_names = [
                name.replace("_", " ") for name in list(self.parameters.keys())
            ]

            if len(safe_param_names) > 1:

                chain = np.loadtxt(
                    os.path.join(self._output_dir, "post_equal_weights.dat"), ndmin=2
                )

                c2 = ChainConsumer()

                c2.add_chain(chain[:, :-1], parameters=safe_param_names).configure(
                    plot_hists=False,
                    contour_labels="sigma",
                    colors="#cd5c5c",
                    flip=False,
                    max_ticks=3,
                )

                c2.plotter.plot(filename=os.path.join(self._output_dir, "corner.pdf"))

            else:
                print('Your model only has one paramter, we cannot make a cornerplot for this.')

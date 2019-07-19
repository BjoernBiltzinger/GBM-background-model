
from astromodels.functions.priors import Uniform_prior, Log_uniform_prior
import pymultinest
import numpy as np
import os, sys
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
import json
import collections
import math
from numpy import exp, log
import matplotlib.pyplot as plt
import shutil
from datetime import datetime


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


class MultiNestFit(object):
    def __init__(self, likelihood, parameters):

        self._likelihood = likelihood

        self._day = self._likelihood._data._day
        self._det = self._likelihood._data._det
        
        self._echan_list = self._likelihood._echan_list
        self.parameters = parameters

        self._n_dim = len(self._likelihood._free_parameters)
        
        if using_mpi:
            if rank==0:
                current_time = datetime.now()
                fits_path = os.path.join(get_path_of_external_data_dir(), 'fits/')
                multinest_out_dir = os.path.join(get_path_of_external_data_dir(), 'fits', 'mn_out/')
                if len(self._echan_list) == 1:
                    date_det_echan_dir = '{}_{}_{:d}/'.format(self._day, self._det, self._echan_list[0])
                else:
                    date_det_echan_dir = '{}_{}_{:d}_ech_{:d}_to_{:d}/'.format(self._day, self._det,
                                                                               len(self._echan_list),
                                                                               self._echan_list[0],
                                                                               self._echan_list[-1])
                time_dir = 'fit_' + current_time.strftime("%m-%d_%H-%M") + '/'

                self.output_dir = os.path.join(multinest_out_dir, date_det_echan_dir, time_dir)

                if not os.access(self.output_dir, os.F_OK):

                    # create directory if it doesn't exist
                    if not os.access(fits_path, os.F_OK):
                        print("Making New Directory")
                        os.mkdir(fits_path)

                    # Create multinest_out if not existend
                    if not os.access(multinest_out_dir, os.F_OK):
                        print("Making New Directory")
                        os.mkdir(multinest_out_dir)

                    # Create date_det_echan_dir if not existend
                    if not os.access(os.path.join(multinest_out_dir, date_det_echan_dir), os.F_OK):
                        print("Making New Directory")
                        os.mkdir(os.path.join(multinest_out_dir, date_det_echan_dir))

                    print("Making New Directory")
                    os.mkdir(self.output_dir)
            else:
                self.output_dir=None

            self.output_dir = comm.bcast(self.output_dir, root=0)
        else:
            current_time = datetime.now()
            fits_path = os.path.join(get_path_of_external_data_dir(), 'fits/')
            multinest_out_dir = os.path.join(get_path_of_external_data_dir(), 'fits', 'mn_out/')
            if len(self._echan_list) == 1:
                date_det_echan_dir = '{}_{}_{:d}/'.format(self._day, self._det, self._echan_list[0])
            else:
                date_det_echan_dir = '{}_{}_{:d}_ech_{:d}_to_{:d}/'.format(self._day, self._det,
                                                                                   len(self._echan_list),
                                                                                   self._echan_list[0],
                                                                                   self._echan_list[-1])
            time_dir = 'fit_' + current_time.strftime("%m-%d_%H-%M") + '/'

            self.output_dir = os.path.join(multinest_out_dir, date_det_echan_dir, time_dir)

            if not os.access(self.output_dir, os.F_OK):

                # create directory if it doesn't exist
                if not os.access(fits_path, os.F_OK):
                    print("Making New Directory")
                    os.mkdir(fits_path)

                # Create multinest_out if not existend
                if not os.access(multinest_out_dir, os.F_OK):
                    print("Making New Directory")
                    os.mkdir(multinest_out_dir)

                # Create date_det_echan_dir if not existend
                if not os.access(os.path.join(multinest_out_dir, date_det_echan_dir), os.F_OK):
                    print("Making New Directory")
                    os.mkdir(os.path.join(multinest_out_dir, date_det_echan_dir))

                print("Making New Directory")
                os.mkdir(self.output_dir)

        # We need to wrap the function, because multinest maximizes instead of minimizing
        def func_wrapper(values, ndim, nparams):
            # values is a wrapped C class. Extract from it the values in a python list
            values_list = [values[i] for i in range(ndim)]
            return self._likelihood(values_list) * (-1)

        # First build a uniform prior for each parameters
        self._param_priors = collections.OrderedDict()

        for parameter_name in self.parameters:

            min_value, max_value = self.parameters[parameter_name].bounds
            prior_type = self.parameters[parameter_name].prior

            assert min_value is not None, "Minimum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            assert max_value is not None, "Maximum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            # Compute the difference in order of magnitudes between minimum and maximum

            if prior_type is not None:
                if prior_type == 'uniform':
                    self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

                elif prior_type == 'log_uniform':
                    self._param_priors[parameter_name] = Log_uniform_prior(lower_bound=min_value, upper_bound=max_value)
                else:
                    raise TypeError('Unknown prior! Please choose uniform or log_uniform prior')
            else:
                if min_value > 0:

                    orders_of_magnitude_span = math.log10(max_value) - math.log10(min_value)

                    if orders_of_magnitude_span > 2:

                        # Use a Log-uniform prior
                        self._param_priors[parameter_name] = Log_uniform_prior(lower_bound=min_value, upper_bound=max_value)

                    else:

                        # Use a uniform prior
                        self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

                else:

                    # Can only use a uniform prior
                    self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self.parameters.items()):

                try:

                    params[i] = self._param_priors[parameter_name].from_unit_cube(params[i])

                except AttributeError:

                    raise RuntimeError("The prior you are trying to use for parameter %s is "
                                       "not compatible with multinest" % parameter_name)

        # Give a test run to the prior to check that it is working. If it crashes while multinest is going
        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self.parameters)

        _ = prior([0.5] * n_dim, n_dim, [])

        # declare local likelihood_wrapper object:
        self._loglike = func_wrapper
        self._prior = prior

    @property
    def output_directory(self):
        return self.output_dir

    def minimize(self, loglike=None, prior=None, n_dim=None, n_live_points=400):

        if loglike is None:
            loglike = self._loglike

        if prior is None:
            prior = self._prior

        if n_dim is None:
            n_dim = self._n_dim

        # Run PyMultiNest

        sampler = pymultinest.run(loglike,
                                  prior,
                                  n_dim,
                                  n_dim,
                                  n_live_points=n_live_points,
                                  outputfiles_basename=self.output_dir,
                                  multimodal=True,#True was default
                                  resume=True,
                                  verbose=True,#False was default
                                  importance_nested_sampling=False,
                                  const_efficiency_mode=False)#False was default
        # Store the sample for further use (if needed)
        self._sampler = sampler

        #if using mpi only analyze in rank=0
        if using_mpi:
            if rank==0:
                self.analyze_result()
        else:
            self.analyze_result()

    def analyze_result(self):

        # Save parameter names
        param_index = []
        for i, parameter in enumerate(self._likelihood._parameters.values()):
            param_index.append(parameter.name)

        self._param_names = param_index
        json.dump(self._param_names, open(self.output_dir + 'params.json', 'w'))

        ## Use PyMULTINEST analyzer to gather parameter info
        multinest_analyzer = pymultinest.analyse.Analyzer(n_params=self._n_dim,
                                                          outputfiles_basename=self.output_dir)

        # Get the function value from the chain
        func_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

        # Get the samples from the sampler

        _raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]
        print(_raw_samples)
        print(_raw_samples[0])

        # Find the minimum of the function (i.e. the maximum of func_wrapper)

        idx = func_values.argmax()

        best_fit_values = _raw_samples[idx]

        minimum = func_values[idx] * (-1)
        self._samples = _raw_samples
        self.multinest_data = multinest_analyzer.get_data()

        # set parameters to best fit values
        self._likelihood.set_free_parameters(best_fit_values)

        return best_fit_values, minimum


    def plot_marginals(self, true_params=None):
        """
        Script that does default visualizations (marginal plots, 1-d and 2-d).

        Author: Johannes Buchner (C) 2013
        """
        #if using mpi only rank 0 should plot the marginals
        if using_mpi:
            if rank==0:
                print('model "%s"' % self.output_dir)
                if not os.path.exists(self.output_dir + 'params.json'):
                    sys.stderr.write("""Expected the file %sparams.json with the parameter names.
                            For example, for a three-dimensional problem:

                            ["Redshift $z$", "my parameter 2", "A"]
                            %s""" % (sys.argv[0], __doc__))
                    sys.exit(2)

                parameters = json.load(open(self.output_dir + 'params.json'))
                n_params = len(parameters)

                a = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=self.output_dir)
                s = a.get_stats()

                json.dump(s, open(self.output_dir + 'stats.json', 'w'), indent=4)

                print('  marginal likelihood:')
                print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
                print('  parameters:')
                for p, m in zip(parameters, s['marginals']):
                    lo, hi = m['1sigma']
                    med = m['median']
                    sigma = (hi - lo) / 2
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                    fmt = '%%.%df' % i
                    fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                    print(fmts % (p, med, sigma))

                print('creating marginal plot ...')
                p = pymultinest.PlotMarginal(a)

                values = a.get_equal_weighted_posterior()
                assert n_params == len(s['marginals'])
                modes = s['modes']

                dim2 = os.environ.get('D', '1' if n_params > 20 else '2') == '2'
                nbins = 100 if n_params < 3 else 20
                if dim2:
                    plt.figure(figsize=(5 * n_params, 5 * n_params))
                    for i in range(n_params):
                        plt.subplot(n_params, n_params, i + 1)
                        plt.xlabel(parameters[i])

                        m = s['marginals'][i]
                        plt.xlim(m['5sigma'])

                        oldax = plt.gca()
                        x, w, patches = oldax.hist(values[:, i], bins=nbins, edgecolor='grey', color='grey',
                                                   histtype='stepfilled', alpha=0.2)
                        oldax.set_ylim(0, x.max())

                        newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                        p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                        newax.set_ylim(0, 1)

                        # Plot vertical lines at the true_params values.
                        if true_params is not None:
                            plt.axvline(x=true_params[i])

                        ylim = newax.get_ylim()
                        y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                        center = m['median']
                        low1, high1 = m['1sigma']
                        # print(center, low1, high1)
                        newax.errorbar(x=center, y=y,
                                       xerr=np.transpose([[center - low1, high1 - center]]),
                                       color='blue', linewidth=2, marker='s')
                        oldax.set_yticks([])
                        # newax.set_yticks([])
                        newax.set_ylabel("Probability")
                        ylim = oldax.get_ylim()
                        newax.set_xlim(m['5sigma'])
                        oldax.set_xlim(m['5sigma'])
                        # plt.close()

                        for j in range(i):
                            plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
                            p.plot_conditional(i, j, bins=20, cmap=plt.cm.gray_r)
                            for m in modes:
                                plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
                            plt.xlabel(parameters[i])
                            plt.ylabel(parameters[j])
                            # plt.savefig('cond_%s_%s.pdf' % (params[i], params[j]), bbox_tight=True)
                            # plt.close()

                    plt.savefig(self.output_dir + 'marg.pdf')
                    plt.savefig(self.output_dir + 'marg.png')
                    # plt.close()
                else:
                    from matplotlib.backends.backend_pdf import PdfPages
                    sys.stderr.write('1dimensional only. Set the D environment variable \n')
                    sys.stderr.write('to D=2 to force 2d marginal plots.\n')
                    pp = PdfPages(self.output_dir + 'marg1d.pdf')

                    for i in range(n_params):
                        plt.figure(figsize=(3, 3))
                        plt.xlabel(parameters[i])
                        plt.locator_params(nbins=5)

                        m = s['marginals'][i]
                        iqr = m['q99%'] - m['q01%']
                        xlim = m['q01%'] - 0.3 * iqr, m['q99%'] + 0.3 * iqr
                        # xlim = m['5sigma']
                        plt.xlim(xlim)

                        oldax = plt.gca()
                        x, w, patches = oldax.hist(values[:, i], bins=np.linspace(xlim[0], xlim[1], 20),
                                                   edgecolor='grey',
                                                   color='grey', histtype='stepfilled', alpha=0.2)
                        oldax.set_ylim(0, x.max())

                        newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                        p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                        newax.set_ylim(0, 1)

                        ylim = newax.get_ylim()
                        y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                        center = m['median']
                        low1, high1 = m['1sigma']
                        # print center, low1, high1
                        newax.errorbar(x=center, y=y,
                                       xerr=np.transpose([[center - low1, high1 - center]]),
                                       color='blue', linewidth=2, marker='s')
                        oldax.set_yticks([])
                        newax.set_ylabel("Probability")
                        ylim = oldax.get_ylim()
                        newax.set_xlim(xlim)
                        oldax.set_xlim(xlim)
                        plt.savefig(pp, format='pdf', bbox_inches='tight')
                        # plt.close()
                    pp.close()
        else:
            print('model "%s"' % self.output_dir)
            if not os.path.exists(self.output_dir + 'params.json'):
                sys.stderr.write("""Expected the file %sparams.json with the parameter names.
            For example, for a three-dimensional problem:
    
            ["Redshift $z$", "my parameter 2", "A"]
            %s""" % (sys.argv[0], __doc__))
                sys.exit(2)

            parameters = json.load(open(self.output_dir + 'params.json'))
            n_params = len(parameters)

            a = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=self.output_dir)
            s = a.get_stats()

            json.dump(s, open(self.output_dir + 'stats.json', 'w'), indent=4)

            print('  marginal likelihood:')
            print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
            print('  parameters:')
            for p, m in zip(parameters, s['marginals']):
                lo, hi = m['1sigma']
                med = m['median']
                sigma = (hi - lo) / 2
                i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))

            print('creating marginal plot ...')
            p = pymultinest.PlotMarginal(a)

            values = a.get_equal_weighted_posterior()
            assert n_params == len(s['marginals'])
            modes = s['modes']

            dim2 = os.environ.get('D', '1' if n_params > 20 else '2') == '2'
            nbins = 100 if n_params < 3 else 20
            if dim2:
                plt.figure(figsize=(5 * n_params, 5 * n_params))
                for i in range(n_params):
                    plt.subplot(n_params, n_params, i + 1)
                    plt.xlabel(parameters[i])

                    m = s['marginals'][i]
                    plt.xlim(m['5sigma'])

                    oldax = plt.gca()
                    x, w, patches = oldax.hist(values[:, i], bins=nbins, edgecolor='grey', color='grey',
                                               histtype='stepfilled', alpha=0.2)
                    oldax.set_ylim(0, x.max())

                    newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                    p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                    newax.set_ylim(0, 1)

                    # Plot vertical lines at the true_params values.
                    if true_params is not None:
                        plt.axvline(x=true_params[i])

                    ylim = newax.get_ylim()
                    y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                    center = m['median']
                    low1, high1 = m['1sigma']
                    # print(center, low1, high1)
                    newax.errorbar(x=center, y=y,
                                   xerr=np.transpose([[center - low1, high1 - center]]),
                                   color='blue', linewidth=2, marker='s')
                    oldax.set_yticks([])
                    # newax.set_yticks([])
                    newax.set_ylabel("Probability")
                    ylim = oldax.get_ylim()
                    newax.set_xlim(m['5sigma'])
                    oldax.set_xlim(m['5sigma'])
                    # plt.close()

                    for j in range(i):
                        plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
                        p.plot_conditional(i, j, bins=20, cmap=plt.cm.gray_r)
                        for m in modes:
                            plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
                        plt.xlabel(parameters[i])
                        plt.ylabel(parameters[j])
                        # plt.savefig('cond_%s_%s.pdf' % (params[i], params[j]), bbox_tight=True)
                        # plt.close()

                plt.savefig(self.output_dir + 'marg.pdf')
                plt.savefig(self.output_dir + 'marg.png')
                #plt.close()
            else:
                from matplotlib.backends.backend_pdf import PdfPages
                sys.stderr.write('1dimensional only. Set the D environment variable \n')
                sys.stderr.write('to D=2 to force 2d marginal plots.\n')
                pp = PdfPages(self.output_dir + 'marg1d.pdf')

                for i in range(n_params):
                    plt.figure(figsize=(3, 3))
                    plt.xlabel(parameters[i])
                    plt.locator_params(nbins=5)

                    m = s['marginals'][i]
                    iqr = m['q99%'] - m['q01%']
                    xlim = m['q01%'] - 0.3 * iqr, m['q99%'] + 0.3 * iqr
                    # xlim = m['5sigma']
                    plt.xlim(xlim)

                    oldax = plt.gca()
                    x, w, patches = oldax.hist(values[:, i], bins=np.linspace(xlim[0], xlim[1], 20), edgecolor='grey',
                                               color='grey', histtype='stepfilled', alpha=0.2)
                    oldax.set_ylim(0, x.max())

                    newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                    p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                    newax.set_ylim(0, 1)

                    ylim = newax.get_ylim()
                    y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                    center = m['median']
                    low1, high1 = m['1sigma']
                    # print center, low1, high1
                    newax.errorbar(x=center, y=y,
                                   xerr=np.transpose([[center - low1, high1 - center]]),
                                   color='blue', linewidth=2, marker='s')
                    oldax.set_yticks([])
                    newax.set_ylabel("Probability")
                    ylim = oldax.get_ylim()
                    newax.set_xlim(xlim)
                    oldax.set_xlim(xlim)
                    plt.savefig(pp, format='pdf', bbox_inches='tight')
                    #plt.close()
                pp.close()

    @property
    def samples(self):

        return self._samples

import numpy as np
import copy

from gbmbkgpy.utils.pha import SPECTRUM
import h5py
from gbmbkgpy.utils.progress_bar import progress_bar

NO_REBIN = 1E-99

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


class DataExporter(object):
    def __init__(self, data, model, saa_object, echan_list, best_fit_values, covariance_matrix):

        self._data = data
        self._model = model  # type: Model
        self._echan_list = echan_list
        self._echan_idx = np.arange(len(echan_list))
        self._best_fit_values = best_fit_values
        self._covariance_matrix = covariance_matrix

        self._name = "Count rate detector %s" % data._det

        # The MET start time of the first used day
        self._day_met = data.day_met[0]

        self._free_parameters = self._model.free_parameters
        self._parameters = self._model.parameters

        self._param_names = []
        for i, parameter in enumerate(self._parameters.values()):
            self._param_names.append(parameter.name)

        # The data object should return all the time bins that are valid... i.e. non-zero
        self._total_time_bins = data.time_bins[2:-2]
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]

        # Get the SAA mask:
        self._saa_mask = saa_object.saa_mask[2:-2]

        # Get the valid time bins by including the total_mask
        self._time_bins = self._total_time_bins[self._saa_mask]

        # Extract the counts from the data object. should be same size as time bins. For all echans together
        self._counts_all_echan = data.counts[2:-2][self._saa_mask]
        self._total_counts_all_echan = data.counts[2:-2]

        self._total_scale_factor = 1.
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False

    def save_data(self, path, result_dir, save_ppc=True):
        """
        Function to save the data needed to create the plots.
        """
        model_counts = np.zeros((len(self._total_time_bin_widths), len(self._echan_list)))
        stat_err = np.zeros_like(model_counts)

        # Get the model counts
        for echan in self._echan_list:
            model_counts[:, echan] = self._model.get_counts(self._total_time_bins, echan, saa_mask=self._saa_mask)

        # Get the statistical error from the posterior samples
        for echan in self._echan_list:
            counts = self._ppc_data(result_dir, echan)
            rates = counts / self._total_time_bin_widths

            low = np.percentile(rates, 50 - 50 * 0.68, axis=0)[0]
            high = np.percentile(rates, 50 + 50 * 0.68, axis=0)[0]

            stat_err[:, echan] = high - low

        if save_ppc:
            ppc_counts_all = []
            for index in self._echan_list:
                ppc_counts_all.append(self._ppc_data(result_dir, index))

        if rank == 0:
            with h5py.File(path, "w") as f1:

                group_general = f1.create_group('general')

                group_general.create_dataset('Detector', data=self._data.det)
                group_general.create_dataset('Dates', data=self._data.day)
                group_general.create_dataset('day_start_times', data=self._data.day_start_times)
                group_general.create_dataset('day_stop_times', data=self._data.day_stop_times)
                group_general.create_dataset('saa_mask', data=self._saa_mask, compression="gzip", compression_opts=9)

                group_general.create_dataset('best_fit_values', data=self._best_fit_values, compression="gzip", compression_opts=9)
                group_general.create_dataset('covariance_matrix', data=self._covariance_matrix, compression="gzip", compression_opts=9)
                group_general.create_dataset('param_names', data=self._param_names, compression="gzip", compression_opts=9)
                group_general.create_dataset('model_counts', data=model_counts, compression="gzip", compression_opts=9)
                group_general.create_dataset('stat_err', data=stat_err, compression="gzip", compression_opts=9)

                for j, index in enumerate(self._echan_idx):
                    source_list = self.get_counts_of_sources(self._total_time_bins, index)

                    model_counts = self._model.get_counts(self._total_time_bins, index, saa_mask=self._saa_mask)

                    time_bins = self._total_time_bins

                    observed_counts = self._total_counts_all_echan[:, index]

                    group_echan = f1.create_group('Echan {}'.format(self._echan_list[j]))

                    # group_ppc = group_echan.create_group('PPC data')

                    group_sources = group_echan.create_group('Sources')

                    group_echan.create_dataset('time_bins_start', data=time_bins[:, 0], compression="gzip", compression_opts=9)
                    group_echan.create_dataset('time_bins_stop', data=time_bins[:, 1], compression="gzip", compression_opts=9)
                    group_echan.create_dataset('total_model_counts', data=model_counts, compression="gzip", compression_opts=9)
                    group_echan.create_dataset('observed_counts', data=observed_counts, compression="gzip", compression_opts=9)
                    for source in source_list:
                        group_sources.create_dataset(source['label'], data=source['data'], compression="gzip", compression_opts=9)

                    if save_ppc:
                        ppc_counts = ppc_counts_all[j]
                        group_echan.create_dataset('PPC', data=ppc_counts, compression="gzip", compression_opts=9)

    def get_counts_of_sources(self, time_bins, echan):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """

        source_list = []
        color_list = ['b', 'g', 'c', 'm', 'y', 'k', 'navy', 'darkgreen', 'cyan']
        i_index = 0
        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, time_bins, self._saa_mask, echan)
            if np.sum(data) != 0:
                source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
                i_index += 1
        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(i, time_bins, self._saa_mask, echan)
            source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
            i_index += 1

        for i, source_name in enumerate(self._model.fit_spectrum_sources):
            data = self._model.get_fit_spectrum_counts(i, time_bins, self._saa_mask, echan)
            source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
            i_index += 1

        saa_data = self._model.get_saa_counts(self._total_time_bins, self._saa_mask, echan)
        if np.sum(saa_data) != 0:
            source_list.append({"label": "SAA_decays", "data": saa_data, "color": color_list[i_index]})
            i_index += 1
        point_source_data = self._model.get_point_source_counts(self._total_time_bins, self._saa_mask, echan)
        if np.sum(point_source_data) != 0:
            source_list.append(
                {"label": "Point_sources", "data": point_source_data, "color": color_list[i_index]})
            i_index += 1
        return source_list

    def _ppc_data(self, result_dir, echan):
        """
        Add ppc plot
        :param result_dir: path to result directory
        :param model: Model object
        :param time_bin: Time bins where to compute ppc steps
        :param saa_mask: Mask which time bins are set to zero
        :param echan: Which echan
        :param q_levels: At which levels the ppc should be plotted
        :param colors: colors for the different q_level
        """
        import pymultinest
        analyzer = pymultinest.analyse.Analyzer(1, result_dir)

        # Make a mask with 300 random True to choose 300 random samples
        N_samples = 500
        rates = []
        counts = []
        a = np.zeros(len(analyzer.get_equal_weighted_posterior()[:, :-1]), dtype=int)
        a[:N_samples] = 1
        np.random.shuffle(a)
        a = a.astype(bool)

        # For these 300 random samples calculate the corresponding rates for all time bins
        # with the parameters of this sample
        if using_mpi:
            points_per_rank = float(N_samples) / float(size)
            points_lower_index = int(np.floor(points_per_rank * rank))
            points_upper_index = int(np.floor(points_per_rank * (rank + 1)))
            if rank == 0:
                with progress_bar(len(analyzer.get_equal_weighted_posterior()[:, :-1][a]
                                      [points_lower_index:points_upper_index]),
                                  title='Calculating PPC for echan {}'.format(echan)) as p:

                    for i, sample in enumerate(analyzer.get_equal_weighted_posterior()[:, :-1][a]
                                               [points_lower_index:points_upper_index]):
                        synth_data = self.get_synthetic_data(sample)
                        counts.append(synth_data.counts[:, echan])
                        p.increase()

            else:
                for i, sample in enumerate(analyzer.get_equal_weighted_posterior()[:, :-1][a]
                                           [points_lower_index:points_upper_index]):
                    synth_data = self.get_synthetic_data(sample)
                    counts.append(synth_data.counts[:, echan])

            counts = np.array(counts)
            counts_g = comm.gather(counts, root=0)
            if rank == 0:
                counts_g = np.concatenate(counts_g)
            counts = comm.bcast(counts_g, root=0)
        else:
            for i, sample in enumerate(analyzer.get_equal_weighted_posterior()[:, :-1][a]):
                synth_data = self.get_synthetic_data(sample)
                counts.append(synth_data.counts[:, echan])
            counts = np.array(counts)
        return counts

    def get_synthetic_data(self, synth_parameters, synth_model=None):
        """
        Creates a ContinousData object with synthetic data based on the total counts from the synth_model
        If no synth_model is passed it makes a deepcopy of the existing model
        :param synth_parameters:
        :return:
        """

        synth_data = copy.deepcopy(self._data)

        if synth_model == None:
            synth_model = copy.deepcopy(self._model)

        for i, parameter in enumerate(synth_model.free_parameters.itervalues()):
            parameter.value = synth_parameters[i]

        for echan in self._echan_idx:
            synth_data.counts[:, echan][2:-2] = np.random.poisson(synth_model.get_counts(synth_data.time_bins[2:-2], echan))

        return synth_data


class PHAExporter(DataExporter):

    def __init__(self, *args, **kwargs):
        super(PHAExporter, self).__init__(*args, **kwargs)

    def save_pha(self, path, result_dir):
        model_counts = np.zeros((len(self._total_time_bin_widths), len(self._echan_list)))
        stat_err = np.zeros_like(model_counts)

        # Get the model counts
        for echan in self._echan_list:
            model_counts[:, echan] = self._model.get_counts(self._total_time_bins, echan, saa_mask=self._saa_mask)
        model_rates = model_counts / self._total_time_bin_widths

        # Get the statistical error from the posterior samples
        for echan in self._echan_list:
            counts = self._ppc_data(result_dir, echan)
            rates = counts / self._total_time_bin_widths

            low = np.percentile(rates, 50 - 50 * 0.68, axis=0)[0]
            high = np.percentile(rates, 50 + 50 * 0.68, axis=0)[0]

            stat_err[:, echan] = high - low

        spectrum = SPECTRUM(tstart=self._total_time_bins[:, 1],
                            telapse=self._total_time_bin_widths,
                            channel=self._echan_list,
                            rate=model_rates,
                            quality=np.zeros_like(model_rates, dtype=int),
                            grouping=np.ones_like(self._echan_list),
                            exposure=self._total_time_bin_widths,
                            backscale=None,
                            respfile=None,
                            ancrfile=None,
                            back_file=None,
                            sys_err=None,
                            stat_err=stat_err,
                            is_poisson=False)

        spectrum.hdu.dump(path)



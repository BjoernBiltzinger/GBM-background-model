import numpy as np
import copy

from gbmbkgpy.utils.pha import SPECTRUM, PHAII
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
        self._model = model
        self._saa_object = saa_object

        self._echan_names = echan_list
        self._echan_list = np.arange(len(echan_list))
        self._best_fit_values = best_fit_values
        self._covariance_matrix = covariance_matrix

        self._name = "Count rate detector %s" % data.det

        # The MET start time of the first used day
        self._day_met = data.day_met[0]

        self._total_time_bins = self._data.time_bins[2:-2]
        self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]
        self._total_counts_all_echan = self._data.counts[2:-2]
        self._saa_mask = self._saa_object.saa_mask[2:-2]

        self._total_scale_factor = 1.
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False

    def save_data(self, file_path, result_dir, save_ppc=True):
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
            counts = self._ppc_data(result_dir, echan)[:, 2:-2]

            low = np.percentile(counts, 50 - 50 * 0.68, axis=0)[0]
            high = np.percentile(counts, 50 + 50 * 0.68, axis=0)[0]

            stat_err[:, echan] = high - low

        if save_ppc:
            ppc_counts_all = []
            for index in self._echan_list:
                ppc_counts_all.append(self._ppc_data(result_dir, index))

        if rank == 0:
            with h5py.File(file_path, "w") as f1:
                group_general = f1.create_group('general')

                group_general.create_dataset('detector', data=self._data.det)
                group_general.create_dataset('dates', data=np.string_(self._data.day))
                group_general.create_dataset('echans', data=self._echan_names)
                group_general.create_dataset('day_start_times', data=self._data.day_start_times)
                group_general.create_dataset('day_stop_times', data=self._data.day_stop_times)
                group_general.create_dataset('saa_mask', data=self._saa_mask, compression="gzip", compression_opts=9)
                group_general.create_dataset('time_bins_start', data=self._total_time_bins[:, 0], compression="gzip", compression_opts=9)
                group_general.create_dataset('time_bins_stop', data=self._total_time_bins[:, 1], compression="gzip", compression_opts=9)

                group_general.create_dataset('best_fit_values', data=self._best_fit_values, compression="gzip", compression_opts=9)
                group_general.create_dataset('param_names', data=np.string_(self._model.parameter_names))
                group_general.create_dataset('model_counts', data=model_counts, compression="gzip", compression_opts=9)
                group_general.create_dataset('stat_err', data=stat_err, compression="gzip", compression_opts=9)

                if self._covariance_matrix is not None:
                    group_general.create_dataset('covariance_matrix', data=self._covariance_matrix, compression="gzip", compression_opts=9)

                for j, index in enumerate(self._echan_list):
                    source_list = self.get_counts_of_sources(self._total_time_bins, index)
                    model_counts = self._model.get_counts(self._total_time_bins, index, saa_mask=self._saa_mask)
                    observed_counts = self._total_counts_all_echan[:, index]

                    group_echan = f1.create_group('echan {}'.format(self._echan_names[j]))
                    group_echan.create_dataset('total_model_counts', data=model_counts, compression="gzip", compression_opts=9)
                    group_echan.create_dataset('observed_counts', data=observed_counts, compression="gzip", compression_opts=9)

                    group_sources = group_echan.create_group('Sources')
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
        :param echan: Which echan
        """
        import pymultinest
        analyzer = pymultinest.analyse.Analyzer(1, result_dir)
        mn_posteriour_samples = analyzer.get_equal_weighted_posterior()[:, :-1]

        counts = []

        # Make a mask with 500 random True to choose N_samples random samples,
        # if multinest returns less then 500 posterior samples use next smaller *00
        N_samples = 500 if len(mn_posteriour_samples) > 500 else int(len(mn_posteriour_samples) / 100) * 100

        a = np.zeros(len(mn_posteriour_samples), dtype=int)
        a[:N_samples] = 1
        np.random.shuffle(a)
        a = a.astype(bool)

        # For these N_samples random samples calculate the corresponding rates for all time bins
        # with the parameters of this sample
        if using_mpi:
            points_per_rank = float(N_samples) / float(size)
            points_lower_index = int(np.floor(points_per_rank * rank))
            points_upper_index = int(np.floor(points_per_rank * (rank + 1)))
            if rank == 0:
                with progress_bar(len(mn_posteriour_samples[a][points_lower_index:points_upper_index]),
                                  title='Calculating PPC for echan {}'.format(echan)) as p:

                    for i, sample in enumerate(mn_posteriour_samples[a][points_lower_index:points_upper_index]):
                        synth_data = self.get_synthetic_data(sample)
                        counts.append(synth_data.counts[:, echan])
                        p.increase()

            else:
                for i, sample in enumerate(mn_posteriour_samples[a][points_lower_index:points_upper_index]):
                    synth_data = self.get_synthetic_data(sample)
                    counts.append(synth_data.counts[:, echan])
            counts = np.array(counts)
            counts_g = comm.gather(counts, root=0)

            if rank == 0:
                counts_g = np.concatenate(counts_g, axis=0)
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

        for i, parameter in enumerate(synth_model.free_parameters.values()):
            parameter.value = synth_parameters[i]

        for echan in self._echan_list:
            synth_data.counts[:, echan][2:-2] = np.random.poisson(synth_model.get_counts(synth_data.time_bins[2:-2], echan))

        return synth_data


class PHAExporter(DataExporter):

    def __init__(self, *args, **kwargs):
        super(PHAExporter, self).__init__(*args, **kwargs)

    def save_pha(self, path, result_dir):
        model_counts = np.zeros((len(self._total_time_bin_widths), len(self._echan_names)))
        stat_err = np.zeros_like(model_counts)

        # Get the model counts
        for echan in self._echan_names:
            model_counts[:, echan] = self._model.get_counts(self._total_time_bins, echan, saa_mask=self._saa_mask)
        model_rates = model_counts / self._total_time_bin_widths

        # Get the statistical error from the posterior samples
        for echan in self._echan_names:
            counts = self._ppc_data(result_dir, echan)[:, 2:-2]
            rates = counts / self._total_time_bin_widths

            low = np.percentile(rates, 50 - 50 * 0.68, axis=0)[0]
            high = np.percentile(rates, 50 + 50 * 0.68, axis=0)[0]

            stat_err[:, echan] = high - low

        spectrum = SPECTRUM(tstart=self._total_time_bins[:, 1],
                            telapse=self._total_time_bin_widths,
                            channel=self._echan_names,
                            rate=model_rates,
                            quality=np.zeros_like(model_rates, dtype=int),
                            grouping=np.ones_like(self._echan_names),
                            exposure=self._total_time_bin_widths,
                            backscale=None,
                            respfile=None,
                            ancrfile=None,
                            back_file=None,
                            sys_err=None,
                            stat_err=stat_err,
                            is_poisson=False)

        spectrum.hdu.dump(path)


class PHACombiner(object):

    def __init__(self, result_file_list):
        self._det = None
        self._dates = None
        self._total_time_bins = None
        self._total_time_bin_widths = None
        self._model_counts = None
        self._model_rates = None
        self._stat_err = None
        self._echan_names = []

        self._load_result_file(result_file_list)

    def _load_result_file(self, result_file_list):
        for i, file in enumerate(result_file_list):
            with h5py.File(file, 'r') as f:
                det = np.array(f['general']['detector'])
                dates = np.array(f['general']['dates'])
                echans = np.array(f['general']['echans'])
                time_bins_start = np.array(f['general']['time_bins_start'])
                time_bins_stop = np.array(f['general']['time_bins_stop'])

                model_counts = np.array(f['general']['model_counts'])
                stat_err = np.array(f['general']['stat_err'])

            if i == 0:
                self._det = det
                self._dates = dates
                self._total_time_bins = np.vstack((time_bins_start, time_bins_stop)).T
                self._total_time_bin_widths = np.diff(self._total_time_bins, axis=1)[:, 0]
                self._model_counts = np.zeros((len(time_bins_stop), 8))
                self._model_rates = np.zeros((len(time_bins_stop), 8))
                self._stat_err = np.zeros_like(self._model_counts)
            else:
                assert self._det == det
                assert self._dates == dates
                assert np.array_equal(self._total_time_bins, np.vstack((time_bins_start, time_bins_stop)).T)

            for index, echan in enumerate(echans):
                assert echan not in self._echan_names, '{} already loaded, you have to resolve the conflict by hand'.format(echan)

                self._echan_names.append(echan)
                self._model_counts[:, echan] = model_counts[:, index]
                self._model_rates[:, echan] = model_counts[:, index] / self._total_time_bin_widths
                self._stat_err[:, echan] = stat_err[:, index]

    def save_pha(self, path):
        spectrum = PHAII(instrument_name='GBM_NAI_0{}'.format(self._det),
                         telescope_name='GLAST',
                         tstart=self._total_time_bins[:, 1],
                         telapse=self._total_time_bin_widths,
                         channel=self._echan_names,
                         rate=self._model_rates,
                         quality=np.zeros_like(self._model_rates, dtype=int),
                         grouping=np.ones_like(self._echan_names),
                         exposure=self._total_time_bin_widths,
                         backscale=None,
                         respfile=None,
                         ancrfile=None,
                         back_file=None,
                         sys_err=None,
                         stat_err=self._stat_err,
                         is_poisson=False)

        spectrum.writeto(path)

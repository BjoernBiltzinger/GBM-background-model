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
        rank = 0
except:
    using_mpi = False
    rank = 0


class DataExporter(object):
    def __init__(self, data, model, saa_object, echans, best_fit_values):

        self._data = data
        self._model = model
        self._saa_object = saa_object

        self._echans = echans
        self._best_fit_values = best_fit_values

        self._time_bins = self._data.time_bins
        self._saa_mask = self._saa_object.saa_mask

        self._total_scale_factor = 1.
        self._rebinner = None
        self._fit_rebinned = False
        self._fit_rebinner = None
        self._grb_mask_calculated = False

    def save_data(self, file_path, result_dir, save_ppc=True):
        """
        Function to save the data needed to create the plots.
        """

        # Get the model counts
        model_counts = self._model.get_counts(
            time_bins=self._time_bins
        )


        # TODO: Vectorize the stat_error and ppc calculation
        #####################

        # Get the statistical error from the posterior samples
        ppc_counts = self._ppc_data(result_dir)

        low = np.percentile(counts, 50 - 50 * 0.68, axis=0)[0]
        high = np.percentile(counts, 50 + 50 * 0.68, axis=0)[0]

        stat_err= high - low

       ####################


        if rank == 0:
            with h5py.File(file_path, "w") as f:

                f.attrs['dates'] = self._data.dates
                f.attrs['detectors'] = self._data.detectors
                f.attrs['echans'] = self._data.echans
                f.attrs['param_names'] = self._model.paramter_names
                f.attrs['best_fit_values'] = self._best_fit_values

                f.create_dataset('day_start_times', data=self._data.day_start_times)
                f.create_dataset('day_stop_times', data=self._data.day_stop_times)
                f.create_dataset('saa_mask', data=self._saa_mask, compression="gzip", compression_opts=9)
                f.create_dataset('time_bins_start', data=self._time_bins[:, 0], compression="gzip", compression_opts=9)
                f.create_dataset('time_bins_stop', data=self._time_bins[:, 1], compression="gzip", compression_opts=9)
                f.create_dataset('observed_counts', data=self._data.counts, compression="gzip", compression_opts=9)

                f.create_dataset('model_counts', data=model_counts, compression="gzip", compression_opts=9)
                f.create_dataset('stat_err', data=stat_err, compression="gzip", compression_opts=9)

                group_sources = f.create_group('sources')
                for source in source_list:
                    group_sources.create_dataset(source['label'], data=source['data'], compression="gzip", compression_opts=9)

                if save_ppc:
                    f.create_dataset('ppc', data=ppc_counts, compression="gzip", compression_opts=9)


    def get_counts_of_sources(self, time_bins):
        """
        Builds a list of the different model sources.
        Each source is a dict containing the label of the source, the data, and the plotting color
        :return:
        """

        source_list = []
        color_list = ['b', 'g', 'c', 'm', 'y', 'k', 'navy', 'darkgreen', 'cyan']
        i_index = 0

        for i, source_name in enumerate(self._model.continuum_sources):
            data = self._model.get_continuum_counts(i, time_bins, self._saa_mask)
            if np.sum(data) != 0:
                source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
                i_index += 1

        for i, source_name in enumerate(self._model._global_sources):
            data = self._model.get_global_counts(i, time_bins, self._saa_mask)
            source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
            i_index += 1

        for i, source_name in enumerate(self._model.fit_spectrum_sources):
            data = self._model.get_fit_spectrum_counts(i, time_bins, self._saa_mask)
            source_list.append({"label": source_name, "data": data, "color": color_list[i_index]})
            i_index += 1

        saa_data = self._model.get_saa_counts(self._total_time_bins, self._saa_mask)
        if np.sum(saa_data) != 0:
            source_list.append({"label": "SAA_decays", "data": saa_data, "color": color_list[i_index]})
            i_index += 1

        point_source_data = self._model.get_point_source_counts(self._total_time_bins, self._saa_mask)
        if np.sum(point_source_data) != 0:
            source_list.append(
                {"label": "Point_sources", "data": point_source_data, "color": color_list[i_index]})
            i_index += 1

        return source_list

    def _ppc_data(self, result_dir):
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
                        counts.append(synth_data.counts)
                        p.increase()

            else:
                for i, sample in enumerate(mn_posteriour_samples[a][points_lower_index:points_upper_index]):
                    synth_data = self.get_synthetic_data(sample)
                    counts.append(synth_data.counts)
            counts = np.array(counts)
            counts_g = comm.gather(counts, root=0)

            if rank == 0:
                counts_g = np.concatenate(counts_g, axis=0)
            counts = comm.bcast(counts_g, root=0)
        else:
            for i, sample in enumerate(analyzer.get_equal_weighted_posterior()[:, :-1][a]):
                synth_data = self.get_synthetic_data(sample)
                counts.append(synth_data.counts)
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

        synth_data.counts = np.random.poisson(synth_model.get_counts(synth_data.time_bins))

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

det_name_lookup = {
    'n0': 'NAI_00',
    'n1': 'NAI_01',
    'n2': 'NAI_02',
    'n3': 'NAI_03',
    'n4': 'NAI_04',
    'n5': 'NAI_05',
    'n6': 'NAI_06',
    'n7': 'NAI_07',
    'n8': 'NAI_08',
    'n9': 'NAI_09',
    'na': 'NAI_10',
    'nb': 'NAI_11',
    'b0': 'BGO_00',
    'b1': 'BGO_01',
}

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

    def save_pha(self, path, start_time=None, end_time=None, trigger_time=None):

        if start_time is not None and end_time is not None:
            idx_min_time = self._total_time_bins[:, 0] >= start_time
            idx_max_time = self._total_time_bins[:, 1] <= end_time
            idx_valid_bin = idx_min_time * idx_max_time
        else:
            idx_valid_bin = np.ones_like(self._total_time_bins[:, 0], dtype=bool)

        if trigger_time is None:
            trigger_time = 0.

        spectrum = PHAII(instrument_name='GBM_{}'.format(det_name_lookup[str(self._det)]),
                         telescope_name='Fermi',
                         tstart=self._total_time_bins[idx_valid_bin][:, 1] - trigger_time,
                         telapse=self._total_time_bin_widths[idx_valid_bin],
                         channel=self._echan_names,
                         rate=self._model_rates[idx_valid_bin],
                         quality=np.zeros_like(self._model_rates[idx_valid_bin], dtype=int),
                         grouping=np.ones_like(self._echan_names),
                         exposure=self._total_time_bin_widths[idx_valid_bin],
                         backscale=np.ones_like(self._model_rates[idx_valid_bin][:, 0]),
                         respfile=None,
                         ancrfile=None,
                         back_file=None,
                         sys_err=None,
                         stat_err=self._stat_err[idx_valid_bin],
                         is_poisson=False)

        spectrum.writeto(path)


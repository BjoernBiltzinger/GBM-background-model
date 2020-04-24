import os
import shutil
import numpy as np
import h5py

from gbmbkgpy.io.export import DataExporter
from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.data.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.fitting.background_like import BackgroundLike
from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.geometry_calc import Geometry
from gbmbkgpy.utils.response_precalculation import Response_Precalculation
import pytest


@pytest.mark.run(order=1)
def test_model_builder():
    path_of_tests = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(get_path_of_external_data_dir(), 'ctime', '150126')
    file_name = 'glg_ctime_n0_150126_test.pha'

    shutil.copy(os.path.join(path_of_tests, 'datasets', file_name), os.path.join(data_path, file_name))

    model_generator = BackgroundModelGenerator()

    model_generator.from_config_file(os.path.join(path_of_tests, 'config_test.yml'))

    # Check that all objects are instantiated correctly
    assert isinstance(model_generator.data, Data)
    assert isinstance(model_generator.external_properties, ExternalProps)
    assert isinstance(model_generator.saa_calc, SAA_calc)
    assert isinstance(model_generator.response, Response_Precalculation)
    assert isinstance(model_generator.geometry, Geometry)
    assert isinstance(model_generator.source_list, list)
    assert isinstance(model_generator.model, Model)
    assert isinstance(model_generator.likelihood, BackgroundLike)
    assert isinstance(model_generator.parameter_bounds, dict)
    assert isinstance(model_generator.config, dict)

    echans = model_generator.config['general']['echans']
    echans_idx = np.arange(len(echans))
    detectors = model_generator.config['general']['detectors']

    test_responses = {}
    test_geometries = {}

    test_sources_saa = {}
    test_sources_continuum = {}
    test_sources_global = {}

    with h5py.File(os.path.join(path_of_tests, 'datasets', '150126_test_comb.hd5'), 'r') as f:
        test_time_bins = f['time_bins'][()]
        test_counts = f['counts'][()]

        test_rebinned_time_bins = f['rebinned_time_bins'][()]
        test_rebinned_counts = f['rebinned_counts'][()]

        test_saa_mask = f['saa_mask'][()]
        test_rebinned_saa_mask = f['rebinned_saa_mask'][()]

        for det in f['responses'].keys():
            test_responses[det] = {}

            test_responses[det]['response_array'] = \
                f['responses'][det]['response_array'][()]


        for det in f['geometries'].keys():
            test_geometries[det] = {}

            test_geometries[det]['earth_az'] = \
                f['geometries'][det]['earth_az'][()]

            test_geometries[det]['earth_zen'] = \
                f['geometries'][det]['earth_zen'][()]


        for src_name in f['sources']['saa'].keys():
            test_sources_saa[src_name] = {}

            test_sources_saa[src_name]['echan'] = \
                f['sources']['saa'][src_name].attrs['echan']

            test_sources_saa[src_name]['counts'] = \
                f['sources']['saa'][src_name]['counts'][()]


        for src_name in f['sources']['continuum'].keys():
            test_sources_continuum[src_name] = {}

            test_sources_continuum[src_name]['echan'] = \
                f['sources']['continuum'][src_name].attrs['echan']

            test_sources_continuum[src_name]['counts'] = \
                f['sources']['continuum'][src_name]['counts'][()]


        for src_name in f['sources']['global'].keys():
            test_sources_global[src_name] = {}

            test_sources_global[src_name]['counts'] = \
                f['sources']['global'][src_name]['counts'][()]


    # Get the test counts from the same echan specified in the config
    test_counts_echans = test_counts[:, :, echans].reshape(
        (len(test_counts), len(detectors), len(echans))
    )

    test_rebinned_counts_echans = test_rebinned_counts[:, :, echans].reshape(
        (len(test_rebinned_counts), len(detectors), len(echans))
    )

    # Test if the Data class importet the synthetic data correctly
    # Use the unbinned time_bins and counts
    assert np.array_equal(test_time_bins, model_generator.data._time_bins)
    assert np.array_equal(test_counts_echans, model_generator.data._counts)

    # Now test if the rebinned counts and time_bins are correct
    assert np.array_equal(test_rebinned_time_bins, model_generator.data.time_bins)
    assert np.array_equal(test_rebinned_counts_echans, model_generator.data.counts)

    # Check if the SAA mask was calculated correctly
    assert np.array_equal(test_saa_mask, model_generator.saa_calc._saa_mask)
    assert np.array_equal(test_rebinned_saa_mask, model_generator.saa_calc.saa_mask)

    # Check if response precalculation is correct
    for det in model_generator.config['general']['detectors']:

        assert np.allclose(
            test_responses[det]['response_array'],
            model_generator.response.responses[det].response_array,
            rtol=1e-6
        )

    # Check if geometry precalculation is correct
    for det in model_generator.config['general']['detectors']:

        assert np.array_equal(
            test_geometries[det]['earth_az'],
            model_generator.geometry.geometries[det].earth_az,
        )

        assert np.array_equal(
            test_geometries[det]['earth_zen'],
            model_generator.geometry.geometries[det].earth_zen
        )


    # Check the saa sources
    for saa_src in test_sources_saa.items():

        assert saa_src[1]['echan'] == model_generator.model.saa_sources[saa_src[0]].echan

        assert np.array_equal(
            saa_src[1]['counts'],
            model_generator.model.saa_sources[saa_src[0]].get_counts(
                model_generator.data.time_bins[2:-2]
            )
        )


    # Check the continuum sources
    for cont_src in test_sources_continuum.items():

        assert cont_src[1]['echan'] == model_generator.model.continuum_sources[cont_src[0]].echan

        assert np.array_equal(
            cont_src[1]['counts'],
            model_generator.model.continuum_sources[cont_src[0]].get_counts(
                model_generator.data.time_bins[2:-2]
            )
        )

    # Check the global sources
    for global_src in test_sources_global.items():

        assert np.allclose(
            global_src[1]['counts'][:, :, echans],
            model_generator.model.global_sources[global_src[0]].get_counts(
                model_generator.data.time_bins[2:-2]
            )[:, echans_idx],
            rtol=1e-7
        )

    pytest._test_model_generator = model_generator



@pytest.mark.run(order=2)
def test_fitting():
    model_generator = pytest._test_model_generator

    if model_generator.config['fit']['method'] == 'multinest':
        minimizer = MultiNestFit(
            likelihood=model_generator.likelihood,
            parameters=model_generator.model.free_parameters
        )

        # Fit with multinest and define the number of live points one wants to use
        minimizer.minimize_multinest(
            n_live_points=model_generator.config['fit']['multinest']['num_live_points'],
            const_efficiency_mode=model_generator.config['fit']['multinest']['constant_efficiency_mode']
        )

        # Plot Marginals
        minimizer.create_corner_plot()
    else:
        raise KeyError('Invalid fit method')

    true_values = np.array([5.03012970e+01, 8.78085840e-01, 1.10441586e+03, 1.56685886e-02, 4.15783951e-02, 8.69100647e-03, 7.81257334e-02])
    assert np.allclose(minimizer.best_fit_values, true_values, rtol=1e2)

    minimizer.comp_covariance_matrix()

    assert isinstance(minimizer.cov_matrix, np.ndarray)

    pytest._test_output_dir = minimizer.output_dir
    pytest._test_best_fit_values = minimizer.best_fit_values
    pytest._test_cov_matrix = minimizer.cov_matrix



@pytest.mark.run(order=3)
def test_data_export():
    config = pytest._test_model_generator.config
    data_exporter = DataExporter(
        data=pytest._test_model_generator.data,
        model=pytest._test_model_generator.model,
        saa_object=pytest._test_model_generator.saa_calc,
        echan_list=config['general']['echan_list'],
        best_fit_values=pytest._test_best_fit_values,
        covariance_matrix=pytest._test_cov_matrix
    )

    result_file_name = "fit_result_{}_{}_e{}.hdf5".format(config['general']['dates'],
                                                          config['general']['detectors'],
                                                          config['general']['echans'])

    data_exporter.save_data(
        file_path=os.path.join(pytest._test_output_dir, result_file_name),
        result_dir=pytest._test_output_dir,
        save_ppc=config['export']['save_ppc']
    )
    assert file_existing_and_readable(os.path.join(pytest._test_output_dir, result_file_name))


@pytest.mark.run(order=4)
def test_plotting():
    config = pytest._test_model_generator.config
    # Create Plotter object that creates the plots

    plot_generator = ResultPlotGenerator.from_result_instance(
        config_file=config,
        data=pytest._test_model_generator.data,
        model=pytest._test_model_generator.model,
        saa_object=pytest._test_model_generator.saa_calc,
    )

    plot_generator.create_plots(
        output_dir=pytest._test_output_dir
    )

    for plot_path in plot_generator._plot_path_list:
        assert file_existing_and_readable(plot_path)

        os.remove(plot_path)


@pytest.mark.run(order=5)
def test_delete_outputs():
    if os.access(pytest._test_output_dir, os.F_OK):
       shutil.rmtree(pytest._test_output_dir)

    data_path = os.path.join(get_path_of_external_data_dir(), 'ctime', '150126')
    file_name = 'glg_ctime_n0_150126_test.pha'
    file_path = os.path.join(data_path, file_name)

    if file_existing_and_readable(file_path):
        os.remove(file_path)

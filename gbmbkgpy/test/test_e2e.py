import os
import shutil
import numpy as np

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

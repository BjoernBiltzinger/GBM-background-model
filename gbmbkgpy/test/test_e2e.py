import os
import shutil

from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.data.external_prop import ExternalProps
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.fitting.background_like import BackgroundLike
from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.geometry_calc import Geometry
from gbmbkgpy.utils.response_precalculation import Response_Precalculation
import pytest


# @pytest.fixture
# def model_generator():
#     path_of_tests = os.path.dirname(os.path.abspath(__file__))
#
#     data_path = get_path_of_external_data_dir()
#     file_path = os.path.join(data_path, 'ctime', '150126')
#     file_name = 'glg_ctime_n0_150126_test.pha'
#
#     shutil.copy(os.path.join(path_of_tests, file_name), os.path.join(file_path, file_name))
#
#     model_gen = BackgroundModelGenerator()
#
#     model_gen.from_config_file(os.path.join(path_of_tests, 'config_test.yml'))
#
#     return model_gen


def test_model_builder_properties():
    print('Started Model Builder')
    path_of_tests = os.path.dirname(os.path.abspath(__file__))

    data_path = get_path_of_external_data_dir()
    file_path = os.path.join(data_path, 'ctime', '150126')
    file_name = 'glg_ctime_n0_150126_test.pha'

    shutil.copy(os.path.join(path_of_tests, file_name), os.path.join(file_path, file_name))

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

    pytest.model_generator = model_generator


def test_fitting():
    model_generator = pytest.model_generator

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
        minimizer.plot_marginals()
    else:
        raise KeyError('Invalid fit method')

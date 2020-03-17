import os
import shutil
import numpy as np

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.importer import FitImporter
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.data.continuous_data import Data
from gbmbkgpy.modeling.model import Model
from gbmbkgpy.fitting.background_like import BackgroundLike
from gbmbkgpy.utils.saa_calc import SAA_calc
import pytest


@pytest.mark.run(order=1)
def test_fit_import():
    print('Started Model Builder')
    path_of_tests = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(get_path_of_external_data_dir(), 'ctime', '150126')
    file_name = 'glg_ctime_n0_150126_test.pha'

    shutil.copy(os.path.join(path_of_tests, 'datasets', file_name), os.path.join(file_path, file_name))

    fit_importer = FitImporter(
        config=os.path.join(path_of_tests, 'config_test.yml'),
        fit_result_hdf5=os.path.join(path_of_tests, 'datasets', 'data_of_fit_test.hdf5')
    )

    assert isinstance(fit_importer.data, Data)
    assert isinstance(fit_importer.model, Model)
    assert isinstance(fit_importer.likelihood, BackgroundLike)
    assert isinstance(fit_importer.saa_calc, SAA_calc)
    assert isinstance(fit_importer.best_fit_values, np.ndarray)
    assert isinstance(fit_importer.config, dict)

    data_path = os.path.join(get_path_of_external_data_dir(), 'ctime', '150126')
    file_name = 'glg_ctime_n0_150126_test.pha'
    file_path = os.path.join(data_path, file_name)

    if file_existing_and_readable(file_path):
        os.remove(file_path)
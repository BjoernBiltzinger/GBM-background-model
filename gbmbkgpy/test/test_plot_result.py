import os

from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
import pytest


@pytest.mark.run(order=1)
def test_plot_result():
    print('Started Model Builder')
    path_of_tests = os.path.dirname(os.path.abspath(__file__))

    result_plot_generator = ResultPlotGenerator(
        config_file=os.path.join(path_of_tests, 'config_plot_test.yml'),
        data_path=os.path.join(path_of_tests, 'datasets', 'data_of_fit_test.hdf5'))

    result_plot_generator.create_plots()

    for plot_path in result_plot_generator._plot_path_list:
        assert file_existing_and_readable(plot_path)

        os.remove(plot_path)

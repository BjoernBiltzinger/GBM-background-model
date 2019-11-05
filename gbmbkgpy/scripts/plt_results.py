import os
import sys
import argparse
import matplotlib

matplotlib.use('AGG')
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, help='Path of the config file')
    parser.add_argument('--data_path', type=str, help='Path of the data file')
    args = parser.parse_args()

    plot_config = {}
    component_config = {}
    style_config = {}
    highlight_config = {}

    if args.config is not None:
        config_path = args.config
    else:
        config_path = 'config_plot_default'

    config_dir_path = os.path.dirname(config_path)

    sys.path.append(config_dir_path)
    module = __import__(config_path, globals(), locals(), ['*'])
    for k in dir(module):
        locals()[k] = getattr(module, k)

    if plot_config == {} or component_config == {} or style_config == {}:
        raise Exception('You should provide a plot_dict and color_dict in your config file')

    if args.data_path is not None:
        plot_config['data_path'] = args.data_path

    result_plot_generator = ResultPlotGenerator(plot_config, component_config, style_config, highlight_config)

    result_plot_generator.create_plots()

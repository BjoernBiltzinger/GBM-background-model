import os
import sys
import argparse
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', type=str, help='Path of the config file')
    args = parser.parse_args()

    plot_dict = {}
    color_dict = {}
    config_dir_path = os.path.dirname(args.config_file)
    sys.path.append(config_dir_path)
    module = __import__(args.config_file, globals(), locals(), ['*'])
    for k in dir(module):
        locals()[k] = getattr(module, k)

    if plot_dict == {} or color_dict == {}:
        raise Exception('You should provide a plot_dict and color_dict in your config file')

    result_plot_generator = ResultPlotGenerator(plot_dict, color_dict)

    # result_plot_generator.add_grb_trigger(grb_name, trigger_time, time_format='UTC', time_offset=0, color='b')
    # result_plot_generator.add_occ_region(occ_name, time_start, time_stop, time_format='UTC', color='grey')

    result_plot_generator.create_plots()

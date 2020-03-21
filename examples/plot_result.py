import argparse
import matplotlib
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator

matplotlib.use('AGG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, help='Path of the config file', required=True)
    parser.add_argument('-d', '--data_path', type=str, help='Path of the data file', required=True)
    args = parser.parse_args()

    result_plot_generator = ResultPlotGenerator(args.config, args.data_path)

    result_plot_generator.create_plots()
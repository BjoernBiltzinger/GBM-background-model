from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.io.export import DataExporter, PHAWriter
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit

from gbmbkgpy.utils import global_exept_hook

global_exept_hook._add_hook_if_enabled()

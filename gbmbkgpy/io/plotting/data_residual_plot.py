import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from gbmbkgpy.io.plotting.step_plots import step_plot
#from threeML.config.config import config
from gbmbkgpy.exceptions.custom_exceptions import custom_warnings

class ResidualPlot(object):

    def __init__(self,**kwargs):
        """
        A class that makes data/residual plots
        :param show_residuals: to show the residuals
        :param ratio_residuals: to use ratios instead of sigma
        :param model_subplot: and axis or list of axes to plot to rather than create a new one
        """


        self._ratio_residuals = False
        self._show_residuals = True


        if 'show_residuals' in kwargs:

            self._show_residuals = bool(kwargs.pop('show_residuals'))

        if 'ratio_residuals' in kwargs:
            self._ratio_residuals = bool(kwargs.pop('ratio_residuals'))


        # this lets you overplot other fits

        if 'model_subplot' in kwargs:

            model_subplot = kwargs.pop('model_subplot')

            # turn on or off residuals

            if self._show_residuals:

                assert type(model_subplot) == list, 'you must supply a list of axes to plot to residual'

                assert len(
                    model_subplot) == 2, 'you have requested to overplot a model with residuals, but only provided one axis to plot'

                self._data_axis, self._residual_axis = model_subplot

            else:

                try:

                    self._data_axis = model_subplot

                    self._fig = self._data_axis.get_figure()

                except(AttributeError):

                    # the user supplied a list of axes

                    self._data_axis = model_subplot[0]

            # we will use the figure associated with
            # the data axis

            self._fig = self._data_axis.get_figure()



        else:

            # turn on or off residuals

            if self._show_residuals:

                self._fig, (self._data_axis, self._residual_axis) = plt.subplots(2, 1,
                                                                                 sharex=True,
                                                                                 gridspec_kw={'height_ratios': [2, 1]},
                                                                                 **kwargs)

            else:

                self._fig, self._data_axis = plt.subplots(**kwargs)




    @property
    def figure(self):
        """
        :return: the figure instance
        """

        return self._fig

    @property
    def data_axis(self):
        """
        :return: the top or data axis
        """

        return self._data_axis

    @property
    def residual_axis(self):
        """
        :return: the bottom or residual axis
        """


        assert self._show_residuals, 'this plot has no residual axis'

        return self._residual_axis

    @property
    def show_residuals(self):
        return self._show_residuals

    @property
    def ratio_residuals(self):
        return self._ratio_residuals

    def add_model_step(self, xmin, xmax, xwidth, y, label, color):
        """
        Add a model but use discontinuous steps for the plotting.
        :param xmin: the low end boundaries
        :param xmax: the high end boundaries
        :param xwidth: the width of the bins
        :param y: the height of the bins
        :param label: the label of the model
        :param color: the color of the model
        :return: None
        """
        step_plot(np.asarray(zip(xmin, xmax)),
                  y / xwidth,
                  self._data_axis,
                  alpha=.8,
                  label=label,
                  color=color)

    def add_vertical_line(self, grb_triggers, time_ref):
        """

        :param grb_triggers:
        :param time_ref:
        :return:
        """

        for key, value in grb_triggers.iteritems():
            self._data_axis.axvline(x=value['met'] - time_ref, color=value['color'], alpha=0.3, label=key)


    def add_occ_region(self, occ_region, time_ref):
        """

        :param occ_region:
        :param time_ref:
        :return:
        """

        for key, value in occ_region.iteritems():

            self._data_axis.axvspan(xmin=value['met'][0] - time_ref,
                                    xmax=value['met'][1] - time_ref,
                                    color=value['color'], alpha=0.1, label=key)


    def add_model(self, x, y, label, color):
        """
        Add a model and interpolate it across the time span for the plotting.
        :param x: the evaluation energies
        :param y: the model values
        :param label: the label of the model
        :param color: the color of the model
        :return: None
        """
        self._data_axis.plot(x, y, label=label, color=color, alpha=.6, zorder=20)

    def add_list_of_sources(self, x, source_list):
        """
         Add a list of model sources and interpolate them across the time span for the plotting.
         :param x: the evaluation energies
         :param y: the model values
         :param label: the label of the model
         :param color: the color of the model
         :return: None
         """
        for i, source in enumerate(source_list):

            self._data_axis.plot(x, source['data'], color=source['color'], label=source['label'], alpha=.6, zorder=18)



    def add_data(self, x, y, residuals, label, xerr=None, yerr=None, residual_yerr=None, color='r', show_data=True, marker_size=3):
        """
        Add the data for the this model
        :param x: energy of the data
        :param y: value of the data
        :param residuals: the residuals for the data
        :param label: label of the data
        :param xerr: the error in energy (or bin width)
        :param yerr: the errorbars of the data
        :param color: color of the
        :return:
        """


        # if we want to show the data

        if show_data:
            self._data_axis.errorbar(x,
                                     y,
                                     yerr=yerr,
                                     xerr=xerr,
                                     fmt='.',
                                     markersize=marker_size,
                                     linestyle='',
                                     elinewidth=1,
                                     alpha=.9,
                                     capsize=0,
                                     label=label,
                                     color=color)


        # if we want to show the residuals

        if self._show_residuals:

            # normal residuals from the likelihood

            if not self.ratio_residuals:

                residual_yerr = np.ones_like(residuals)

            self._residual_axis.axhline(0, linestyle='--', color='k')


            self._residual_axis.errorbar(x,
                                         residuals,
                                         yerr=residual_yerr,
                                         capsize=0,
                                         fmt='.',
                                         elinewidth=1,
                                         markersize=3,
                                         color=color)


    def finalize(self, xlabel='x', ylabel='y',xscale='log',yscale='log', show_legend=True,invert_y=False):
        """
        :param xlabel:
        :param ylabel:
        :param xscale:
        :param yscale:
        :param show_legend:
        :return:
        """


        if show_legend:
            self._data_axis.legend(fontsize='x-small', loc=0)

        self._data_axis.set_ylabel(ylabel)

        self._data_axis.set_xscale(xscale)
        if yscale == 'log':

            self._data_axis.set_yscale(yscale, nonposy='clip')

        else:

            self._data_axis.set_yscale(yscale)

        if self._show_residuals:

            self._residual_axis.set_xscale(xscale)

            locator = MaxNLocator(prune='upper', nbins=5)
            self._residual_axis.yaxis.set_major_locator(locator)

            self._residual_axis.set_xlabel(xlabel)

            if self.ratio_residuals:
                custom_warnings.warn("Residuals plotted as ratios: beware that they are not statistical quantites, and can not be used to asses fit quality")
                self._residual_axis.set_ylabel("Residuals\n(fraction of model)")
            else:
                self._residual_axis.set_ylabel("Residuals\n($\sigma$)")


        else:

            self._data_axis.set_xlabel(xlabel)




            # This takes care of making space for all labels around the figure

        self._fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        self._fig.subplots_adjust(hspace=0)

        if invert_y:
            self._data_axis.set_ylim(self._data_axis.get_ylim()[::-1])


        return self._fig
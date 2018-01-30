import numpy as np
import matplotlib.pyplot as plt


def step_plot(xbins, y, ax, fill=False, fill_min=0, **kwargs):
    """
    Routine for plotting a in steps with the ability to fill the plot
    xbins is a 2D list of start and stop values.
    y are the values in the bins.
    """

    if fill:

        x = []
        newy = []

        for t, v in zip(xbins, y):
            x.append(t[0])
            newy.append(v)
            x.append(t[1])
            newy.append(v)

        ax.fill_between(x, newy, fill_min, **kwargs)

    else:

        # This supports a mask, so the line will not be drawn for missing bins

        new_x = []
        new_y = []

        for (x1, x2), y in zip(xbins, y):

            if len(new_x) == 0:

                # First iteration

                new_x.append(x1)
                new_x.append(x2)
                new_y.append(y)

            else:

                if x1 == new_x[-1]:

                    # This bin is contiguous to the previous one

                    new_x.append(x2)
                    new_y.append(y)

                else:

                    # This bin is not contiguous to the previous one
                    # Add a "missing bin"
                    new_x.append(x1)
                    new_y.append(np.nan)
                    new_x.append(x2)
                    new_y.append(y)

        new_y.append(new_y[-1])

        new_y = np.ma.masked_where(~np.isfinite(new_y), new_y)

        ax.step(new_x, new_y, where='post', **kwargs)


def disjoint_patch_plot(ax, bin_min, bin_max, top, bottom, mask, **kwargs):
    # type: (plt.Axes, np.array, np.array, float, float, np.array, dict) -> None
    """
    plots patches that are disjoint given by the mask
    :param ax: matplotlib Axes to plot to
    :param bin_min: bin starts
    :param bin_max: bin stops
    :param top: top y value to plot
    :param bottom: bottom y value to plot
    :param mask: mask of the bins
    :param kwargs: matplotlib plot keywords
    :return:
    """
    # Figure out the best limit

    # Find the contiguous regions that are selected


    non_zero = (mask).nonzero()[0]

    if len(non_zero) > 0:

        slices = slice_disjoint(non_zero)

        for region in slices:
            ax.fill_between([bin_min[region[0]], bin_max[region[1]]],
                            bottom,
                            top,
                            **kwargs)

    ax.set_ylim(bottom, top)




def slice_disjoint(arr):
    """
    Returns an array of disjoint indices from a bool array
    :param arr: and array of bools
    """

    slices = []
    start_slice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices


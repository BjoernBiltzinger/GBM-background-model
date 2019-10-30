####################################################################################################
################ Config file for plotting ############### ##########################################
####################################################################################################


############################### Input for plotting ##################################################
### bin_width to bin the data
### change_time: from MET to seconds since midnight
### show residuals?
### show data?
### show grb trigger times? if yes at which time ('20:57:03.000' format) and what name?
### show ppc?
### xlim and ylim (as tuple e.g. (0,100))
### legend outside of the plot?
### dpi of plot
### mpl_style: path to custom style file
#####################################################################################################


plot_config = {
    'data_path':        'path_to_data_for_plots_file.hdf5',
    'bin_width':        5,
    'change_time':      True,
    'xlim':             None,
    'ylim':             None,
    'dpi':              400,
    'show_legend':      True,
    'legend_outside':   True,
}

component_config = {
    'show_data':        True,
    'show_model':       True,
    'show_residuals':   False,
    'show_ppc':         True,

    'show_all_sources': True,
    'show_earth':       True,
    'show_cgb':         True,
    'show_sun':         True,
    'show_saa':         True,
    'show_cr':          True,
    'show_constant':    True,
    'show_crab':        True,

    'show_occ_region':  False,
    'show_grb_trigger': False,
}

style_config = {
    'mpl_style':    None,
    'model':        {'color': 'red',        'alpha': .9},
    'data':         {'color': 'k',          'alpha': .8},
    'sources': {
        'earth':    {'color': 'navy',       'alpha': .6},
        'cgb':      {'color': 'magenta',    'alpha': .6},
        'sun':      {'color': 'gold',       'alpha': .6},
        'saa':      {'color': 'saddlebrown','alpha': .6},
        'cr':       {'color': 'magenta',    'alpha': .6},
        'constant': {'color': 'salmon',     'alpha': .6},
        'crab':     {'color': 'cyan',       'alpha': .6},
        'default':  {'color': 'blue',       'alpha': .6},
        'global':   {'cmap':  'viridis',    'alpha': .6},
        'use_global': True,
    },

    'ppc': {
        'color':    ['lightgreen', 'green', 'darkgreen'],
        'alpha':    0.6
    },
}
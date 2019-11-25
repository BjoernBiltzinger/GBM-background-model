####################################################################################################
################ Config file for plotting ##########################################################
####################################################################################################


############################### Input for plotting ##################################################
### bin_width to bin the data
### change_time: from MET to seconds since midnight
### show grb trigger times? if yes at which time ('20:57:03.000' format) and what name?
### xlim and ylim (as tuple e.g. (0,100))
### legend outside of the plot?
### dpi of plot
### mpl_style: path to custom style file
###
### Optional Highlight config:
### highlight_config = {
###    'grb_trigger':  None,
###    'occ_region':   None
### }
#####################################################################################################


plot_config = {
    'data_path':            'path_to_data_for_plots_file.hdf5',
    'bin_width':            5,
    'time_since_midnight':  True,
    'time_format':          'h',
    'time_t0':              None,
    'xlim':                 None,
    'ylim':                 None,
    'xlabel':               None,
    'ylabel':               None,
    'dpi':                  400,
    'show_legend':          True,
    'legend_outside':       False,
    'show_title':           True,
    'axis_title':           None,
}

component_config = {
    'show_data':            True,
    'show_model':           True,
    'show_residuals':       False,
    'show_ppc':             True,

    'show_all_sources':     True,
    'show_earth':           True,
    'show_cgb':             True,
    'show_sun':             True,
    'show_saa':             True,
    'show_cr':              True,
    'show_constant':        True,
    'show_crab':            True,

    'show_occ_region':      False,
    'show_grb_trigger':     False,
}

style_config = {
    'mpl_style':    'aa.mplstyle',
    'model':        {'color': 'firebrick',                      'alpha': .8, 'linewidth': 0.6, 'show_label': True},
    'data':         {'color': 'k',                              'alpha': .5, 'linewidth': 0.1, 'marker_size': 1, 'elinewidth': 0.2, 'show_label': True},
    'sources': {
        'cr':       {'color': [0.267004, 0.004874, 0.329415],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'earth':    {'color': [0.267968, 0.223549, 0.512008],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'cgb':      {'color': [0.190631, 0.407061, 0.556089],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'constant': {'color': [0.127568, 0.566949, 0.550556],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'saa':      {'color': [0.20803,  0.718701, 0.472873],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'crab':     {'color': [0.565498, 0.84243,  0.262877],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'sun':      {'color': [0.993248, 0.906157, 0.143936],   'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'default':  {'color': 'blue',                           'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'global':   {'cmap':  'viridis',                        'alpha': .6, 'linewidth': 0.8, 'show_label': True},
        'use_global': False,
    },

    'ppc': {
        'color':    ['lightgreen', 'green', 'darkgreen'],
        'alpha':    0.6
    },
    'legend_kwargs': {
        'loc': 'center left',
        'bbox_to_anchor': (-0.17, -0.5),
        'ncol': 3
    }
}

highlight_config = {
    'grb_trigger':  [
        {'name': 'GRB 150126 - KONUS-WIND Trigger',
         'trigger_time': '2015-01-26T20:51:43.524',
         'time_format': 'UTC',
         'time_offset': 0.,
         'color': 'b',
         'linestyle': '-',
         'linewidth': 0.8
         },
    ],
    'occ_region':   [
        {'name': 'Earth occultation',
         'time_start': '2015-01-26T00:00:00.000',
         'time_stop': '2015-01-26T20:50:00.000',
         'time_format': 'UTC',
         'color': 'grey'
         },
    ]
}

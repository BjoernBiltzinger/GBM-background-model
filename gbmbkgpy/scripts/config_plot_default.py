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
    'model':        {'color': 'firebrick',  'alpha': .9},
    'data':         {'color': 'k',          'alpha': .5},
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

highlight_config = {
    'grb_trigger':  [
        {'name': 'GRB 150126 - KONUS-WIND Trigger',
         'trigger_time': '2015-01-26T20:51:43.524',
         'time_format': 'UTC',
         'time_offset': 0.,
         'color': 'b'
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
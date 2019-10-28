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


plot_dict = {
    'data_path': '',
    'bin_width': 30,
    'change_time': True,
    'show_residuals': False,
    'show_data': True,
    'show_sources': True,
    'show_ppc': True,
    'show_grb_trigger': True,
    'times_mark': [],
    'names_mark': [],
    'xlim': None,
    'ylim': None,
    'legend_outside': False,
    'dpi': 400,
    'mpl_style': None,
}

color_dict = {
    'model_color': 'red',
    'source_colors': ['navy', 'magenta', 'cyan', 'salmon', 'saddlebrown', 'gold', 'navy', 'magenta', 'cyan', 'salmon', 'saddlebrown', 'gold'],
    'data_color': 'k',
    'ppc_colors': ['lightgreen', 'green', 'darkgreen']
}

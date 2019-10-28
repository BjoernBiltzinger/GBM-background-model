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
    'data_path': 'path_to_data_for_plots_file.hdf5',
    'bin_width': 5,
    'change_time': True,
    'xlim': None,
    'ylim': None,
    'dpi': 400,
    'legend_outside': True,
    'mpl_style': None,

    'show_residuals': False,
    'show_data': True,
    'show_sources': True,
    'show_model': True,
    'show_ppc': True,
    'show_legend': True,
    'show_occ_region': False,
    'show_grb_trigger': False,

}

color_dict = {
    'model_color': 'red',
    'source_colors': ['navy', 'magenta', 'cyan', 'salmon', 'saddlebrown', 'gold', 'navy', 'magenta', 'cyan', 'salmon', 'saddlebrown', 'gold'],
    'data_color': 'k',
    'ppc_colors': ['lightgreen', 'green', 'darkgreen']
}


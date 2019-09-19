####################################################################################################
################ Config file for all the input parameters ##########################################
####################################################################################################



################## General Input (Days, Detector, Data Type, Energy Channels) #######################

general_dict = {'dates': ['160310', '160311', '160312','160313','160314','160315', '160316', '160317','160318','160319'],
               'detector': 'nb',
               'data_type': 'cspec',
               'echan_list': [5, 15,25,35,45,55, 85, 100]}

################# Input for response precalculation (how many grid poits?) ##########################

response_dict = {'Ngrid': 40000}

####### Input for SAA mask precaluclation (time to exclude after SAA, delete short time intervals? ##

saa_dict = {'time_after_SAA': 5000,
            'short_time_intervals': False}

###### Input for geometry calculation (n time bins per day to calculate the geometry ################

geom_dict = {'n_bins_to_calculate': 800}

##### Input for source Setup (use CR, use Earth, use CGB, point source list, fix earth, fix cgb #####

setup_dict = {'use_SAA': False,
              'use_CR': True,
              'use_Earth': True,
              'use_CGB': True,
              'ps_list': ['CRAB'],
              'fix_ps': [False],
              'fix_earth': False,
              'fix_cgb': False}

################################ Bounds for the different sources ###################################
####### SAA: Amplitude and decay constant, CR: Constant and McIlwain normalization ##################
####### Point source: Amplitude, Earth/CGB fixec: Amplitude, Earth/CGB free: Amplitude, #############
############################ index1, index2 and break energy#########################################

bounds_dict = {'saa_bound': [(1, 10**4), (10**-5, 10**-1)],
               'cr_bound': [(0.1, 100), (0.1, 100)],
               'earth_fixed_bound': [(0.001, 1)],
               'cgb_fixed_bound': [(0.01, 0.5)],
               'earth_free_bound': [(0.001, 1), (-8, -3), (1.1, 1.9), (20, 40)],
               'cgb_free_bound': [(0.01, 0.5), (0.5, 1.7), (2.2, 3.1), (27, 40)],
               'ps_fixed_bound': [(5, 100)],
               'ps_free_bound': [(1, 10), (2, 2.3)]}

##################### Input for multinest sampler ###################################################

multi_dict = {'num_live_points': 400,
              'constant_efficiency_mode': True}

############################### Input for plotting ##################################################
### bin_width to bin the data, change_time from MET to seconds since midnight, show residuals? ######
### show data?, show grb trigger times? if yes at which time ('20:57:03.000' format) and what name? #
### show ppc?, xlim and ylim (as tuple e.g. (0,100)), legend outside of the plot? ####################


plot_dict = {'bin_width': 30,
             'change_time': True,
             'show_residuals': False,
             'show_data': True,
             'plot_sources': True,
             'show_grb_trigger': True,
             'times_mark': [],
             'names_mark': [],
             'ppc': True,
             'xlim': None,
             'ylim': None,
             'legend_outside': False}

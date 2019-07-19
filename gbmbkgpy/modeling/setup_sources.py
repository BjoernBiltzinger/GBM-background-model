
from gbmbkgpy.modeling.source import Source, ContinuumSource, FlareSource, PointSource, SAASource, GlobalSource, FitSpectrumSource
from gbmbkgpy.modeling.function import Function, ContinuumFunction
from gbmbkgpy.modeling.functions import (Solar_Flare, Solar_Continuum, SAA_Decay,
                                         Magnetic_Continuum, Cosmic_Gamma_Ray_Background, Point_Source_Continuum, Earth_Albedo_Continuum, offset, Magnetic_Secondary_Continuum,West_Effect_Continuum, Magnetic_Continuum_Global, Magnetic_Constant_Global, Earth_Albedo_Continuum_Fit_Spectrum, Cosmic_Gamma_Ray_Background_Fit_Spectrum,SAA_Decay_Linear)

from gbmbkgpy.modeling.model_cr_asymmetrie import cr_asymmetrie_nb_n5

import numpy as np


    # see if we have mpi and/or are upalsing parallel                                                                                                                                                                                                                          
try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities                                                                                                                                                                                                             
        using_mpi = True ###################33                                                                                                                                                                                                                                 

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False
def setup_sources(cd, ep, echan):
    """
    Instantiate all Source Object included in the model and return them as an array
    :param echan:
    :param cd: ContinuousData object
    :param ep: ExternalProps object
    :return:
    """

    ## LAT ACD STUFF ##################
    #ep_mask = ep.acd_saa_mask(cd.mean_time[2:-2])
    ##################################
    
    
    # SAA Decay Source
    SAA_Decay_list = []
    if cd.use_SAA:
        saa_n = 0
        day_start = np.array(cd._day_met)
        start_times = np.append(day_start, cd.saa_mean_times)

        for time in start_times:
            saa_dec = SAA_Decay(str(saa_n), str(echan))
            saa_dec.set_saa_exit_time(np.array([time]))
            saa_dec.set_time_bins(cd.time_bins[2:-2])
            # precalculation for later evaluation
            saa_dec.precalulate_time_bins_integral()
            SAA_Decay_list.append(SAASource('saa_{:d} echan_{}'.format(saa_n,echan), saa_dec, echan))
            saa_n += 1
    if echan==0:
        # Solar Continuum Source
        sol_con = Solar_Continuum(str(echan))
        sol_con.set_function_array(cd.effective_angle(cd.sun_angle(cd.time_bins[2:-2]), echan))
        sol_con.set_saa_zero(cd.saa_mask[2:-2])
        sol_con.remove_vertical_movement()
        # precalculate the integration over the time bins
        sol_con.integrate_array(cd.time_bins[2:-2])
        Source_Solar_Continuum = ContinuumSource('Sun effective angle_echan_{:d}'.format(echan), sol_con, echan)

    if not cd.use_SAA:
        # add -b*t Term for long decay during non SAA passages
        saa_dec_long = SAA_Decay_Linear(str(echan))
        saa_dec_long.set_function_array(-1*(cd.time_bins[2:-2]-cd.time_bins[2,0]))
        saa_dec_long.set_saa_zero(cd.saa_mask[2:-2]) 
        saa_dec_long.integrate_array(cd.time_bins[2:-2])
        print(-1*(cd.time_bins[2:-2]-cd.time_bins[2,0]))
        SAA_Decay_list = [ContinuumSource('SAA long decay echan_{}'.format(echan), saa_dec_long, echan)]
    # Magnetic Continuum Source
    mag_con = Magnetic_Continuum(str(echan))

    print(cd.time_bins[2:-2:100][:,0].shape)
    
    
    #east = cd.ned_east(cd.time_bins[2:-2])
    #down = cd.ned_down(cd.time_bins[2:-2])
    #print(cr_asymmetrie_nb_n5(east,down)[::100][:,0].shape)
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(cd.time_bins[2:-2:100][:,0], cr_asymmetrie_nb_n5(east,down)[::100][:,0], label='asy', c='magenta', s=0.5)
    #ax.scatter(cd.time_bins[2:-2:100][:,0], ep.mc_l((cd.time_bins[2:-2]))[::100][:,0], label='L-param', c='cyan', s=0.5)
    #ax.legend()
    #fig.colorbar(sc)
    #fig.savefig('/home/bbiltzing/testing_setup_source.pdf')
    #print(ep.mc_l((cd.time_bins[2:-2]))[0], cr_asymmetrie_nb_n5(east,down)[0], (ep.mc_l((cd.time_bins[2:-2]))*cr_asymmetrie_nb_n5(east,down))[0])
    #mag_con.set_function_array(ep.mc_l((cd.time_bins[2:-2]))/cr_asymmetrie_nb_n5(east,down))
    mag_con.set_function_array(ep.mc_l((cd.time_bins[2:-2])))
    mag_con.set_saa_zero(cd.saa_mask[2:-2])
    mag_con.remove_vertical_movement()
    # precalculate the integration over the time bins
    mag_con.integrate_array(cd.time_bins[2:-2])
    Source_Magnetic_Continuum = ContinuumSource('McIlwain L-parameter_echan_{:d}'.format(echan), mag_con, echan)

    ####################################################
    try:
        # asymmetry source
        east = cd.ned_east(cd.time_bins[2:-2])                                                                                          
        down = cd.ned_down(cd.time_bins[2:-2])
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax =fig.add_subplot(111)
        ax.scatter(cd.time_bins[2:-2][:,0], (ep.mc_l((cd.time_bins[2:-2]))/cr_asymmetrie_nb_n5(east,down))[:,0], s=0.5)
        fig.savefig('testtest.pdf')
        mag_con_a = Magnetic_Continuum(str(echan)+'_asym')
        #mag_con_a.set_function_array(cr_asymmetrie_nb_n5(east,down))
        mag_con_a.set_function_array(ep.mc_l((cd.time_bins[2:-2]))*cr_asymmetrie_nb_n5(east,down))
        mag_con_a.set_saa_zero(cd.saa_mask[2:-2])
        mag_con_a.remove_vertical_movement()
        mag_con_a.integrate_array(cd.time_bins[2:-2])
        Asymmetry_Magnetic_Continuum = ContinuumSource('Assymetry mag_echan_{:d}'.format(echan), mag_con_a, echan)
    except Exception as e:
        print(e)
        print('NO ASYMMETRY')
    #independent two mag_con
    mag_con_1 = Magnetic_Continuum(str(echan)+'.1')
    mask_1 = cd.west_angle(cd.time_bins[2:-2])<90
    mask_2 = cd.west_angle(cd.time_bins[2:-2])>90
    mag_con_1.set_function_array(ep.mc_l((cd.time_bins[2:-2])))
    mag_con_1.set_saa_zero(mask_1)
    mag_con_1.remove_vertical_movement()
    # precalculate the integration over the time bins                                                                                                                                                                                                                          
    mag_con_1.integrate_array(cd.time_bins[2:-2])
    Source_Magnetic_Continuum_1 = ContinuumSource('McIlwain L-parameter_echan_{:d}_1'.format(echan), mag_con_1, echan)

    mag_con_2 = Magnetic_Continuum(str(echan)+'.2')
    mag_con_2.set_function_array(ep.mc_l((cd.time_bins[2:-2])))
    mag_con_2.set_saa_zero(mask_2)
    mag_con_2.remove_vertical_movement()
    # precalculate the integration over the time bins                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                               
    mag_con_2.integrate_array(cd.time_bins[2:-2])
    Source_Magnetic_Continuum_2 = ContinuumSource('McIlwain L-parameter_echan_{:d}_2'.format(echan), mag_con_2, echan)
    
    ##################################################
    #constant term
    
    Constant = offset(str(echan))
    Constant.set_function_array(cd.cgb_background(cd.time_bins[2:-2]))
    Constant.set_saa_zero(cd.saa_mask[2:-2])
    # precalculate the integration over the time bins
    Constant.integrate_array(cd.time_bins[2:-2])
    Constant_Continuum = ContinuumSource('Constant_echan_{:d}'.format(echan), Constant, echan)
    try:
        Constant_a = offset(str(echan)+'_asym')
        Constant_a.set_function_array(cd.cgb_background(cd.time_bins[2:-2])*cr_asymmetrie_nb_n5(east,down))
        Constant_a.set_saa_zero(cd.saa_mask[2:-2])
        # precalculate the integration over the time bins
        Constant_a.integrate_array(cd.time_bins[2:-2])
        Asymmetry_Constant_Continuum = ContinuumSource('Asym Constant_echan_{:d}'.format(echan), Constant_a, echan)
    except:
        print('NO ASYMMETRY')
    #################################################

    #independent const_term
    Constant_1 = offset(str(echan)+'.1')
    Constant_1.set_function_array(cd.cgb_background(cd.time_bins[2:-2]))
    Constant_1.set_saa_zero(mask_1)
    # precalculate the integration over the time bins                                                                                                                                                                                                                          
    Constant_1.integrate_array(cd.time_bins[2:-2])
    Constant_Continuum_1 = ContinuumSource('Constant_echan_{:d}_1'.format(echan), Constant_1, echan)

    Constant_2 = offset(str(echan)+'.2')
    Constant_2.set_function_array(cd.cgb_background(cd.time_bins[2:-2]))
    Constant_2.set_saa_zero(mask_2)
    # precalculate the integration over the time bins                                                                                                                                                                                                                          
    Constant_2.integrate_array(cd.time_bins[2:-2])
    Constant_Continuum_2 = ContinuumSource('Constant_echan_{:d}_2'.format(echan), Constant_2, echan)
    
    ################################################

    ###############################################
    try:
        # LAT ACD rate
        mag_con_lat = Magnetic_Continuum(str(echan)+'.1')
        mag_con_lat.set_function_array(ep.lat_acd(cd.time_bins[2:-2], use_side = 'A'))
        mag_con_lat.set_saa_zero(cd.saa_mask[2:-2])
        mag_con_lat.remove_vertical_movement()
        # precalculate the integration over the time bins                                                                                                                                                                                                                          
        mag_con_lat.integrate_array(cd.time_bins[2:-2])
        Source_Magnetic_Continuum_LAT_ACD = ContinuumSource('McIlwain L-parameter_echan_{:d}'.format(echan), mag_con_lat, echan)
    except:
        print('NO ASYMMETRY')
    ############################# WEST effect                                                                                                                                                                                                                                  
    #west_effect = West_Effect_Continuum(str(echan))
    #west_effect.set_function_array(cd.west(cd.time_bins[2:-2]))
    #west_effect.set_saa_zero(cd.saa_mask[2:-2])
    #west_effect.integrate_array(cd.time_bins[2:-2])
    #West_Continuum = ContinuumSource('West_echan_{:d}'.format(echan), west_effect, echan)

    ############################# 
    #source_list = SAA_Decay_list
    source_list = [Source_Magnetic_Continuum, Constant_Continuum] #+ SAA_Decay_list
    #source_list = [Source_Magnetic_Continuum, Asymmetry_Magnetic_Continuum, Constant_Continuum,Asymmetry_Constant_Continuum] + SAA_Decay_list
    #source_list = [Source_Magnetic_Continuum_LAT_ACD, Constant_Continuum] #+ SAA_Decay_list
    #source_list = [Source_Magnetic_Continuum_1, Source_Magnetic_Continuum_2 , Constant_Continuum_1, Constant_Continuum_2] + SAA_Decay_list
    if echan==0:
        source_list.append(Source_Solar_Continuum)

    return source_list

def setup_sources_golbal(cd, ep, include_point_sources=False, point_source_list=[]):
    """
    set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param cd:
    :param echan:
    :return:
    """
    assert len(point_source_list)==0 or include_point_sources==False, 'Either include all point sources ' \
                                                                          'or give a list with the wanted sources.' \
                                                                          'Not both!'
    # Earth Albedo Continuum Source
    earth_albedo = Earth_Albedo_Continuum()

    ###Testing rigidity dependence
    #rigidity_array = (14.823/((ep.mc_l(cd.time_bins[2:-2]))**2))**(-1.13)
    ##
    
    earth_albedo.set_function_array(cd.earth_rate_array(cd.time_bins[2:-2]))#*rigidity_array)
    earth_albedo.set_saa_zero(cd.saa_mask[2:-2])
    #earth_albedo.set_base_function_all_times(cd.point_base_rate(cd.time_bins[2:-2]))#
    #earth_albedo.set_angle_of_points_all_times(cd.point_earth_angle(cd.time_bins[2:-2]))#
    # precalculate the integration over the time bins
    earth_albedo.integrate_array(cd.time_bins[2:-2])
    Source_Earth_Albedo_Continuum = GlobalSource('Earth occultation', earth_albedo)

    # Cosmic gamma-ray background
    cgb = Cosmic_Gamma_Ray_Background()
    cgb.set_function_array(cd.cgb_rate_array(cd.time_bins[2:-2]))
    cgb.set_saa_zero(cd.saa_mask[2:-2])
    cgb.integrate_array(cd.time_bins[2:-2])
    Source_CGB_Continuum = GlobalSource('CGB', cgb)

    
    PS_Sources_list = []
    # Point-Source Sources
    if include_point_sources:
        ep.build_point_sources(cd)
        PS_Continuum_dic = {}
        PS_Sources_list = []

        for i, ps in enumerate(ep.point_sources.itervalues()):
            PS_Continuum_dic['{}'.format(ps.name)] = Point_Source_Continuum(str(i))
            PS_Continuum_dic['{}'.format(ps.name)].set_function_array(ps.ps_rate_array(cd.time_bins[2:-2]))
            PS_Continuum_dic['{}'.format(ps.name)].set_saa_zero(cd.saa_mask[2:-2])
            PS_Continuum_dic['{}'.format(ps.name)].integrate_array(cd.time_bins[2:-2])

            PS_Sources_list.append(GlobalSource('{}'.format(ps.name), PS_Continuum_dic['{}'.format(ps.name)]))
    if len(point_source_list)>0:
        ep.build_some_source(cd, point_source_list)
        PS_Continuum_dic = {}
        PS_Sources_list = []
        for i, ps in enumerate(ep.point_sources.itervalues()):
            PS_Continuum_dic['{}'.format(ps.name)] = Point_Source_Continuum(str(i))
            #PS_Continuum_dic[ps.name].set_function_array(cd.effective_angle(ps.calc_occ_array(cd.time_bins[2:-2]),
            #                                                                echan))
            PS_Continuum_dic['{}'.format(ps.name)].set_function_array(ps.ps_rate_array(cd.time_bins[2:-2]))
            PS_Continuum_dic['{}'.format(ps.name)].set_saa_zero(cd.saa_mask[2:-2])
            #precalculate the integration over the time bins
            PS_Continuum_dic['{}'.format(ps.name)].integrate_array(cd.time_bins[2:-2])
            #PS_Continuum_dic[ps.name].set_earth_zero(ps.earth_occ_of_ps(cd.mean_time[2:-2]))

            PS_Sources_list.append(GlobalSource('{}'.format(ps.name), PS_Continuum_dic['{}'.format(ps.name)]))

    ###############################################
    # LAT ACD rate
    #mag_con_lat = Magnetic_Continuum_Global()
    #function_array = []
    #basis = ep.lat_acd(cd.time_bins[2:-2], use_side = 'C')
    #for i in range(8):
    #    function_array.append(cd.ebins_size[i]*basis)
    #mag_con_lat.set_function_array(np.array(function_array))
    #mag_con_lat.set_saa_zero(cd.saa_mask[2:-2])
    ##mag_con_lat.remove_vertical_movement()
    # precalculate the integration over the time bins
    #mag_con_lat.integrate_array(cd.time_bins[2:-2])
    #Source_Magnetic_Continuum_LAT_ACD = GlobalSource('ACD rate', mag_con_lat)

    # LAT ACD rate
    #mag_con_lat = Magnetic_Constant_Global()
    #function_array = []
    #basis = np.ones_like(cd.time_bins[2:-2])
    #for i in range(8):
    #    function_array.append(cd.ebins_size[i]*basis)
    #mag_con_lat.set_function_array(np.array(function_array))
    #mag_con_lat.set_saa_zero(cd.saa_mask[2:-2])
    #mag_con_lat.remove_vertical_movement()
    # precalculate the integration over the time bins
    #mag_con_lat.integrate_array(cd.time_bins[2:-2])
    #Source_Magnetic_Constant = GlobalSource('CR constant', mag_con_lat)

    
    source_list = [Source_Earth_Albedo_Continuum, Source_CGB_Continuum] + PS_Sources_list
    #source_list = [Source_Earth_Albedo_Continuum, Source_CGB_Continuum, Source_Magnetic_Continuum_LAT_ACD, Source_Magnetic_Constant] + PS_Sources_list
    #source_list = PS_Sources_list
    return source_list

def setup_sources_fit_spectra(cd):
    """
    set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param cd:
    :param echan:
    :return:
    """

    # Earth Albedo Continuum Source
    earth_albedo = Earth_Albedo_Continuum_Fit_Spectrum()
    earth_albedo.set_response_array(cd.response_array_earth) #???
    earth_albedo.set_basis_function_array(cd.time_bins[2:-2])
    earth_albedo.set_saa_zero(cd.saa_mask[2:-2])
    earth_albedo.set_interpolation_times(cd.interpolation_time)
    earth_albedo.energy_boundaries(cd.Ebin_source)
    Source_Earth_Albedo_Continuum = FitSpectrumSource('Earth occultation', earth_albedo)

    # Cosmic gamma-ray background
    cgb = Cosmic_Gamma_Ray_Background_Fit_Spectrum()
    
    cgb.set_response_array(cd.response_array_cgb) #???                                                                                                                                                       
    cgb.set_basis_function_array(cd.time_bins[2:-2])                                                                                                                                                     
    cgb.set_saa_zero(cd.saa_mask[2:-2])
    cgb.set_interpolation_times(cd.interpolation_time)
    cgb.energy_boundaries(cd.Ebin_source)
    Source_CGB_Continuum = FitSpectrumSource('CGB', cgb)
    
    source_list = [Source_Earth_Albedo_Continuum, Source_CGB_Continuum]
    return source_list

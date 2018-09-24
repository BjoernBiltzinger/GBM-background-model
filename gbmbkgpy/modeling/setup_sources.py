
from gbmbkgpy.modeling.source import Source, ContinuumSource, FlareSource, PointSource, SAASource, GlobalSource
from gbmbkgpy.modeling.function import Function, ContinuumFunction
from gbmbkgpy.modeling.functions import (Solar_Flare, Solar_Continuum, SAA_Decay,
Magnetic_Continuum, Cosmic_Gamma_Ray_Background, Point_Source_Continuum, Earth_Albedo_Continuum, offset)

import numpy as np


def setup_sources(cd, ep, echan, include_point_sources=False, point_source_list=[]):
    """
    Instantiate all Source Object included in the model and return them as an array
    :param echan:
    :param cd: ContinuousData object
    :param ep: ExternalProps object
    :return:
    """
    assert len(point_source_list)==0 or include_point_sources==False, 'Either include all point sources ' \
                                                                          'or give a list with the wanted sources.' \
                                                                          'Not both!'
    PS_Sources_list = []

    # Point-Source Sources
    if include_point_sources:
        ep.build_point_sources(cd)
        PS_Continuum_dic = {}
        PS_Sources_list = []

        for i, ps in enumerate(ep.point_sources.itervalues()):
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)] = Point_Source_Continuum(str(i), str(echan))
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)].set_function_array(ps.ps_rate_array(cd.time_bins[2:-2], echan))
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)].integrate_array(cd.time_bins[2:-2])

            PS_Sources_list.append(PointSource('{}{:d}'.format(ps.name, echan), PS_Continuum_dic['{}{:d}'.format(ps.name, echan)], echan))
    if len(point_source_list)>0:
        ep.build_some_source(cd, point_source_list)
        PS_Continuum_dic = {}
        PS_Sources_list = []

        for i, ps in enumerate(ep.point_sources.itervalues()):
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)] = Point_Source_Continuum(str(i),str(echan))
            #PS_Continuum_dic[ps.name].set_function_array(cd.effective_angle(ps.calc_occ_array(cd.time_bins[2:-2]),
            #                                                                echan))
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)].set_function_array(ps.ps_rate_array(cd.time_bins[2:-2], echan))
            PS_Continuum_dic['{}{:d}'.format(ps.name, echan)].integrate_array(cd.time_bins[2:-2])
            #PS_Continuum_dic[ps.name].set_earth_zero(ps.earth_occ_of_ps(cd.mean_time[2:-2]))

            PS_Sources_list.append(PointSource('{}{:d}'.format(ps.name, echan), PS_Continuum_dic['{}{:d}'.format(ps.name, echan)], echan))
    # SAA Decay Source
    SAA_Decay_list = []
    if cd.use_SAA:
        saa_n = 0
        day_start = np.array(cd._day_met)
        start_times = np.append(day_start, cd.saa_mean_times)

        for time in start_times:
            saa_dec = SAA_Decay(str(saa_n))
            saa_dec.set_saa_exit_time(np.array([time]))
            saa_dec.set_time_bins(cd.time_bins[2:-2])
            SAA_Decay_list.append(SAASource('saa_{:d}'.format(saa_n), saa_dec, echan))
            saa_n += 1
    if echan==0:
        # Solar Continuum Source
        sol_con = Solar_Continuum(str(echan))
        sol_con.set_function_array(cd.effective_angle(cd.sun_angle(cd.time_bins[2:-2]), echan))
        sol_con.set_saa_zero(cd.saa_mask[2:-2])
        sol_con.remove_vertical_movement()
        sol_con.integrate_array(cd.time_bins[2:-2])
        Source_Solar_Continuum = ContinuumSource('Sun effective angle_echan_{:d}'.format(echan), sol_con, echan)

    # Magnetic Continuum Source
    mag_con = Magnetic_Continuum(str(echan))
    mag_con.set_function_array(ep.mc_l((cd.time_bins[2:-2])))
    mag_con.set_saa_zero(cd.saa_mask[2:-2])
    mag_con.remove_vertical_movement()
    mag_con.integrate_array(cd.time_bins[2:-2])
    Source_Magnetic_Continuum = ContinuumSource('McIlwain L-parameter_echan_{:d}'.format(echan), mag_con, echan)

    #constant term
    Constant = offset(str(echan))
    Constant.set_function_array(cd.cgb_background(cd.time_bins[2:-2]))
    Constant.set_saa_zero(cd.saa_mask[2:-2])
    Constant.integrate_array(cd.time_bins[2:-2])
    Constant_Continuum = ContinuumSource('Constant_echan_{:d}'.format(echan), Constant, echan)

    source_list = [Source_Magnetic_Continuum, Constant_Continuum] + SAA_Decay_list + PS_Sources_list

    if echan==0:
        source_list.append(Source_Solar_Continuum)

    return source_list

def setup_sources_golbal(cd):
    """
    set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param cd:
    :param echan:
    :return:
    """

    # Earth Albedo Continuum Source
    earth_albedo = Earth_Albedo_Continuum()
    earth_albedo.set_function_array(cd.earth_rate_array(cd.time_bins[2:-2]))
    earth_albedo.set_saa_zero(cd.saa_mask[2:-2])
    earth_albedo.integrate_array(cd.time_bins[2:-2])
    Source_Earth_Albedo_Continuum = GlobalSource('Earth occultation', earth_albedo)

    # Cosmic gamma-ray background
    cgb = Cosmic_Gamma_Ray_Background()
    cgb.set_function_array(cd.cgb_rate_array(cd.time_bins[2:-2]))
    cgb.set_saa_zero(cd.saa_mask[2:-2])
    cgb.integrate_array(cd.time_bins[2:-2])
    Source_CGB_Continuum = GlobalSource('CGB', cgb)

    source_list = [Source_Earth_Albedo_Continuum, Source_CGB_Continuum]

    return source_list



from gbmbkgpy.modeling.source import Source, ContinuumSource, FlareSource, PointSource, SAASource
from gbmbkgpy.modeling.function import Function, ContinuumFunction
from gbmbkgpy.modeling.functions import (Solar_Flare, Solar_Continuum, SAA_Decay,
Magnetic_Continuum, Cosmic_Gamma_Ray_Background, Point_Source_Continuum, Earth_Albedo_Continuum)

import numpy as np


def setup_sources(cd, ep, echan, include_point_sources=False):
    """
    Instantiate all Source Object included in the model and return them as an array
    :param echan:
    :param cd: ContinuousData object
    :param ep: ExternalProps object
    :return:
    """
    PS_Sources_list = []

    # Point-Source Sources
    if include_point_sources:
        ep.build_point_sources(cd)
        PS_Continuum_dic = {}
        PS_Sources_list = []

        for i, ps in enumerate(ep.point_sources.itervalues()):
            PS_Continuum_dic[ps.name] = Point_Source_Continuum(str(i))
            PS_Continuum_dic[ps.name].set_function_array(cd.effective_angle(ps.calc_occ_array, echan))

            PS_Sources_list.append(PointSource(ps.name, PS_Continuum_dic[ps.name]))

    # SAA Decay Source
    SAA_Decay_list = []
    saa_n = 0
    day_start = np.array(cd._day_met)
    start_times = np.append(day_start, cd.saa_mean_times)

    for time in start_times:
        saa_dec = SAA_Decay(str(saa_n))
        saa_dec.set_saa_exit_time(np.array([time]))
        saa_dec.set_time_bins(cd.time_bins[2:-2])
        SAA_Decay_list.append(SAASource('saa_' + str(saa_n), saa_dec))
        saa_n += 1

    # Solar Continuum Source
    sol_con = Solar_Continuum()
    sol_con.set_function_array(cd.effective_angle(cd.sun_angle(cd.time_bins[2:-2]), echan))
    sol_con.set_saa_zero(cd.saa_mask[2:-2])
    sol_con.remove_vertical_movement()
    Source_Solar_Continuum = ContinuumSource('Sun effective angle', sol_con)

    # Magnetic Continuum Source
    mag_con = Magnetic_Continuum()
    mag_con.set_function_array(ep.mc_l((cd.time_bins[2:-2])))
    mag_con.set_saa_zero(cd.saa_mask[2:-2])
    mag_con.remove_vertical_movement()
    Source_Magnetic_Continuum = ContinuumSource('McIlwain L-parameter', mag_con)

    # Earth Albedo Continuum Source
    earth_albedo = Earth_Albedo_Continuum()
    earth_albedo.set_function_array(cd.effective_area(cd.earth_angle(cd.time_bins[2:-2]), echan))
    earth_albedo.set_saa_zero(cd.saa_mask[2:-2])
    earth_albedo.remove_vertical_movement()
    Source_Earth_Albedo_Continuum = ContinuumSource('Earth occultation', earth_albedo)

    # Cosmic gamma-ray background
    cgb = Cosmic_Gamma_Ray_Background()
    cgb.set_function_array(cd.cgb_background(cd.time_bins[2:-2]))
    cgb.set_saa_zero(cd.saa_mask[2:-2])
    Source_CGB_Continuum = ContinuumSource('CGB', cgb)

    source_list = [Source_CGB_Continuum, Source_Magnetic_Continuum,# Source_Solar_Continuum,
                   Source_Earth_Albedo_Continuum] + SAA_Decay_list + PS_Sources_list

    return source_list

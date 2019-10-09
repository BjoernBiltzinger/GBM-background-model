from gbmbkgpy.modeling.source import ContinuumSource, SAASource, GlobalSource, FitSpectrumSource

from gbmbkgpy.modeling.functions import (SAA_Decay, Magnetic_Continuum, Cosmic_Gamma_Ray_Background,
                                         Point_Source_Continuum, Earth_Albedo_Continuum, Offset,
                                         Earth_Albedo_Continuum_Fit_Spectrum, Cosmic_Gamma_Ray_Background_Fit_Spectrum,
                                         Point_Source_Continuum_Fit_Spectrum)

import numpy as np
from scipy import interpolate

# see if we have mpi and/or are upalsing parallel
try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False


def Setup(data, saa_object, ep, geom_object, echan_list=[], response_object=None, albedo_cgb_object=None, use_SAA=False,
          use_CR=True, use_Earth=True, use_CGB=True, use_all_ps=False, point_source_list=[], fix_ps=[],
          fix_Earth=False, fix_CGB=False):
    """
    Setup all sources
    :param fix_ps:
    :param data: Data object
    :param saa_object: saa precaculation object
    :param ep: external prob object
    :param geom_object: geometry precalculation object
    :param echan_list: list of all echans which should be used
    :param response_object: response precalculation object
    :param albedo_cgb_object: albedo_cgb precalculation object
    :param use_SAA: use saa?
    :param use_CR: use cr?
    :param use_Earth: use earth?
    :param use_CGB: use cgb?
    :param use_all_ps: use all ps?
    :param point_source_list: PS to use
    :param fix_Earth: fix earth spectrum?
    :param fix_CGB: fix cgb spectrum?
    :return:
    """

    assert len(echan_list) > 0, 'Please give at least one echan'

    assert type(use_SAA) == bool and type(use_CR) == bool and type(use_Earth) == bool and type(use_CGB) == bool and \
           type(fix_Earth) == bool and type(fix_CGB) == bool and type(use_all_ps) == bool, 'Please only use True or False here.'

    total_sources = []

    # Go through all possible types of sources and add them in a list

    for index, echan in enumerate(echan_list):

        if use_SAA:
            total_sources.extend(setup_SAA(data, saa_object, echan, index))

        if use_CR:
            total_sources.extend(setup_CosmicRays(data, ep, saa_object, echan, index))

    if use_all_ps:
        total_sources.extend(setup_ps(data, ep, saa_object, response_object, geom_object, echan_list,
                                      include_point_sources=True, free_spectrum=np.logical_not(fix_ps)))

    elif len(point_source_list) != 0:
        total_sources.extend(setup_ps(data, ep, saa_object, response_object, geom_object, echan_list,
                                      point_source_list=point_source_list, free_spectrum=np.logical_not(fix_ps)))

    if use_Earth:

        if fix_Earth:
            total_sources.append(setup_earth_fix(data, albedo_cgb_object, saa_object))

        else:
            total_sources.append(setup_earth_free(data, albedo_cgb_object, saa_object))

    if use_CGB:

        if fix_CGB:
            total_sources.append(setup_cgb_fix(data, albedo_cgb_object, saa_object))

        else:
            total_sources.append(setup_cgb_free(data, albedo_cgb_object, saa_object))

    return total_sources


def setup_SAA(data, saa_object, echan, index):
    """
    Setup for SAA sources
    :param index:
    :param saa_object: SAA precalculation object
    :param echan: energy channel
    :param data: Data object
    :return: List of all SAA decay sources
    """

    # SAA Decay Source
    SAA_Decay_list = []
    saa_n = 0
    # Add 'SAA' decay at start of the day if fitting only one day to account for leftover excitation
    day_start = np.array(data.day_met) if len(data.day_met) <= 1 else []
    start_times = np.append(day_start, saa_object.saa_exit_times)

    for time in start_times:
        saa_dec = SAA_Decay(str(saa_n), str(echan))
        saa_dec.set_saa_exit_time(np.array([time]))
        saa_dec.set_time_bins(data.time_bins[2:-2])

        # precalculation for later evaluation
        saa_dec.precalulate_time_bins_integral()
        SAA_Decay_list.append(SAASource('saa_{:d} echan_{}'.format(saa_n, echan), saa_dec, index))
        saa_n += 1
    return SAA_Decay_list


def setup_CosmicRays(data, ep, saa_object, echan, index):
    """
    Setup for CosmicRay source
    :param index:
    :param saa_object:
    :param ep: external prob object
    :param echan: energy channel
    :param data: Data object
    :return: Constant and magnetic continuum source
    """

    Constant = Offset(str(echan))
    Constant.set_function_array(np.ones_like(data.time_bins[2:-2]))
    Constant.set_saa_zero(saa_object.saa_mask[2:-2])
    # precalculate the integration over the time bins
    Constant.integrate_array(data.time_bins[2:-2])
    Constant_Continuum = ContinuumSource('Constant_echan_{:d}'.format(echan), Constant, index)

    # Magnetic Continuum Source
    mag_con = Magnetic_Continuum(str(echan))
    mag_con.set_function_array(ep.mc_l((data.time_bins[2:-2])))
    mag_con.set_saa_zero(saa_object.saa_mask[2:-2])
    mag_con.remove_vertical_movement()
    # precalculate the integration over the time bins
    mag_con.integrate_array(data.time_bins[2:-2])
    Source_Magnetic_Continuum = ContinuumSource('McIlwain L-parameter_echan_{:d}'.format(echan),
                                                mag_con, index)
    return [Constant_Continuum, Source_Magnetic_Continuum]


def setup_ps(data, ep, saa_object, response_object, geom_object, echan_list,
             include_point_sources=False, point_source_list=[], free_spectrum=[]):
    """
    Set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param include_point_sources:
    :param point_source_list:
    :param free_spectrum:
    :param echan_list:
    :param geom_object:
    :param saa_object:
    :param response_object:
    :param data:
    :return:
    """
    assert len(point_source_list) == 0 or include_point_sources == False, 'Either include all point sources ' \
                                                                          'or give a list with the wanted sources.' \
                                                                          'Not both!'
    if len(free_spectrum) > 0:
        assert len(free_spectrum) == len(point_source_list), 'free_spectrum and point_source_list must have same length'

    PS_Sources_list = []

    # Point-Source Sources
    if include_point_sources:
        ep.build_point_sources(response_object, geom_object, echan_list, free_spectrum=free_spectrum)

    if len(point_source_list) > 0:
        ep.build_some_source(response_object, geom_object, point_source_list, echan_list, free_spectrum=free_spectrum)

    PS_Continuum_dic = {}

    for i, ps in enumerate(ep.point_sources.itervalues()):
        if len(free_spectrum) > 0 and free_spectrum[i]:
            PS_Continuum_dic['{}'.format(ps.name)] = Point_Source_Continuum_Fit_Spectrum(str(i))
            response_array = ps.ps_response_array
            PS_Continuum_dic['{}'.format(ps.name)].set_response_array(response_array)
            PS_Continuum_dic['{}'.format(ps.name)].set_basis_function_array(data.time_bins[2:-2])
            PS_Continuum_dic['{}'.format(ps.name)].set_saa_zero(saa_object.saa_mask[2:-2])
            PS_Continuum_dic['{}'.format(ps.name)].set_interpolation_times(ps.geometry_times)
            PS_Continuum_dic['{}'.format(ps.name)].energy_boundaries(ps.Ebin_in_edge)

            PS_Sources_list.append(FitSpectrumSource('{}'.format(ps.name), PS_Continuum_dic['{}'.format(ps.name)]))
        else:
            PS_Continuum_dic['{}'.format(ps.name)] = Point_Source_Continuum(str(i))
            rate_inter = interpolate.interp1d(ps.geometry_times, ps.ps_rate_array.T)
            PS_Continuum_dic['{}'.format(ps.name)].set_function_array(rate_inter(data.time_bins[2:-2]))
            PS_Continuum_dic['{}'.format(ps.name)].set_saa_zero(saa_object.saa_mask[2:-2])
            PS_Continuum_dic['{}'.format(ps.name)].integrate_array(data.time_bins[2:-2])

            PS_Sources_list.append(GlobalSource('{}'.format(ps.name), PS_Continuum_dic['{}'.format(ps.name)]))

    return PS_Sources_list


def setup_earth_free(data, albedo_cgb_object, saa_object):
    """
    Setup Earth Albedo source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """

    eff_response = albedo_cgb_object.earth_effective_response

    earth_albedo = Earth_Albedo_Continuum_Fit_Spectrum()
    earth_albedo.set_response_array(eff_response)
    earth_albedo.set_basis_function_array(data.time_bins[2:-2])
    earth_albedo.set_saa_zero(saa_object.saa_mask[2:-2])
    earth_albedo.set_interpolation_times(albedo_cgb_object.geometry_times)
    earth_albedo.energy_boundaries(albedo_cgb_object.Ebin_in_edge)
    Source_Earth_Albedo_Continuum = FitSpectrumSource('Earth occultation', earth_albedo)

    return Source_Earth_Albedo_Continuum


def setup_earth_fix(data, albedo_cgb_object, saa_object):
    """
    Setup Earth Albedo source with fixed spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """

    earth_albedo = Earth_Albedo_Continuum()
    rate_inter = interpolate.interp1d(albedo_cgb_object.geometry_times, albedo_cgb_object.earth_rate_array.T)

    earth_albedo.set_function_array(rate_inter(data.time_bins[2:-2]))
    earth_albedo.set_saa_zero(saa_object.saa_mask[2:-2])
    earth_albedo.integrate_array(data.time_bins[2:-2])
    Source_Earth_Albedo_Continuum = GlobalSource('Earth Albedo', earth_albedo)

    return Source_Earth_Albedo_Continuum


def setup_cgb_free(data, albedo_cgb_object, saa_object):
    """
    Setup CGB source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    eff_response = albedo_cgb_object.cgb_effective_response

    cgb = Cosmic_Gamma_Ray_Background_Fit_Spectrum()
    cgb.set_response_array(eff_response)
    cgb.set_basis_function_array(data.time_bins[2:-2])
    cgb.set_saa_zero(saa_object.saa_mask[2:-2])
    cgb.set_interpolation_times(albedo_cgb_object.geometry_times)
    cgb.energy_boundaries(albedo_cgb_object.Ebin_in_edge)
    Source_CGB_Albedo_Continuum = FitSpectrumSource('CGB', cgb)

    return Source_CGB_Albedo_Continuum


def setup_cgb_fix(data, albedo_cgb_object, saa_object):
    """
    Setup CGB source with fixed spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    cgb = Cosmic_Gamma_Ray_Background()
    rate_inter = interpolate.interp1d(albedo_cgb_object.geometry_times, albedo_cgb_object.cgb_rate_array.T)

    cgb.set_function_array(rate_inter(data.time_bins[2:-2]))
    cgb.set_saa_zero(saa_object.saa_mask[2:-2])
    cgb.integrate_array(data.time_bins[2:-2])
    Source_CGB_Albedo_Continuum = GlobalSource('CGB', cgb)

    return Source_CGB_Albedo_Continuum

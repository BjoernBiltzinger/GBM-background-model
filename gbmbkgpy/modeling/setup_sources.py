from gbmbkgpy.modeling.source import (
    ContinuumSource,
    SAASource,
    GlobalSource,
    FitSpectrumSource,
)

from gbmbkgpy.modeling.functions import (
    SAA_Decay,
    Magnetic_Continuum,
    Cosmic_Gamma_Ray_Background,
    Point_Source_Continuum,
    Earth_Albedo_Continuum,
    Offset,
    Earth_Albedo_Continuum_Fit_Spectrum,
    Cosmic_Gamma_Ray_Background_Fit_Spectrum,
    Point_Source_Continuum_Fit_Spectrum,
)

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


def Setup(
    data,
    saa_object,
    ep,
    det_geometries,
    echans=[],
    sun_object=None,
    det_responses=None,
    albedo_cgb_object=None,
    use_saa=False,
    use_constant=True,
    use_cr=True,
    use_earth=True,
    use_cgb=True,
    use_sun=True,
    use_all_ps=False,
    point_source_list=[],
    fix_ps=[],
    fix_earth=False,
    fix_cgb=False,
    nr_saa_decays=1,
    decay_at_day_start=True,
    bgo_cr_approximation=False,
    use_numba=False,
):
    """
    Setup all sources
    :param data: Data object
    :param saa_object: saa precaculation object
    :param ep: external prob object
    :param det_geometries: geometry precalculation object
    :param echan_list: list of all echans which should be used
    :param sun_object:
    :param det_responses: response precalculation object
    :param albedo_cgb_object: albedo_cgb precalculation object
    :param use_saa: use saa?
    :param use_constant:
    :param use_cr: use cr?
    :param use_earth: use earth?
    :param use_cgb: fix cgb spectrum?
    :param use_sun:
    :param use_all_ps: use all ps?
    :param point_source_list: PS to use
    :param fix_ps:
    :param fix_earth: fix earth spectrum?
    :param fix_cgb: use cgb?
    :param nr_saa_decays:
    :param decay_at_day_start:
    :param bgo_cr_approximation: Use bgo cr approximation
    :return:
    """

    assert len(echans) > 0, "Please give at least one echan"

    assert (
        type(use_saa) == bool
        and type(use_cr) == bool
        and type(use_earth) == bool
        and type(use_cgb) == bool
        and type(fix_earth) == bool
        and type(fix_cgb) == bool
        and type(use_all_ps) == bool
    ), "Please only use True or False here."

    total_sources = []

    # Go through all possible types of sources and add them in a list

    for index, echan in enumerate(echans):

        if use_saa:
            total_sources.extend(
                setup_SAA(
                    data, saa_object, echan, index, nr_saa_decays, decay_at_day_start
                )
            )

        if use_constant:
            total_sources.append(setup_Constant(data, saa_object, echan, index))

        if use_cr:
            total_sources.append(
                setup_CosmicRays(
                    data, ep, saa_object, echan, index, bgo_cr_approximation
                )
            )

    if use_sun:
        total_sources.append(
            setup_sun(
                data, sun_object, saa_object, det_responses, det_geometries, echans
            )
        )

    if use_all_ps or len(point_source_list) > 0:
        total_sources.extend(
            setup_ps(
                data=data,
                ep=ep,
                saa_object=saa_object,
                det_responses=det_responses,
                det_geometries=det_geometries,
                echans=echans,
                use_numba=use_numba,
                include_all_ps=use_all_ps,
                point_source_list=point_source_list,
                free_spectrum=np.logical_not(fix_ps),
            )
        )

    if use_earth:

        if fix_earth:
            total_sources.append(setup_earth_fix(data, albedo_cgb_object, saa_object))

        else:
            total_sources.append(
                setup_earth_free(data, albedo_cgb_object, saa_object, use_numba)
            )

    if use_cgb:

        if fix_cgb:
            total_sources.append(setup_cgb_fix(data, albedo_cgb_object, saa_object))

        else:
            total_sources.append(
                setup_cgb_free(data, albedo_cgb_object, saa_object, use_numba)
            )

    return total_sources


def setup_SAA(data, saa_object, echan, index, nr_decays=1, decay_at_day_start=True):
    """
    Setup for SAA sources
    :param decay_at_day_start:
    :param nr_decays: Number of decays that should be fittet to each SAA Exit
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

    day_start = []

    if decay_at_day_start and len(data.day_met) <= 1:
        for i in range(nr_decays):
            day_start.append(data.day_met)

    start_times = np.append(np.array(day_start), saa_object.saa_exit_times)

    for time in start_times:
        saa_dec = SAA_Decay(str(saa_n), str(echan))

        saa_dec.set_saa_exit_time(np.array([time]))

        saa_dec.set_time_bins(data.time_bins)

        saa_dec.set_nr_detectors(len(data._detectors))

        # precalculation for later evaluation
        saa_dec.precalulate_time_bins_integral()

        SAA_Decay_list.append(
            SAASource("saa_{:d} echan_{}".format(saa_n, echan), saa_dec, index)
        )
        saa_n += 1
    return SAA_Decay_list


def setup_sun(cd, sun_object, saa_object, response_object, geom_object, echan_list):
    """
    Setup for sun as bkg source
    """
    Sun = Point_Source_Continuum_Fit_Spectrum("sun")

    Sun.set_response_array(sun_object.sun_response_array)

    Sun.set_basis_function_array(cd.time_bins)

    Sun.set_saa_zero(saa_object.saa_mask)

    Sun.set_interpolation_times(sun_object.geometry_times)

    Sun.energy_boundaries(sun_object.Ebin_in_edge)

    Sun_Continuum = FitSpectrumSource("sun", Sun)

    return Sun_Continuum


def setup_Constant(data, saa_object, echan, index):
    Constant = Offset(str(echan))

    Constant.set_function_array(np.ones((len(data.time_bins), len(data._detectors), 2)))

    Constant.set_saa_zero(saa_object.saa_mask)

    # precalculate the integration over the time bins
    Constant.integrate_array(data.time_bins)

    Constant_Continuum = ContinuumSource(
        "Constant_echan_{:d}".format(echan), Constant, index
    )
    return Constant_Continuum


def setup_CosmicRays(data, ep, saa_object, echan, index, bgo_cr_approximation):
    """
    Setup for CosmicRay source
    :param index:
    :param saa_object:
    :param ep: external prob object
    :param echan: energy channel
    :param data: Data object
    :return: Constant and magnetic continuum source
    """

    if bgo_cr_approximation:
        mag_con = Magnetic_Continuum(str(echan))

        mag_con.set_function_array(ep.bgo_cr_approximation((data.time_bins)))

        mag_con.remove_vertical_movement()

        mag_con.set_saa_zero(saa_object.saa_mask)

        mag_con.integrate_array(data.time_bins)

        Source_Magnetic_Continuum = ContinuumSource(
            "BGO_CR_Approx_echan_{:d}".format(echan), mag_con, index
        )
        return Source_Magnetic_Continuum

    else:
        # Magnetic Continuum Source
        mag_con = Magnetic_Continuum(str(echan))

        mag_con.set_function_array(ep.mc_l_rates((data.time_bins)),)

        mag_con.set_saa_zero(saa_object.saa_mask)

        mag_con.remove_vertical_movement()

        # precalculate the integration over the time bins
        mag_con.integrate_array(data.time_bins)

        Source_Magnetic_Continuum = ContinuumSource(
            "McIlwain_L-parameter_echan_{:d}".format(echan), mag_con, index
        )
        return Source_Magnetic_Continuum


def setup_ps(
    data,
    ep,
    saa_object,
    det_responses,
    det_geometries,
    echans,
    use_numba,
    include_all_ps,
    point_source_list,
    free_spectrum=[],
):
    """
    Set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param include_point_sources:
    :param point_source_list:
    :param free_spectrum:
    :param echan_list:
    :param det_geometries:
    :param saa_object:
    :param det_responses:
    :param data:
    :return:
    """
    if len(free_spectrum) > 0:
        assert len(free_spectrum) == len(
            point_source_list
        ), "free_spectrum and point_source_list must have same length"

    PS_Sources_list = []

    # Point-Source Sources
    ep.build_point_sources(
        det_responses=det_responses,
        det_geometries=det_geometries,
        echans=echans,
        include_all_ps=include_all_ps,
        point_source_list=point_source_list,
        free_spectrum=free_spectrum,
    )

    PS_Continuum_dic = {}

    for i, ps in enumerate(ep.point_sources.values()):

        if len(free_spectrum) > 0 and free_spectrum[i]:
            PS_Continuum_dic[
                "{}".format(ps.name)
            ] = Point_Source_Continuum_Fit_Spectrum(
                "ps_{}_spectrum_fitted".format(ps.name), E_norm=25.0
            )

            PS_Continuum_dic["{}".format(ps.name)].build_spec_integral(
                use_numba=use_numba
            )

            PS_Continuum_dic["{}".format(ps.name)].set_effective_responses(
                effective_responses=ps.ps_effective_response
            )

            PS_Continuum_dic["{}".format(ps.name)].set_dets_echans(
                detectors=data.detectors, echans=data.echans
            )

            PS_Continuum_dic["{}".format(ps.name)].set_time_bins(
                time_bins=data.time_bins
            )

            PS_Continuum_dic["{}".format(ps.name)].set_saa_mask(
                saa_mask=saa_object.saa_mask
            )

            PS_Continuum_dic["{}".format(ps.name)].set_interpolation_times(
                interpolation_times=ps.geometry_times
            )

            PS_Continuum_dic["{}".format(ps.name)].set_responses(responses=ps.responses)

            PS_Sources_list.append(
                FitSpectrumSource(
                    name="{}".format(ps.name),
                    continuum_shape=PS_Continuum_dic["{}".format(ps.name)],
                )
            )

        else:
            PS_Continuum_dic["{}".format(ps.name)] = Point_Source_Continuum(
                name="norm_point_source-{}".format(ps.name)
            )

            PS_Continuum_dic["{}".format(ps.name)].set_function_array(
                ps.get_ps_rates(data.time_bins)
            )

            PS_Continuum_dic["{}".format(ps.name)].set_saa_zero(saa_object.saa_mask)

            PS_Continuum_dic["{}".format(ps.name)].integrate_array(data.time_bins)

            PS_Sources_list.append(
                GlobalSource(
                    name="{}".format(ps.name),
                    continuum_shape=PS_Continuum_dic["{}".format(ps.name)],
                )
            )

    return PS_Sources_list


def setup_earth_free(data, albedo_cgb_object, saa_object, use_numba):
    """
    Setup Earth Albedo source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """

    earth_albedo = Earth_Albedo_Continuum_Fit_Spectrum()

    earth_albedo.build_spec_integral(use_numba=use_numba)

    earth_albedo.set_dets_echans(detectors=data.detectors, echans=data.echans)

    earth_albedo.set_effective_responses(
        effective_responses=albedo_cgb_object.earth_effective_response
    )

    earth_albedo.set_time_bins(time_bins=data.time_bins)

    earth_albedo.set_saa_mask(saa_mask=saa_object.saa_mask)

    earth_albedo.set_interpolation_times(
        interpolation_times=albedo_cgb_object.geometry_times
    )

    earth_albedo.set_responses(responses=albedo_cgb_object.responses)

    Source_Earth_Albedo_Continuum = FitSpectrumSource(
        name="Earth occultation", continuum_shape=earth_albedo
    )

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

    earth_albedo.set_function_array(albedo_cgb_object.get_earth_rates(data.time_bins))
    earth_albedo.set_saa_zero(saa_object.saa_mask)

    earth_albedo.integrate_array(data.time_bins)

    Source_Earth_Albedo_Continuum = GlobalSource("Earth Albedo", earth_albedo)

    return Source_Earth_Albedo_Continuum


def setup_cgb_free(data, albedo_cgb_object, saa_object, use_numba):
    """
    Setup CGB source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    cgb = Cosmic_Gamma_Ray_Background_Fit_Spectrum()

    cgb.build_spec_integral(use_numba=use_numba)

    cgb.set_dets_echans(detectors=data.detectors, echans=data.echans)

    cgb.set_effective_responses(
        effective_responses=albedo_cgb_object.cgb_effective_response
    )

    cgb.set_time_bins(time_bins=data.time_bins)

    cgb.set_saa_mask(saa_mask=saa_object.saa_mask)

    cgb.set_interpolation_times(interpolation_times=albedo_cgb_object.geometry_times)

    cgb.set_responses(responses=albedo_cgb_object.responses)

    Source_CGB_Continuum = FitSpectrumSource(name="CGB", continuum_shape=cgb)

    return Source_CGB_Continuum


def setup_cgb_fix(data, albedo_cgb_object, saa_object):
    """
    Setup CGB source with fixed spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    cgb = Cosmic_Gamma_Ray_Background()

    cgb.set_function_array(albedo_cgb_object.get_cgb_rates(data.time_bins))

    cgb.set_saa_zero(saa_object.saa_mask)

    cgb.integrate_array(data.time_bins)

    Source_CGB_Albedo_Continuum = GlobalSource("CGB", cgb)

    return Source_CGB_Albedo_Continuum

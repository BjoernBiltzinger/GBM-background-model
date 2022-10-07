from gbmbkgpy.modeling.source import (
    ContinuumSource,
    SAASource,
    GlobalSource,
    FitSpectrumSource,
)

from gbmbkgpy.modeling.functions import (
    SAA_Decay,
    ContinuumFunction,
    GlobalFunction,
    GlobalFunctionSpectrumFit,
)

from gbmbkgpy.modeling.point_source import PointSrc_fixed, PointSrc_free
from gbmbkgpy.modeling.sun import Sun

import numpy as np
import pandas as pd
import tempfile
import os

from gbmbkgpy.io.package_data import (
    get_path_of_data_dir,
    get_path_of_external_data_dir,
    get_path_of_data_file,
)
from gbmbkgpy.utils.select_pointsources import SelectPointsources

# see if we have mpi and/or are upalsing parallel
try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLDGlobalFunction
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
    geometry,
    echans=[],
    det_responses=None,
    albedo_cgb_object_free=None,
    albedo_cgb_object_fixed=None,
    gc_object=None,
    use_saa=False,
    use_constant=True,
        norm_constant=1,
    use_cr=True,
        norm_cr=1,
    use_earth=True,
    use_cgb=True,
    use_sun=True,
    use_gc=False,
    point_source_list=[],
    fix_earth=False,
    fix_cgb=False,
    saa_decays_per_exit=1,
    saa_decay_per_detector=False,
    saa_decay_at_day_start=True,
    saa_decay_model="exponential",
    cr_approximation="MCL",
    use_numba=False,
):
    """
    Setup all sources
    :param data: Data object
    :param saa_object: saa precaculation object
    :param ep: external prob object
    :param geometry: geometry precalculation object
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
    :param use_gc: use galactic center?
    :param point_source_list: PS to use
    :param fix_earth: fix earth spectrum?
    :param fix_cgb: use cgb?
    :param nr_saa_decays:
    :param decay_at_day_start:
    :param bgo_cr_approximation: Use bgo cr approximation
    :return:
    """

    assert len(echans) > 0, "Please give at least one echan"
    #echans = sorted(echans)

    assert (
        type(use_saa) == bool
        and type(use_cr) == bool
        and type(use_earth) == bool
        and type(use_cgb) == bool
        and type(fix_earth) == bool
        and type(fix_cgb) == bool
    ), "Please only use True or False here."

    total_sources = []

    # Go through all possible types of sources and add them in a list

    for index, echan in enumerate(echans):

        if use_saa:
            total_sources.extend(
                setup_SAA(
                    data,
                    saa_object,
                    echan,
                    index,
                    saa_decays_per_exit,
                    saa_decay_at_day_start,
                    saa_decay_per_detector,
                    saa_decay_model,
                )
            )

        if use_constant:
            total_sources.append(setup_Constant(data, saa_object, echan, index, norm_constant))

        if use_cr:
            total_sources.append(
                setup_CosmicRays(data, ep, saa_object, echan, index, cr_approximation, norm_cr)
            )

    if use_sun:
        total_sources.append(
            setup_sun(data,
                      saa_object,
                      det_responses,
                      geometry,
                      echans,
            )
        )

    if use_gc:
        total_sources.append(
            setup_gc(data, gc_object, saa_object)
        )

    if point_source_list:
        if len(point_source_list) > 0:
            total_sources.extend(
                setup_ps(
                    data=data,
                    ep=ep,
                    saa_object=saa_object,
                    det_responses=det_responses,
                    geometry=geometry,
                    echans=echans,
                    point_source_list=point_source_list,
                    use_numba=use_numba,
                )
            )

    if use_earth:

        if fix_earth:
            total_sources.append(setup_earth_fix(data, albedo_cgb_object_fixed, saa_object))

        else:
            total_sources.append(
                setup_earth_free(
                    data, albedo_cgb_object_free, saa_object
                )
            )

    if use_cgb:

        if fix_cgb:
            total_sources.append(setup_cgb_fix(data, albedo_cgb_object_fixed, saa_object))

        else:
            total_sources.append(
                setup_cgb_free(data, albedo_cgb_object_free, saa_object)
            )

    return total_sources


def setup_SAA(
    data,
    saa_object,
    echan,
    index,
    decays_per_exit=1,
    decay_at_day_start=True,
    decay_per_detector=False,
    decay_model="exponential",
):
    """
    Setup for SAA sources
    :param decay_at_day_start:
    :param decays_per_exit: Number of decays that should be fittet to each SAA Exit
    :param decay_model: used model for decay: "exponential" or "linear"
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
        for i in range(decays_per_exit):
            day_start.append(data.day_met)

    start_times = np.append(np.array(day_start), saa_object.saa_exit_times)

    for time in start_times:

        if decay_per_detector:

            for det_idx, det in enumerate(data.detectors):

                saa_dec = SAA_Decay(
                    saa_number=str(saa_n),
                    echan=str(echan),
                    model=decay_model,
                    detector=det,
                    det_idx=det_idx,
                )

                saa_dec.set_saa_exit_time(np.array([time]))

                saa_dec.set_time_bins(data.time_bins)

                saa_dec.set_nr_detectors(len(data._detectors))

                saa_dec.set_det_idx(det_idx)

                # precalculation for later evaluation
                saa_dec.precalulate_time_bins_integral()

                SAA_Decay_list.append(
                    SAASource(
                        f"saa_{saa_n} det_{det} echan_{echan}",
                        saa_dec,
                        index,
                    )
                )

            saa_n += 1

        else:

            saa_dec = SAA_Decay(
                saa_number=str(saa_n),
                echan=str(echan),
                model=decay_model,
                detector="all",
                det_idx=None,
            )

            saa_dec.set_saa_exit_time(np.array([time]))

            saa_dec.set_time_bins(data.time_bins)

            saa_dec.set_nr_detectors(len(data._detectors))

            # precalculation for later evaluation
            saa_dec.precalulate_time_bins_integral()

            SAA_Decay_list.append(
                SAASource(f"saa_{saa_n} echan_{echan}", saa_dec, index)
            )
            saa_n += 1

    return SAA_Decay_list


def setup_sun(data,
              saa_object,
              det_responses,
              geometry,
              echans):
    """
    Setup for sun as bkg source
    """

    sun_spec = {"spectrum_type": "pl", "powerlaw_index": 3}
    
    sun_object = Sun(det_responses=det_responses,
                     geometry_object=geometry,
                     echans=echans,
                     spec=sun_spec,
    )
    
    sun = GlobalFunction("sun_norm")

    sun.set_function_array(sun_object.get_sun_rates(data.time_bins))

    sun.set_saa_zero(saa_object.saa_mask)

    sun.integrate_array(data.time_bins)

    sun_continuum = GlobalSource(name="sun", continuum_shape=sun)

    return sun_continuum


def setup_Constant(data, saa_object, echan, index, const_norm=1):
    """
    Constant source
    """
    Constant = ContinuumFunction(f"norm_constant_echan-{echan}")

    Constant.set_function_array(const_norm*np.ones((len(data.time_bins), len(data._detectors), 2)))

    Constant.set_saa_zero(saa_object.saa_mask)

    # precalculate the integration over the time bins
    Constant.integrate_array(data.time_bins)

    Constant_Continuum = ContinuumSource(f"constant_echan-{echan}", Constant, index)
    return Constant_Continuum


def setup_CosmicRays(data, ep, saa_object, echan, index,
                     cr_approximation, cr_norm=1):
    """
    Setup for CosmicRay source
    :param index:
    :param saa_object:
    :param ep: external prob object
    :param echan: energy channel
    :param data: Data object
    :return: Constant and magnetic continuum source
    """
    mag_con = ContinuumFunction(f"norm_magnetic_echan-{echan}")
    if cr_approximation == "BGO":

        mag_con.set_function_array(cr_norm*ep.bgo_cr_approximation((data.time_bins)))

        mag_con.remove_vertical_movement()

        mag_con.set_saa_zero(saa_object.saa_mask)

        mag_con.integrate_array(data.time_bins)

        Source_Magnetic_Continuum = ContinuumSource(
            f"BGO_CR_Approx_echan-{echan}", mag_con, index
        )

    elif cr_approximation == "MCL":

        mag_con.set_function_array(cr_norm*ep.mc_l_rates((data.time_bins)))

        mag_con.set_saa_zero(saa_object.saa_mask)

        mag_con.remove_vertical_movement()

        # precalculate the integration over the time bins
        mag_con.integrate_array(data.time_bins)

        Source_Magnetic_Continuum = ContinuumSource(
            f"McIlwain_L-parameter_echan-{echan}", mag_con, index
        )

    else:

        mag_con.set_function_array(cr_norm*ep.acd_cr_approximation(data.time_bins))

        mag_con.set_saa_zero(saa_object.saa_mask)

        mag_con.remove_vertical_movement()

        # precalculate the integration over the time bins
        mag_con.integrate_array(data.time_bins)

        Source_Magnetic_Continuum = ContinuumSource(
            f"LAT_ACD-parameter_echan-{echan}", mag_con, index
        )

    return Source_Magnetic_Continuum


def setup_ps(
    data,
    ep,
    saa_object,
    det_responses,
    geometry,
    echans,
    point_source_list,
    use_numba=False,
):
    """
    Set up the global sources which are the same for all echans.
    At the moment the Earth Albedo and the CGB.
    :param point_source_list:
    :param echan_list:
    :param geometry:
    :param saa_object:
    :param det_responses:
    :param data:
    :return:
    """
    #piv = np.mean(list(det_responses.responses.values())[0].Ebin_out_edge)

    detectors = list(det_responses.responses.keys())
    rsp = det_responses.responses
    data_type = rsp[detectors[0]].data_type
    if data_type == "ctime" or data_type == "trigdat":
        echans_mask = []

        for e in echans:
            bounds = e.split("-")
            mask = np.zeros(8, dtype=bool)
            if len(bounds) == 1:
                # Only one echan given
                index = int(bounds[0])
                mask[index] = True
            else:
                # Echan start and stop given
                index_start = int(bounds[0])
                index_stop = int(bounds[1])
                mask[index_start : index_stop + 1] = np.ones(
                    1 + index_stop - index_start, dtype=bool
                )
            echans_mask.append(mask)

    elif data_type == "cspec":
        echans_mask = []
        
        for e in echans:
            bounds = e.split("-")
            mask = np.zeros(128, dtype=bool)
            if len(bounds) == 1:
                # Only one echan given
                index = int(bounds[0])
                mask[index] = True
            else:
                # Echan start and stop given
                index_start = int(bounds[0])
                index_stop = int(bounds[1])
                mask[index_start : index_stop + 1] = np.ones(
                    1 + index_stop - index_start, dtype=bool
                )
            echans_mask.append(mask)

    Eout_edges = rsp[detectors[0]].Ebin_out_edge
    Ebins = np.zeros((len(Eout_edges)-1,2))
    Ebins[:,0] = Eout_edges[:-1]
    Ebins[:,1] = Eout_edges[1:]
    mi = np.zeros(len(echans_mask))
    ma = np.zeros(len(echans_mask))
    for i, mask in enumerate(echans_mask):
        mi[i] = np.min(np.argwhere(mask))
        ma[i] = np.max(np.argwhere(mask))
    minindex = int(np.min(mi))
    maxindex = int(np.max(ma)) 
    piv = np.sqrt(Ebins[minindex,0]*Ebins[maxindex,1])     
    
    PS_Sources_list = []

    # Point-Source Sources
    point_sources = build_point_sources(
        det_responses=det_responses,
        geometry=geometry,
        echans=echans,
        point_source_list=point_source_list,
        data=data,
    )

    PS_Continuum_dic = {}
    if "auto_swift" in point_source_list.keys():
        limit = point_source_list["auto_swift"]["flux_limit"]
        day = data.dates[0]
        filepath = os.path.join(
            get_path_of_external_data_dir(),
            "point_sources",
            f"ps_swift_{day}_limit_{limit}.dat"
        )
        ps_df_add = pd.read_table(filepath, names=["name", "ra", "dec"])

        auto_swift_ps = [entry[1].upper() for entry in ps_df_add.itertuples()]
        exclude = [
            entry.upper() for
            entry in point_source_list["auto_swift"]["exclude"]
        ]

    else:
        auto_swift_ps = []
        exclude = []
    for i, (key, ps) in enumerate(point_sources.items()):

        name = ps.name.upper()
        if not isinstance(ps, PointSrc_fixed):

            identifier = "_".join(key.split("_")[:-1])
            if identifier == "":
                identifier = key

            identifier = identifier.upper()
            
            if (identifier in auto_swift_ps) and (
                identifier not in exclude
            ):
                spec = "pl"

            else:
                if "bb" in point_source_list[identifier]["spectrum"]:
                    if "pl" in point_source_list[identifier]["spectrum"]:
                        spec = "bb+pl"
                    else:
                        spec = "bb"
                elif "pl" in point_source_list[identifier]["spectrum"]:
                    spec = "pl"
                else:
                    raise NotImplementedError(
                        "Only pl or bb or both spectra for point sources!"
                    )
            PS_Continuum_dic[name] = GlobalFunctionSpectrumFit(
                f"ps_{name}_spectrum_fitted",
                spectrum=spec,
                E_norm=piv,
                use_numba=use_numba,
            )

            PS_Continuum_dic[name].build_spec_integral()

            PS_Continuum_dic[name].set_effective_responses(
                effective_responses=ps.ps_effective_response
            )

            PS_Continuum_dic[name].set_dets_echans(
                detectors=data.detectors, echans=data.echans
            )

            PS_Continuum_dic[name].set_time_bins(
                time_bins=data.time_bins
            )

            PS_Continuum_dic[name].set_saa_mask(
                saa_mask=saa_object.saa_mask
            )

            PS_Continuum_dic[name].set_interpolation_times(
                interpolation_times=ps._geom.geometry_times
            )

            PS_Continuum_dic[name].set_responses(responses=ps.responses)
            
            if ps._time_variation_interp is None:
                PS_Continuum_dic[name].set_norm_time_variability(
                    np.ones_like(np.mean(data.time_bins, axis=1))
                )
            else:
                PS_Continuum_dic[name].set_norm_time_variability(
                    ps._time_variation_interp(np.mean(data.time_bins, axis=1))
                )

            PS_Sources_list.append(
                FitSpectrumSource(
                    name=name,
                    continuum_shape=PS_Continuum_dic[name],
                )
            )

        else:

            spec_name = ps.spec_type
            PS_Continuum_dic[f"{name}_{spec_name}"] = GlobalFunction(
                f"norm_{name}_{spec_name}"
            )

            PS_Continuum_dic[f"{name}_{spec_name}"].set_function_array(
                ps.get_ps_rates(data.time_bins)
            )

            PS_Continuum_dic[f"{name}_{spec_name}"].set_saa_zero(saa_object.saa_mask)

            PS_Continuum_dic[f"{name}_{spec_name}"].integrate_array(data.time_bins)

            PS_Sources_list.append(
                GlobalSource(
                    name=f"{name}_{spec_name}",
                    continuum_shape=PS_Continuum_dic[f"{name}_{spec_name}"],
                )
            )

    return PS_Sources_list


def setup_earth_free(data, albedo_cgb_object, saa_object, use_numba=True):
    """
    Setup Earth Albedo source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """

    earth_albedo = GlobalFunctionSpectrumFit(
        "earth_albedo_spectrum_fitted", spectrum="bpl", use_numba=use_numba
    )

    earth_albedo.build_spec_integral()

    earth_albedo.set_dets_echans(detectors=data.detectors, echans=data.echans)

    earth_albedo.set_effective_responses(
        effective_responses=albedo_cgb_object.earth_effective_response
    )

    earth_albedo.set_time_bins(time_bins=data.time_bins)

    earth_albedo.set_saa_mask(saa_mask=saa_object.saa_mask)

    earth_albedo.set_interpolation_times(
        interpolation_times=albedo_cgb_object.interp_times
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

    earth_albedo = GlobalFunction("norm_earth_albedo")

    earth_albedo.set_function_array(albedo_cgb_object.get_earth_rates(data.time_bins))
    earth_albedo.set_saa_zero(saa_object.saa_mask)

    earth_albedo.integrate_array(data.time_bins)

    Source_Earth_Albedo_Continuum = GlobalSource("earth_albedo", earth_albedo)

    return Source_Earth_Albedo_Continuum


def setup_gc(data, gc_object, saa_object):
    """
    Setup galactic center source with fixed spectrum (for now only fixed spectrum available).
    :param data:
    :param gc_object:
    :param saa_object:
    :return:
    """

    gc = GlobalFunction("norm_gc")
    gc.set_function_array(gc_object.get_gc_rates(data.time_bins))
    gc.set_saa_zero(saa_object.saa_mask)

    gc.integrate_array(data.time_bins)

    Source_gc_Continuum = GlobalSource("galactic_center", gc)

    return Source_gc_Continuum


def setup_cgb_free(data, albedo_cgb_object, saa_object, use_numba=True):
    """
    Setup CGB source with free spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    cgb = GlobalFunctionSpectrumFit(
        "CGB_spectrum_fitted", spectrum="bpl", use_numba=use_numba
    )

    cgb.build_spec_integral()

    cgb.set_dets_echans(detectors=data.detectors, echans=data.echans)

    cgb.set_effective_responses(
        effective_responses=albedo_cgb_object.cgb_effective_response
    )

    cgb.set_time_bins(time_bins=data.time_bins)

    cgb.set_saa_mask(saa_mask=saa_object.saa_mask)

    cgb.set_interpolation_times(interpolation_times=albedo_cgb_object.interp_times)

    cgb.set_responses(responses=albedo_cgb_object.responses)

    Source_CGB_Continuum = FitSpectrumSource(name="cgb", continuum_shape=cgb)

    return Source_CGB_Continuum


def setup_cgb_fix(data, albedo_cgb_object, saa_object):
    """
    Setup CGB source with fixed spectrum
    :param data:
    :param albedo_cgb_object:
    :param saa_object:
    :return:
    """
    cgb = GlobalFunction("norm_cgb")

    cgb.set_function_array(albedo_cgb_object.get_cgb_rates(data.time_bins))

    cgb.set_saa_zero(saa_object.saa_mask)

    cgb.integrate_array(data.time_bins)

    Source_CGB_Albedo_Continuum = GlobalSource("cgb", cgb)

    return Source_CGB_Albedo_Continuum


def build_point_sources(
    det_responses, geometry, echans, point_source_list=[], data=None
):
    """
    This function reads the point_sources.dat file and builds the point sources
    :param echans:
    :param include_all_ps:
    :param source_list:
    :return:
    """
    file_path = os.path.join(
        get_path_of_external_data_dir(),
        "point_sources",
        "ps_all_swift.dat"
    )
    if not os.path.exists(file_path):
        # default file
        file_path = get_path_of_data_file(
            "background_point_sources/", "point_sources_swift.dat"
        )
        
    ps_df = pd.read_table(file_path, names=["name", "ra", "dec"])

    # instantiate dic of point source objects
    point_sources_dic = {}

    ### Single core calc ###
    for i, ps in enumerate(point_source_list):
        for row in ps_df.itertuples():
            
            if row[1].upper() == ps:
                
                if not point_source_list[ps]["fixed"]:
                    point_sources_dic[row[1]] = PointSrc_free(
                        name=row[1],
                        ra=row[2],
                        dec=row[3],
                        det_responses=det_responses,
                        geometry=geometry,
                        echans=echans,
                    )

                else:
                    for entry in point_source_list[ps]["spectrum"]:
                        point_sources_dic[f"{row[1]}_{entry}"] = PointSrc_fixed(
                            name=row[1],
                            ra=row[2],
                            dec=row[3],
                            det_responses=det_responses,
                            geometry=geometry,
                            echans=echans,
                            spec=point_source_list[ps]["spectrum"][entry],
                        )
                
                break


    # Add the point sources that are given as file with list of point sources
    for i, ps in enumerate(point_source_list):
        if ps[:4] == "list":
            ps_df_add = pd.read_table(
                point_source_list[ps]["path"], names=["name", "ra", "dec"]
            )
            for row in ps_df_add.itertuples():
                if not point_source_list[ps]["fixed"]:
                    point_sources_dic[f"{ps}_{row[1]}"] = PointSrc_free(
                        name=row[1],
                        ra=row[2],
                        dec=row[3],
                        det_responses=det_responses,
                        geometry=geometry,
                        echans=echans,
                    )

                else:
                    for entry in point_source_list[ps]["spectrum"]:
                        point_sources_dic[f"{ps}_{row[1]}_{entry}"] = PointSrc_fixed(
                            name=row[1],
                            ra=row[2],
                            dec=row[3],
                            det_responses=det_responses,
                            geometry=geometry,
                            echans=echans,
                            spec=point_source_list[ps]["spectrum"][entry],
                        )
    # Add the auto point source selection using the Swift survey if wanted
    if "auto_swift" in point_source_list.keys():
        # Write a temp .dat file with all the point sources, after that we can do the
        # same as above for a given list

        # Threshold flux which point sources should be added in units of Crab 15-50keV Flux
        limit = point_source_list["auto_swift"]["flux_limit"]
        min_separation = point_source_list["auto_swift"].get(
            "min_separation_angle", None
        )

        # Use first day in data object to get the needed point sources
        day = data.dates[0]

        # Exclude some of them?
        exclude = point_source_list["auto_swift"]["exclude"]
        exclude = [entry.upper() for entry in exclude]
        free = point_source_list["auto_swift"].get("free", [])
        free = [entry.upper() for entry in free]
        time_variable = point_source_list["auto_swift"].get("time_variable", False)

        all_time_variable_same = True
        if type(time_variable) is not bool:
            time_variable = [entry.upper() for entry in time_variable]
            all_time_variable_same = False

        # Initalize Pointsource selection
        sp = SelectPointsources(
            limit,
            time_string=day,
            update=point_source_list["auto_swift"].get("update_catalog", None),
            min_separation_angle=min_separation,
        )

        # Create ps file
        filepath_all = os.path.join(
            get_path_of_external_data_dir(),
            "point_sources",
            "ps_all_swift.dat"
        )

        sp.write_all_psfile(filepath_all)


        filepath = os.path.join(
            get_path_of_external_data_dir(),
            "point_sources",
            f"ps_swift_{day}_limit_{limit}.dat"
        )
        sp.write_psfile(filepath)

        if all_time_variable_same:
            if time_variable:
                ps_time_var_interp = sp.ps_time_variation()
        else:
            if len(time_variable) > 0:
                ps_time_var_interp = sp.ps_time_variation()
        
        # Read it as pandas
        ps_df_add = pd.read_table(filepath, names=["name", "ra", "dec"])
        # Add all of them as fixed pl sources
        spec = {"spectrum_type": "pl", "powerlaw_index": "swift", "norm": 1.0}

        for row in ps_df_add.itertuples():
            if not row[1].upper() in exclude:
                if row[1].upper() in free:
                    point_sources_dic[f"{row[1]}_pl"] = PointSrc_free(
                        name=row[1],
                        ra=row[2],
                        dec=row[3],
                        det_responses=det_responses,
                        geometry=geometry,
                        echans=echans,
                    )

                else:

                    point_sources_dic[f"{row[1]}_pl"] = PointSrc_fixed(
                        name=row[1],
                        ra=row[2],
                        dec=row[3],
                        det_responses=det_responses,
                        geometry=geometry,
                        echans=echans,
                        spec=spec,
                    )

                if all_time_variable_same:

                    if time_variable:
                        print(f"Point source {row[1]} is set to variate with time")

                        point_sources_dic[f"{row[1]}_pl"].set_time_variation_interp(
                            ps_time_var_interp[row[1]]
                        )
                else:
                    if row[1].upper() in ",".join(time_variable):
                        print(f"Point source {row[1]} is set to variate with time")

                        point_sources_dic[f"{row[1]}_pl"].set_time_variation_interp(
                            ps_time_var_interp[row[1]]
                        )
        # temp.close()

    return point_sources_dic

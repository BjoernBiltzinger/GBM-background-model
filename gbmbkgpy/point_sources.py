import astropy.io.fits as fits
import numpy as np
from gbmbkgpy.external_prop import ExternalProps, writefile
from gbmbkgpy.work_module_refactor import calculate


class PointSources(object):

    def __init__(self, day, detector_name, echan, data_type):

        self._point_sources = ExternalProps(day).point_sources()



    def point_source_data (self, sun_ang_bin, detector_name, day, bin_time_mid, data_type, echan, total_rate):
        sources_data = ExternalProps.point_sources(ExternalProps())
        sources_names = sources_data[0]
        sources_coordinates = sources_data[1]
        sources_number = int(len(sources_names))

        # calculate the pointsources data
        sources_ang_bin = np.array([np.zeros(len(sun_ang_bin))])

        for i in range(0, sources_number):
            src_ra = sources_coordinates[0, i]
            src_dec = sources_coordinates[1, i]

            src_data = calculate.burst_ang_bin(calculate(), detector_name, day, src_ra, src_dec, bin_time_mid,
                                               data_type)
            src_ang_bin = src_data[0]
            src_ang_bin = calculate.ang_eff(calculate(), src_ang_bin, echan, data_type)[0]
            src_occ = \
                calculate.src_occultation_bin(calculate(), day, src_ra, src_dec, bin_time_mid, detector_name, data_type)[0]

            src_ang_bin[np.where(src_occ == 0)] = 0
            src_ang_bin[np.where(total_rate == 0)] = 0

            # remove vertical movement from scaling the src_ang_bin functions
            src_ang_bin[src_ang_bin > 0] = src_ang_bin[src_ang_bin > 0] - np.min(src_ang_bin[src_ang_bin > 0])

            src_ang_bin = np.array(src_ang_bin)

            sources_ang_bin = np.concatenate((sources_ang_bin, [src_ang_bin]), axis=0)

        sources_ang_bin = np.delete(sources_ang_bin, 0, 0)

        return sources_ang_bin, sources_number
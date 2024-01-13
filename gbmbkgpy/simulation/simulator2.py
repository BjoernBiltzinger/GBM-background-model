import numpy as np
import astropy.io.fits as fits
import os

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.package_data import get_path_of_external_data_dir

class Simulator:
    def __init__(self, config_yml):

        self._bkg_model_generator = BackgroundModelGenerator()
        self._bkg_model_generator.from_config_file(config_yml)
        self._config = self._bkg_model_generator.config
    def simulate(self):
        # CR
        #norm = self._config["sources"]["cr_norm"]
        #if type(norm) is np.float:
        #    norm = np.ones(len(self._bkg_model_generator.data._echans))*norm

        #for e, v in enumerate(self._bkg_model_generator.data._echans):
        #    self._bkg_model_generator.model.free_parameters[f"norm_magnetic-echan-{v}"] = norm[e]


        # Constant
        #norm = self._config["sources"]["const_norm"]
        #if type(norm) is np.float:
        #    norm = np.ones(len(self._bkg_model_generator.data._echans))*norm

        #for e, v in enumerate(self._bkg_model_generator.data._echans):
        #    self._bkg_model_generator.model.free_parameters[f"norm_constant_echan-{v}"] = norm[e]

        self._counts = self._bkg_model_generator.model.get_counts(
            self._bkg_model_generator.data.time_bins
        )
        
    def save_to_fits(self, overwrite=False):

        with progress_bar(
                12, title="Exporting simulation to fits for all 12 NaI detectors:"
        ) as p:
            for det_idx, det in enumerate(self._bkg_model_generator.data._detectors):
                print(np.max(self._counts[:,det_idx]))
                # Add Poisson noise
                np.random.seed(self._bkg_model_generator.config["general"]["random_seed"])

                counts_poisson = np.random.poisson(self._counts[:,det_idx])

                # Write to fits file
                primary_hdu = fits.PrimaryHDU()

                # Ebounds
                c1 = fits.Column(name="E_MIN",
                                 array=self._bkg_model_generator.response.responses[det]._Ebin_out_edge[:-1],
                                 format="1E")

                c2 = fits.Column(name="E_MAX",
                                 array=self._bkg_model_generator.response.responses[det]._Ebin_out_edge[1:],
                                 format="1E")

                ebounds = fits.BinTableHDU.from_columns([c1, c2], name="EBOUNDS")

                # SPECTRUM
                if self._bkg_model_generator.data.data_type == "ctime":

                    c3 = fits.Column(name="COUNTS", array=counts_poisson, format="8I")

                else:

                    c3 = fits.Column(name="COUNTS", array=counts_poisson, format="128I")

                c4 = fits.Column(name="TIME",
                                 array=self._bkg_model_generator.data.time_bins[:, 0],
                                 format="1D")

                c5 = fits.Column(name="ENDTIME",
                                 array=self._bkg_model_generator.data.time_bins[:, 1],
                                 format="1D")

                data = fits.BinTableHDU.from_columns([c3, c4, c5], name="SPECTRUM")

                hdul = fits.HDUList([primary_hdu, ebounds, data])

                result_dir = os.path.join(
                    get_path_of_external_data_dir(),
                    "simulation",
                    self._bkg_model_generator.data.data_type,
                    self._bkg_model_generator.data.dates[0],
                )

                if not os.path.exists(result_dir):

                    os.makedirs(result_dir)

                hdul.writeto(
                    os.path.join(
                        result_dir,
                        f"glg_{self._bkg_model_generator.data.data_type}_{det}_{self._bkg_model_generator.data.dates[0]}_v00.pha"
                    ),
                    overwrite=overwrite,
                )

                p.increase()

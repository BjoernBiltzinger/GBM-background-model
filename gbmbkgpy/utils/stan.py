import numpy as np
from scipy.interpolate import interp1d
import arviz as av
import matplotlib.pyplot as plt


class StanModelConstructor(object):
    """
    Object to construct the .stan model
    """

    def __init__(self, model_generator):

        data = model_generator.data
        model = model_generator.model

        num_dets = len(data.detectors)
        num_echans = len(data.echans)

        # How many of which sources?

        # Global
        sources = model.global_sources
        self._num_fixed_global_sources = len(sources)
        if self._num_fixed_global_sources > 0:
            self._use_fixed_global_sources = True
        else:
            self._use_fixed_global_sources = False

        # Cont
        sources = model.continuum_sources
        self._num_cont_sources = int(len(sources) / num_echans)
        if self._num_cont_sources > 0:
            assert self._num_cont_sources == 2, "Must be two cont sources!"
            self._use_cont_sources = True
        else:
            self._use_cont_sources = False

        # SAA
        sources = model.saa_sources
        self._num_saa_exits = int(len(sources) / num_echans)
        if self._num_saa_exits > 0:
            self._use_saa = True
        else:
            self._use_saa = False

        # Free spectrum
        sources = model.fit_spectrum_sources
        self._num_free_ps = 0
        self._use_free_earth = False
        self._use_free_cgb = False
        for k in sources.keys():
            if k == "Earth occultation":
                self._use_free_earth = True
            elif k == "CGB":
                self._use_free_cgb = True
            else:
                self._num_free_ps += 1

        if self._num_free_ps > 0:
            self._use_free_ps = True
        else:
            self._use_free_ps = False

    def source_count(self):

        return dict(
            use_free_earth=self._use_free_earth,
            use_free_cgb=self._use_free_cgb,
            num_free_ps=self._num_free_ps,
            num_saa_exits=self._num_saa_exits,
            num_cont_sources=self._num_cont_sources,
            num_fixed_global_sources=self._num_fixed_global_sources,
        )

    def create_stan_file(self, save_path, total_only=False):

        if not total_only:
            text = (
                self.function_block()
                + self.data_block()
                + self.trans_data_block()
                + self.parameter_block()
                + self.trans_parameter_block()
                + self.model_block()
                + self.generated_block()
            )
        else:
            text = (
                self.function_block()
                + self.data_block()
                + self.trans_data_block()
                + self.parameter_block()
                + self.trans_parameter_block()
                + self.model_block()
                + self.generated_block_total_only()
            )

        with open(save_path, "w") as f:
            f.write(text)

    def function_block(self):
        text = "functions { \n"

        if self._use_saa:
            text += (
                "\tvector saa_total(vector[] saa_norm_vec, vector[] saa_decay_vec, matrix[] t_t0, int num_data_points,int num_saa_exits){\n"
                "\t\tvector[num_data_points] total_saa_counts = rep_vector(0.0, num_data_points);\n"
                "\t\tfor (i in 1:num_saa_exits){\n"
                "\t\t\ttotal_saa_counts += saa_norm_vec[i]./saa_decay_vec[i].*(exp(-t_t0[i,:,1].*saa_decay_vec[i]) - exp(-t_t0[i,:, 2].*saa_decay_vec[i, :]));\n"
                "\t\t}\n"
                "\t\treturn total_saa_counts;\n\t}\n\n"
            )

        # Main partial sum function
        main = "\treal partial_sum(int[] counts, int start, int stop\n"

        if self._use_fixed_global_sources:
            main += "\t, vector[] base_counts_array, real[] norm_fixed\n"

        if self._use_free_earth:
            main += "\t, matrix base_response_array_earth, vector earth_spec\n"

        if self._use_free_cgb:
            main += "\t, matrix base_response_array_cgb, vector cgb_spec\n"

        if self._use_free_ps:
            main += "\t, matrix[] base_response_array_free_ps, vector[] ps_spec\n"

        if self._use_cont_sources:
            main += "\t, vector[] norm_cont_vec, vector[] base_counts_array_cont\n"

        if self._use_saa:
            main += "\t, matrix[] t_t0, vector[] saa_decay_vec, vector[] saa_norm_vec\n"

        main += "\t){\n\t\treturn poisson_propto_lpmf(counts | 0\n"

        if self._use_saa:
            for i in range(self._num_saa_exits):
                main += (
                    f"\t\t\t+saa_norm_vec[{i+1}, start:stop]./saa_decay_vec[{i+1}, start:stop].*"
                    f"(exp(-t_t0[{i+1},start:stop,1].*saa_decay_vec[{i+1}, start:stop])-"
                    f"exp(-t_t0[{i+1},start:stop,2].*saa_decay_vec[{i+1}, start:stop]))\n"
                )

        if self._use_fixed_global_sources:
            for i in range(self._num_fixed_global_sources):
                main += (
                    f"\t\t\t+norm_fixed[{i+1}]*base_counts_array[{i+1},start:stop]\n"
                )

        if self._use_cont_sources:
            for i in range(self._num_cont_sources):
                main += f"\t\t\t+norm_cont_vec[{i+1}, start:stop].*base_counts_array_cont[{i+1}, start:stop]\n"

        if self._use_free_earth:
            main += "\t\t\t+base_response_array_earth[start:stop]*earth_spec\n"

        if self._use_free_cgb:
            main += "\t\t\t+base_response_array_cgb[start:stop]*cgb_spec\n"

        if self._use_free_ps:
            for i in range(self._num_free_ps):
                main += f"\t\t\t+base_response_array_free_ps[{i+1}, start:stop]*ps_spec[{i+1}]\n"

        main += "\t\t\t);\n\t}\n"

        text = text + main + "}\n\n"
        return text

    def data_block(self):
        # Start
        text = "data { \n"
        # This we need always:
        text += "\tint<lower=1> num_time_bins;\n"
        text += "\tint<lower=1> num_dets;\n"
        text += "\tint<lower=1> num_echans;\n"

        text += "\tint<lower=1> rsp_num_Ein;\n"
        text += "\tvector[rsp_num_Ein] Ebins_in[2];\n"

        text += "\tint<lower=1> grainsize;\n"
        text += "\tmatrix[num_time_bins, 2] time_bins;\n"

        text += "\tint counts[num_time_bins*num_dets*num_echans];\n"

        # Optional input
        if self._use_fixed_global_sources:
            text += "\tint<lower=0> num_fixed_comp;\n"
            text += "\tvector[num_time_bins*num_dets*num_echans] base_counts_array[num_fixed_comp];\n"
            text += "\tvector[num_fixed_comp] mu_norm_fixed;\n"
            text += "\tvector[num_fixed_comp] sigma_norm_fixed;\n"

        if self._use_cont_sources:
            text += "\tint num_cont_comp;\n"
            text += "\tvector[num_time_bins*num_dets*num_echans] base_counts_array_cont[num_cont_comp];\n"
            text += "\treal mu_norm_cont[num_cont_comp, num_dets, num_echans];\n"
            text += "\treal sigma_norm_cont[num_cont_comp, num_dets, num_echans];\n"

        if self._use_saa:
            text += "\tint num_saa_exits;\n"
            text += "\tvector[num_saa_exits] saa_start_times;\n"
            text += "\treal mu_norm_saa[num_saa_exits, num_dets, num_echans];\n"
            text += "\treal sigma_norm_saa[num_saa_exits, num_dets, num_echans];\n"
            text += "\treal mu_decay_saa[num_saa_exits, num_dets, num_echans];\n"
            text += "\treal sigma_decay_saa[num_saa_exits, num_dets, num_echans];\n"

        if self._use_free_ps:
            text += "\tint num_free_ps_comp;\n"
            text += "\tmatrix[num_echans*num_dets*num_time_bins, rsp_num_Ein] base_response_array_free_ps[num_free_ps_comp];\n"

        if self._use_free_cgb:
            text += "\tmatrix[num_echans*num_dets*num_time_bins, rsp_num_Ein] base_response_array_cgb;\n"

        if self._use_free_earth:
            text += "\tmatrix[num_echans*num_dets*num_time_bins, rsp_num_Ein] base_response_array_earth;\n"

        # Close
        text = text + "}\n\n"
        return text

    def trans_data_block(self):
        text = "transformed data { \n"

        text += "\tint num_data_points = num_time_bins*num_dets*num_echans;\n"

        if self._use_saa:
            text += "\tmatrix[num_data_points,2] t_t0[num_saa_exits];\n"

            text += (
                "\tfor (j in 1:num_saa_exits){\n"
                "\t\tfor (i in 1:num_time_bins){\n"
                "\t\t\tif (time_bins[i,1]>saa_start_times[j]){\n"
                "\t\t\t\tt_t0[j,(i-1)*num_dets*num_echans+1:i*num_dets*num_echans] = rep_matrix(time_bins[i]-saa_start_times[j], num_dets*num_echans);\n"
                "\t\t\t}\n"
                "\t\t\telse {\n"
                "\t\t\t\tt_t0[j,(i-1)*num_dets*num_echans+1:i*num_dets*num_echans] = rep_matrix(0.0, num_dets*num_echans, 2);\n"
                "\t\t\t}\n\t\t}\n\t}\n"
            )

        text = text + "}\n\n"
        return text

    def parameter_block(self):
        text = "parameters { \n"

        if self._use_fixed_global_sources:
            text += "\treal log_norm_fixed[num_fixed_comp];\n"

        if self._use_saa:
            text += "\treal log_norm_saa[num_saa_exits, num_dets, num_echans];\n"
            text += "\treal<lower=0.01,upper=10> decay_saa[num_saa_exits, num_dets, num_echans];\n"

        if self._use_cont_sources:
            text += "\treal log_norm_cont[num_cont_comp, num_dets, num_echans];\n"

        if self._use_free_earth:
            text += "\treal log_norm_earth;\n"
            text += "\treal alpha_earth;\n"
            text += "\treal Ec_earth;\n"

        if self._use_free_cgb:
            text += "\treal log_norm_cgb;\n"
            text += "\tordered[2] indices_cgb;\n"
            text += "\treal Eb_cgb;\n"

        if self._use_free_ps:
            text += "\treal log_norm_free_ps[num_free_ps_comp];\n"
            text += "\treal index_free_ps[num_free_ps_comp];\n"

        text = text + "}\n\n"
        return text

    def trans_parameter_block(self):
        text = "transformed parameters { \n"

        if self._use_fixed_global_sources:
            text += "\treal norm_fixed[num_fixed_comp] = exp(log_norm_fixed);\n"

        if self._use_saa:
            text += "\treal norm_saa[num_saa_exits,num_dets, num_echans]=exp(log_norm_saa);\n"

            text += "\tvector[num_data_points] saa_decay_vec[num_saa_exits];\n"
            text += "\tvector[num_data_points] saa_norm_vec[num_saa_exits];\n"

        if self._use_cont_sources:
            text += "\treal norm_cont[num_cont_comp, num_dets, num_echans] = exp(log_norm_cont);\n"
            text += "\tvector[num_data_points] norm_cont_vec[num_cont_comp];\n"

        if self._use_free_earth:
            text += "\treal norm_earth = exp(log_norm_earth);\n"
            text += "\tvector[rsp_num_Ein] earth_spec;\n"

        if self._use_free_cgb:
            text += "\treal norm_cgb = exp(log_norm_cgb);\n"
            text += "\tvector[rsp_num_Ein] cgb_spec;\n"

        if self._use_free_ps:
            text += "\treal norm_free_ps[num_free_ps_comp] = exp(log_norm_free_ps);\n"
            text += "\tvector[rsp_num_Ein] ps_spec[num_free_ps_comp];\n"

        if self._use_cont_sources:
            text += (
                "\tfor (l in 1:num_cont_comp){\n"
                "\t\tfor (i in 1:num_dets){\n"
                "\t\t\tfor (j in 1:num_echans){\n"
                "\t\t\t\tfor (k in 1:num_time_bins){\n"
                "\t\t\t\t\tnorm_cont_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = norm_cont[l,i,j];\n"
                "\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n"
            )

        if self._use_saa:
            text += (
                "\tfor (l in 1:num_saa_exits){\n"
                "\t\tfor (i in 1:num_dets){\n"
                "\t\t\tfor (j in 1:num_echans){\n"
                "\t\t\t\tfor (k in 1:num_time_bins){\n"
                "\t\t\t\t\tsaa_decay_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = 0.0001*decay_saa[l,i,j];\n"
                "\t\t\t\t\tsaa_norm_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = norm_saa[l,i,j];\n"
                "\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n"
            )

        if self._use_free_ps:
            text += (
                "\tfor (j in 1:num_free_ps_comp){\n"
                "\t\tfor (i in 1:rsp_num_Ein){\n"
                "\t\t\tps_spec[j][i] = (Ebins_in[2,i]-Ebins_in[1,i])*norm_free_ps[j]*(pow(0.1*Ebins_in[1,i], -index_free_ps[j])+pow(0.1*Ebins_in[2,i], -index_free_ps[j]))/2;\n"
                "\t\t}\n\t}\n"
            )

        if self._use_free_earth:
            text += (
                "\tfor (i in 1:rsp_num_Ein){\n"
                "\t\tearth_spec[i] = (Ebins_in[2,i]-Ebins_in[1,i])*0.5*norm_earth*(pow(Ebins_in[1,i], -alpha_earth)*exp(-Ec_earth/Ebins_in[1,i])+pow(Ebins_in[2,i], -alpha_earth)*exp(-Ec_earth/Ebins_in[2,i]));\n"
                "\t}\n"
            )

        if self._use_free_cgb:
            text += (
                "\tfor (i in 1:rsp_num_Ein){\n"
                "\t\tcgb_spec[i] = (Ebins_in[2,i]-Ebins_in[1,i])*0.5*(norm_cgb/(pow(Ebins_in[1,i]/Eb_cgb, indices_cgb[1])+pow(Ebins_in[1,i]/Eb_cgb, indices_cgb[2]))+norm_cgb/(pow(Ebins_in[2,i]/Eb_cgb, indices_cgb[1])+pow(Ebins_in[2,i]/Eb_cgb, indices_cgb[2])));\n"
                "\t}\n"
            )

        text = text + "}\n\n"
        return text

    def model_block(self):
        text = "model { \n"

        # Priors - Fixed at the moment!:
        # TODO Use config file to get the priors!
        if self._use_fixed_global_sources:
            text += "\tlog_norm_fixed ~ normal(mu_norm_fixed, sigma_norm_fixed);\n"

        if self._use_free_earth:
            text += "\talpha_earth ~ normal(-5, 2);\n"
            text += "\tEc_earth ~ normal(30,4);\n"
            text += "\tlog_norm_earth ~ normal(-4.1,0.5);\n"

        if self._use_free_cgb:
            text += "\tindices_cgb[1] ~ normal(1.32, 0.2);\n"
            text += "\tindices_cgb[2] ~ normal(2.88, 0.3);\n"
            text += "\tEb_cgb ~ normal(35,5);\n"
            text += "\tlog_norm_cgb ~ normal(-2.3,0.5);\n"

        if self._use_free_ps:
            text += "\tindex_free_ps ~ normal(3,1);\n"
            text += "\tlog_norm_free_ps ~ normal(0,1);\n"

        if self._use_cont_sources:
            text += (
                "\tfor (d in 1:num_dets){\n"
                "\t\tfor (g in 1:num_echans){\n"
                "\t\t\tlog_norm_cont[:,d,g] ~ normal(mu_norm_cont[:,d,g], sigma_norm_cont[:,d,g]);\n"
                "\t\t}\n\t}\n"
            )

        if self._use_saa:
            text += (
                "\tfor (d in 1:num_dets){\n"
                "\t\tfor (e in 1:num_echans){\n"
                "\t\t\tlog_norm_saa[:,d,e] ~ normal(mu_norm_saa[:, d, e],sigma_norm_saa[:, d, e]);\n"
                "\t\t\tdecay_saa[:,d,e] ~ lognormal(mu_decay_saa[:, d, e],sigma_decay_saa[:, d, e]);\n"
                "\t\t}\n\t}\n"
            )

        # Reduce sum call
        main = "\ttarget += reduce_sum(partial_sum, counts, grainsize\n"
        if self._use_fixed_global_sources:
            main += "\t\t, base_counts_array, norm_fixed\n"

        if self._use_free_earth:
            main += "\t\t, base_response_array_earth, earth_spec\n"

        if self._use_free_cgb:
            main += "\t\t, base_response_array_cgb, cgb_spec\n"

        if self._use_free_ps:
            main += "\t\t, base_response_array_free_ps, ps_spec\n"

        if self._use_cont_sources:
            main += "\t\t, norm_cont_vec,  base_counts_array_cont\n"

        if self._use_saa:
            main += "\t\t, t_t0, saa_decay_vec, saa_norm_vec\n"

        main += "\t);\n"

        text = text + main + "}\n\n"
        return text

    def generated_block(self):
        text = "generated quantities { \n"

        text += "\tint ppc[num_data_points];\n"
        text += "\tvector[num_data_points] tot=rep_vector(0.0, num_data_points);\n"

        if self._use_saa:
            text += "\tvector[num_data_points] f_saa;\n"

        if self._use_cont_sources:
            text += "\tvector[num_data_points] f_cont[num_cont_comp];\n"

        if self._use_fixed_global_sources:
            text += "\tvector[num_data_points] f_fixed_global[num_fixed_comp];\n"

        if self._use_free_earth:
            text += "\tvector[num_data_points] f_earth;\n"

        if self._use_free_cgb:
            text += "\tvector[num_data_points] f_cgb;\n"

        if self._use_free_ps:
            text += "\tvector[num_data_points] f_free_ps[num_free_ps_comp];\n"

        if self._use_saa:
            text += "\tf_saa = saa_total(saa_norm_vec, saa_decay_vec, t_t0, num_data_points, num_saa_exits);\n"
            text += "\ttot += f_saa;\n"

        if self._use_cont_sources:
            text += (
                "\tfor (i in 1:num_cont_comp){\n"
                "\t\tf_cont[i] = norm_cont_vec[i].*base_counts_array_cont[i];\n"
                "\t\ttot += f_cont[i];\n"
                "\t}\n"
            )

        if self._use_fixed_global_sources:
            text += (
                "\tfor (i in 1:num_fixed_comp){\n"
                "\t\tf_fixed_global[i]=norm_fixed[i]*base_counts_array[i];\n"
                "\t\ttot+=f_fixed_global[i];\n"
                "\t}\n"
            )

        if self._use_free_earth:
            text += "\tf_earth = base_response_array_earth*earth_spec;\n"
            text += "\ttot += f_earth;\n"

        if self._use_free_cgb:
            text += "\tf_cgb = base_response_array_cgb*cgb_spec;\n"
            text += "\ttot += f_cgb;\n"

        if self._use_free_ps:
            text += (
                "\tfor (i in 1:num_free_ps_comp){\n"
                "\t\tf_free_ps[i]=base_response_array_free_ps[i]*ps_spec[i];\n"
                "\t\ttot+=f_free_ps[i];\n"
                "\t}\n"
            )

        text += "\tppc = poisson_rng(tot);\n"

        text = text + "}\n\n"
        return text

    def generated_quantities(self):
        keys = []

        keys.append("tot")

        if self._use_cont_sources:
            keys.append("f_cont")

        if self._use_fixed_global_sources:
            keys.append("f_fixed_global")

        if self._use_free_earth:
            keys.append("f_earth")

        if self._use_free_cgb:
            keys.append("f_cgb")

        if self._use_free_ps:
            keys.append("free_ps")

        if self._use_saa:
            keys.append("f_saa")

        return keys

    def generated_block_total_only(self):
        text = "generated quantities { \n"

        text += "\tint ppc[num_data_points];\n"
        text += "\tvector[num_data_points] tot=rep_vector(0.0, num_data_points);\n"

        if self._use_saa:
            text += "\ttot += saa_total(saa_norm_vec, saa_decay_vec, t_t0, num_data_points, num_saa_exits);\n"

        if self._use_cont_sources:
            text += (
                "\tfor (i in 1:num_cont_comp){\n"
                "\t\ttot += norm_cont_vec[i].*base_counts_array_cont[i];\n"
                "\t}\n"
            )

        if self._use_fixed_global_sources:
            text += (
                "\tfor (i in 1:num_fixed_comp){\n"
                "\t\ttot +=norm_fixed[i]*base_counts_array[i];\n"
                "\t}\n"
            )

        if self._use_free_earth:
            text += "\ttot += base_response_array_earth*earth_spec;\n"

        if self._use_free_cgb:
            text += "\t tot+= base_response_array_cgb*cgb_spec;\n"

        if self._use_free_ps:
            text += (
                "\tfor (i in 1:num_free_ps_comp){\n"
                "\t\ttot +=base_response_array_free_ps[i]*ps_spec[i];\n"
                "\t}\n"
            )

        text += "\tppc = poisson_rng(tot);\n"

        text = text + "}\n\n"
        return text


class StanDataConstructor(object):
    """
    Object to construct the data dictionary for stan!
    """

    def __init__(
        self,
        data=None,
        model=None,
        response=None,
        geometry=None,
        model_generator=None,
        threads_per_chain=1,
    ):
        """
        Init with data, model, response and geometry object or model_generator object
        """

        if model_generator is None:
            self._data = data
            self._model = model
            self._response = response
            self._geometry = geometry
        else:
            self._data = model_generator.data
            self._model = model_generator.model
            self._response = model_generator.response
            self._geometry = model_generator.geometry

        self._threads = threads_per_chain

        self._dets = self._data.detectors
        self._echans = self._data.echans
        self._time_bins = self._data.time_bins

        self._time_bin_edges = np.append(self._time_bins[:, 0], self._time_bins[-1, 1])

        self._ndets = len(self._dets)
        self._nechans = len(self._echans)
        self._ntimebins = len(self._time_bins)

        self._param_lookup = []

    def global_sources(self):
        """
        Fixed photon sources (e.g. point sources or CGB/Earth if spectrum not fitted)
        """

        s = self._model.global_sources

        if len(s) == 0:
            self._global_counts = None
            return None

        mu_norm_fixed = np.zeros(len(s))
        sigma_norm_fixed = np.zeros(len(s))
        global_counts = np.zeros((len(s), self._ntimebins, self._ndets, self._nechans))

        for i, k in enumerate(s.keys()):
            global_counts[i] = s[k].get_counts(self._time_bins)

            for p in s[k].parameters.values():

                if "norm" in p.name:
                    if p.gaussian_parameter[0] is not None:
                        mu_norm_fixed[i] = p.gaussian_parameter[0]
                    else:
                        mu_norm_fixed[i] = 0

                    if p.gaussian_parameter[1] is not None:
                        sigma_norm_fixed[i] = p.gaussian_parameter[1]
                    else:
                        sigma_norm_fixed[i] = 1

                    self._param_lookup.append(
                        {
                            "name": p.name,
                            "idx_in_model": self._model.parameter_names.index(p.name),
                            "stan_param_name": f"norm_fixed[{i+1}]",
                            "scale": 1,
                        }
                    )
                else:
                    raise Exception("Unknown parameter name")

        # Flatten along time, detectors and echans
        global_counts = global_counts[:, 2:-2].reshape(len(s), -1)

        self._global_counts = global_counts
        self._mu_norm_fixed = mu_norm_fixed
        self._sigma_norm_fixed = sigma_norm_fixed

    def continuum_sources(self):
        """
        Sources with an independent norm per echan and detector (Cosmic rays).
        At the moment hard coded for 2 sources (Constant and CosmicRay)
        """

        if len(self._model.continuum_sources) == 0:
            self._cont_counts = None
            return None

        # In the python code we have an individual source for every echan. For the stan code we need one in total.
        num_cont_sources = 2

        continuum_counts = np.zeros(
            (num_cont_sources, self._ntimebins, self._ndets, self._nechans)
        )
        mu_norm_cont = np.zeros((num_cont_sources, self._ndets, self._nechans))
        sigma_norm_cont = np.zeros((num_cont_sources, self._ndets, self._nechans))

        for i, s in enumerate(list(self._model.continuum_sources.values())):
            if "constant" in s.name.lower():
                index = 0
            else:
                index = 1
            continuum_counts[index, :, :, s.echan] = s.get_counts(self._time_bins)

            for p in s.parameters.values():

                if "norm" in p.name:
                    if p.gaussian_parameter[0] is not None:
                        mu_norm_cont[:, :, s.echan] = p.gaussian_parameter[0]
                    else:
                        mu_norm_cont[:, :, s.echan] = 0

                    if p.gaussian_parameter[1] is not None:
                        sigma_norm_cont[:, :, s.echan] = p.gaussian_parameter[1]
                    else:
                        sigma_norm_cont[:, :, s.echan] = 1

                    for det_idx, det in enumerate(self._dets):
                        self._param_lookup.append(
                            {
                                "name": f"{p.name}_{det}",
                                "idx_in_model": self._model.parameter_names.index(
                                    p.name
                                ),
                                "stan_param_name": f"norm_cont[{index+1},{det_idx + 1},{s.echan + 1}]",
                                "scale": 1,
                            }
                        )
                else:
                    raise Exception("Unknown parameter name")

        self._cont_counts = continuum_counts[:, 2:-2].reshape(2, -1)
        self._mu_norm_cont = mu_norm_cont
        self._sigma_norm_cont = sigma_norm_cont

    def free_spectrum_sources(self):
        """
        Free spectrum sources
        """

        s = self._model.fit_spectrum_sources

        self._Ebins_in = np.vstack(
            (
                self._response.responses[self._dets[0]].Ebin_in_edge[:-1],
                self._response.responses[self._dets[0]].Ebin_in_edge[1:],
            )
        )

        self._num_Ebins_in = len(self._Ebins_in[0])

        self._base_response_array_earth = None
        self._base_response_array_cgb = None
        self._base_response_array_ps = None

        if len(s) == 0:
            return None

        base_response_array_earth = None
        base_response_array_cgb = None
        base_rsp_ps_free = None

        for k in s.keys():
            rsp_detectors = s[k]._shape._effective_responses
            ar = np.zeros(
                (
                    self._ndets,
                    len(self._geometry.geometry_times),
                    self._num_Ebins_in,
                    self._nechans,
                )
            )
            for i, det in enumerate(self._dets):
                ar[i] = rsp_detectors[det]
            if k == "Earth occultation":
                base_response_array_earth = ar
            elif k == "CGB":
                base_response_array_cgb = ar
            else:
                if base_rsp_ps_free is not None:
                    base_rsp_ps_free = np.append(
                        base_rsp_ps_free, np.array([ar]), axis=0
                    )
                else:
                    base_rsp_ps_free = np.array([ar])

        if base_response_array_earth is not None:

            eff_rsp_new_earth = interp1d(
                self._geometry.geometry_times, base_response_array_earth, axis=1
            )

            rsp_all_earth = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_earth(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                0,
                1,
            )

            # Trapz integrate over time bins
            base_response_array_earth = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_earth[:-1] + rsp_all_earth[1:])
            )

            self._base_response_array_earth = base_response_array_earth[2:-2].reshape(
                -1, self._num_Ebins_in
            )

        if base_response_array_cgb is not None:

            eff_rsp_new_cgb = interp1d(
                self._geometry.geometry_times, base_response_array_cgb, axis=1
            )

            rsp_all_cgb = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_cgb(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                0,
                1,
            )

            # Trapz integrate over time bins
            base_response_array_cgb = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_cgb[:-1] + rsp_all_cgb[1:])
            )

            self._base_response_array_cgb = base_response_array_cgb[2:-2].reshape(
                -1, self._num_Ebins_in
            )

        if base_rsp_ps_free is not None:
            eff_rsp_new_free_ps = interp1d(
                self._geometry.geometry_times, base_rsp_ps_free, axis=2
            )

            rsp_all_ps = np.swapaxes(
                np.array(
                    np.swapaxes(eff_rsp_new_free_ps(self._time_bin_edges), -1, -2),
                    dtype=float,
                ),
                1,
                2,
            )

            # Trapz integrate over time bins
            base_rsp_ps_free = (
                0.5
                * (
                    self._time_bins[:, 1, np.newaxis, np.newaxis, np.newaxis]
                    - self._time_bins[:, 0, np.newaxis, np.newaxis, np.newaxis]
                )
                * (rsp_all_ps[:, :-1] + rsp_all_ps[:, 1:])
            )

            self._base_response_array_ps = base_rsp_ps_free[:, 2:-2].reshape(
                base_rsp_ps_free.shape[0], -1, self._num_Ebins_in
            )

    def saa_sources(self):
        """
        The Saa exit sources
        """
        # One source per exit (not per exit and echan like in the python code)
        self._num_saa_exits = int(len(self._model.saa_sources) / self._nechans)

        mu_norm_saa = np.zeros((self._num_saa_exits, self._ndets, self._nechans))
        sigma_norm_saa = np.zeros((self._num_saa_exits, self._ndets, self._nechans))
        mu_decay_saa = np.zeros((self._num_saa_exits, self._ndets, self._nechans))
        sigma_decay_saa = np.zeros((self._num_saa_exits, self._ndets, self._nechans))

        saa_start_times = np.zeros(self._num_saa_exits)

        for i, s in enumerate(
            list(self._model.saa_sources.values())[: self._num_saa_exits]
        ):
            saa_start_times[i] = s._shape._saa_exit_time[0]

        for i, s in enumerate(list(self._model.saa_sources.values())):

            if s._shape._det_idx is None:
                det_idx = np.arange(0, self._ndets)
                det_idx_stan = 1
            else:
                det_idx = s._shape._det_idx
                det_idx_stan = det_idx + 1

            for p in s.parameters.values():

                if "norm" in p.name:
                    if p.gaussian_parameter[0] is not None:
                        mu_norm_saa[:, det_idx, s.echan] = p.gaussian_parameter[0]
                    else:
                        mu_norm_saa[:, det_idx, s.echan] = 0

                    if p.gaussian_parameter[1] is not None:
                        sigma_norm_saa[:, det_idx, s.echan] = p.gaussian_parameter[1]
                    else:
                        sigma_norm_saa[:, det_idx, s.echan] = 1

                    self._param_lookup.append(
                        {
                            "name": p.name,
                            "idx_in_model": self._model.parameter_names.index(p.name),
                            "stan_param_name": f"norm_saa[{i+1},{det_idx_stan},{s.echan+1}]",
                            "scale": 1,
                        }
                    )
                elif "decay" in p.name:
                    if p.gaussian_parameter[0] is not None:
                        mu_decay_saa[:, det_idx, s.echan] = p.gaussian_parameter[0]
                    else:
                        mu_decay_saa[:, det_idx, s.echan] = 0

                    if p.gaussian_parameter[1] is not None:
                        sigma_decay_saa[:, det_idx, s.echan] = p.gaussian_parameter[1]

                    else:
                        sigma_decay_saa[:, det_idx, s.echan] = 1

                    self._param_lookup.append(
                        {
                            "name": p.name,
                            "idx_in_model": self._model.parameter_names.index(p.name),
                            "stan_param_name": f"decay_saa[{i+1},{det_idx_stan},{s.echan+1}]",
                            "scale": 0.0001,
                        }
                    )
                else:
                    raise Exception("Unknown parameter name")

        self._saa_start_times = saa_start_times
        self._mu_norm_saa = mu_norm_saa
        self._sigma_norm_saa = sigma_norm_saa
        self._mu_decay_saa = mu_decay_saa
        self._sigma_decay_saa = sigma_decay_saa

    def construct_data_dict(self):
        self.global_sources()
        self.continuum_sources()
        self.saa_sources()
        self.free_spectrum_sources()

        data_dict = {}

        data_dict["num_dets"] = self._ndets
        data_dict["num_echans"] = self._nechans

        counts = np.array(self._data.counts[2:-2], dtype=int).flatten()
        mask_zeros = np.array(self._data.counts[2:-2], dtype=int).flatten() != 0

        data_dict["counts"] = np.array(self._data.counts[2:-2], dtype=int).flatten()[
            mask_zeros
        ]
        data_dict["time_bins"] = self._data.time_bins[2:-2][
            mask_zeros[:: self._ndets * self._nechans]
        ]
        data_dict["num_time_bins"] = len(data_dict["time_bins"])

        data_dict["rsp_num_Ein"] = self._num_Ebins_in
        data_dict["Ebins_in"] = self._Ebins_in

        # Global sources
        if self._global_counts is not None:
            data_dict["num_fixed_comp"] = len(self._global_counts)
            data_dict["base_counts_array"] = self._global_counts[:, mask_zeros]
            data_dict["mu_norm_fixed"] = self._mu_norm_fixed
            data_dict["sigma_norm_fixed"] = self._sigma_norm_fixed
        else:
            raise NotImplementedError

        if self._base_response_array_ps is not None:
            data_dict["num_free_ps_comp"] = len(self._base_response_array_ps)
            data_dict["base_response_array_free_ps"] = self._base_response_array_ps[
                :, mask_zeros
            ]
        if self._base_response_array_earth is not None:
            data_dict["base_response_array_earth"] = self._base_response_array_earth[
                mask_zeros
            ]
        if self._base_response_array_cgb is not None:
            data_dict["base_response_array_cgb"] = self._base_response_array_cgb[
                mask_zeros
            ]

        if self._base_response_array_cgb is not None:
            data_dict["earth_cgb_free"] = 1
        else:
            data_dict["earth_gb_free"] = 0

        if len(self._model.saa_sources) > 0:
            data_dict["num_saa_exits"] = self._num_saa_exits
            data_dict["saa_start_times"] = self._saa_start_times
            data_dict["mu_norm_saa"] = self._mu_norm_saa
            data_dict["sigma_norm_saa"] = self._sigma_norm_saa
            data_dict["mu_decay_saa"] = self._mu_decay_saa
            data_dict["sigma_decay_saa"] = self._sigma_decay_saa

        if self._cont_counts is not None:
            data_dict["num_cont_comp"] = 2
            data_dict["base_counts_array_cont"] = self._cont_counts[:, mask_zeros]
            data_dict["mu_norm_cont"] = self._mu_norm_cont
            data_dict["sigma_norm_cont"] = self._sigma_norm_cont

        # Stan grainsize for reduced_sum
        if self._threads == 1:
            data_dict["grainsize"] = 1
        else:
            data_dict["grainsize"] = int(
                (self._ntimebins - 4) * self._ndets * self._nechans / self._threads
            )
        return data_dict

    @property
    def param_lookup(self):
        return self._param_lookup


class ReadStanArvizResult(object):
    def __init__(self, nc_files):
        for i, nc_file in enumerate(nc_files):
            if i == 0:
                self._arviz_result = av.from_netcdf(nc_file)
            else:
                self._arviz_result = av.concat(
                    self._arviz_result, av.from_netcdf(nc_file), dim="chain"
                )

        self._model_parts = self._arviz_result.predictions.keys()

        self._dets = self._arviz_result.constant_data["dets"].values
        self._echans = self._arviz_result.constant_data["echans"].values

        self._ndets = len(self._dets)
        self._nechans = len(self._echans)

        self._time_bins = self._arviz_result.constant_data["time_bins"].values
        self._time_bins -= self._time_bins[0, 0]
        self._bin_width = self._time_bins[:, 1] - self._time_bins[:, 0]

        self._counts = self._arviz_result.observed_data["counts"].values

        predictions = self._arviz_result.predictions.stack(sample=("chain", "draw"))
        self._parts = {}
        for key in self._model_parts:
            self._parts[key] = predictions[key].values

        self._ppc = self._arviz_result.posterior_predictive.stack(
            sample=("chain", "draw")
        )["ppc"].values

    def ppc_plots(self, save_dir):

        colors = {
            "f_fixed": "red",
            "f_ps": "red",
            "f_saa": "navy",
            "f_cont": "magenta",
            "f_earth": "purple",
            "f_cgb": "cyan",
            "tot": "green",
        }

        for d_index, d in enumerate(self._dets):
            for e_index, e in enumerate(self._echans):

                mask = np.arange(len(self._counts), dtype=int)[
                    e_index + d_index * self._ndets :: self._ndets * self._nechans
                ]
                fig, ax = plt.subplots()

                for i in np.linspace(0, self._ppc.shape[1] - 1, 30, dtype=int):
                    if i == 0:
                        ax.scatter(
                            np.mean(self._time_bins, axis=1),
                            self._ppc[mask][:, i] / self._bin_width,
                            color="darkgreen",
                            alpha=0.025,
                            edgecolor="green",
                            facecolor="none",
                            lw=0.9,
                            s=2,
                            label="PPC",
                        )
                    else:
                        ax.scatter(
                            np.mean(self._time_bins, axis=1),
                            self._ppc[mask][:, i] / self._bin_width,
                            color="darkgreen",
                            alpha=0.025,
                            edgecolor="darkgreen",
                            facecolor="none",
                            lw=0.9,
                            s=2,
                        )

                    for key in self._parts.keys():
                        # Check if there are several sources in this class
                        if len(self._parts[key].shape) == 3:
                            for k in range(len(self._parts[key])):
                                if k == 0 and i == 0:
                                    ax.scatter(
                                        np.mean(self._time_bins, axis=1),
                                        self._parts[key][k][mask][:, i]
                                        / self._bin_width,
                                        alpha=0.025,
                                        edgecolor=colors.get(key, "gray"),
                                        facecolor="none",
                                        lw=0.9,
                                        s=2,
                                        label=key,
                                    )
                                else:
                                    ax.scatter(
                                        np.mean(self._time_bins, axis=1),
                                        self._parts[key][k][mask][:, i]
                                        / self._bin_width,
                                        alpha=0.025,
                                        edgecolor=colors.get(key, "gray"),
                                        facecolor="none",
                                        lw=0.9,
                                        s=2,
                                    )
                        else:
                            if i == 0:
                                ax.scatter(
                                    np.mean(self._time_bins, axis=1),
                                    self._parts[key][mask][:, i] / self._bin_width,
                                    alpha=0.025,
                                    edgecolor=colors.get(key, "gray"),
                                    facecolor="none",
                                    lw=0.9,
                                    s=2,
                                    label=key,
                                )
                            else:
                                ax.scatter(
                                    np.mean(self._time_bins, axis=1),
                                    self._parts[key][mask][:, i] / self._bin_width,
                                    alpha=0.025,
                                    edgecolor=colors.get(key, "gray"),
                                    facecolor="none",
                                    lw=0.9,
                                    s=2,
                                )

                ax.scatter(
                    np.mean(self._time_bins, axis=1),
                    self._counts[mask] / self._bin_width,
                    color="darkgreen",
                    alpha=0.25,
                    edgecolor="black",
                    facecolor="none",
                    lw=0.9,
                    s=2,
                    label="Data",
                )
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
                lgd = fig.legend(loc="center right")  # , bbox_to_anchor=(1, 0.5))
                for lh in lgd.legendHandles:
                    lh.set_alpha(1)
                t = fig.suptitle(f"Detector {d} - Echan {e}")
                fig.savefig(
                    f"ppc_result_det_{d}_echan_{e}.png"
                )  # , bbox_extra_artists=(lgd,t), dpi=450, bbox_inches='tight')

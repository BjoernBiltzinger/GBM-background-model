#!/user/bin/env python3
import netCDF4
import arviz as av
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from datetime import datetime

from gbmbkgpy.utils.stan import StanDataConstructor
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot import Plotter
from gbmbkgpy.io.export import DataExporter

from cmdstanpy import cmdstan_path, CmdStanModel
import os

start = datetime.now()

config_file = "config_stan_saa.yml"

# Load the config.yml
with open(config_file) as f:
    config = yaml.load(f)

############## Generate the GBM-background-model ##################
model_generator = BackgroundModelGenerator()

model_generator.from_config_dict(config)


# Create Stan Model
model = CmdStanModel(stan_file="stan_model_saa.stan",
                     cpp_options={'STAN_THREADS': 'TRUE'})


# Number of threads per Chain
threads_per_chain = 64


# StanDataConstructor
stan_data = StanDataConstructor(model_generator=model_generator,
                                threads_per_chain=threads_per_chain)

data_dict =  stan_data.construct_data_dict()

# Sample
fit = model.sample(data=data_dict,
                   output_dir="./",
                   chains=1,
                   seed=int(np.random.rand()*10000),
                   parallel_chains=1,
                   threads_per_chain=threads_per_chain,
                   iter_warmup=300,
                   iter_sampling=300,
                   show_progress=True)


# Buiöd arviz object
ar = av.from_cmdstanpy(fit,
                       posterior_predictive="ppc",
                       observed_data={"counts": data_dict["counts"]},
                       constant_data={"time_bins": data_dict["time_bins"],
                                     "dets": model_generator.data.detectors,
                                     "echans": model_generator.data.echans},
                       predictions=["f_fixed","f_saa","f_cont"])


# Save this object
dateTimeObj = datetime.now()
stamp = dateTimeObj.strftime("%d_%b_%Y-%H%M%S")
ar.to_netcdf(f"saa_fit_res_{stamp}.nc")

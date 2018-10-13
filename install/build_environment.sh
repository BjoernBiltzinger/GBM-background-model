#!/usr/bin/env bash

# This installation Script install all dependencies for the Background Model.
#
# The install location has to be passed as an argument.
#
# All packages are cloned to the Install location and installed to the virtual environment
#
# On a development system the GBM_Background model should be installed with python setup.py develop
#
# Author: Felix Kunzweiler


if [ -z "$1" ]
  then
    echo "No working_directory supplied, please run script as following: ./build_environment.sh {virtual_env_name} {path_to_working_dir}"
    exit
fi

cp install_3ML.sh $1
cp installMultiNest.sh $1
cd $1

echo "Install ThreeML"
/bin/bash install_3ML.sh

echo "Activating virtual environment threeML"
source activate threeML

echo "Setting environment variables"
echo 'export LD_LIBRARY_PATH='$1'/MultiNest/lib:'$LD_LIBRARY_PATH >> ~/.bashrc
echo 'export MULTINEST='$1'/MultiNest' >> ~/.bashrc
echo 'export GBM_DATA='$1'/gbm_data' >> ~/.bashrc


echo "Installing python packages"
pip install --upgrade pip
pip install numpy scipy ipython jupyter pymultinest matplotlib h5py seaborn pandas pytest cython mpi4py spherical_geometry healpy sympy astropy==2.0.6 --ignore-installed six

echo "Install MultiNest"
/bin/bash installMultiNest.sh $1

echo "Change to work directory $1"
mkdir gmb_data

echo "Install astromodels"
git clone https://github.com/giacomov/astromodels.git
cd astromodels  && python setup.py install
cd ../ # && rm -rf astromodels

echo "Install gbmgeometry"
git clone https://github.com/grburgess/gbmgeometry.git
cd gbmgeometry  && python setup.py install
cd ../ #&& rm -rf gbmgeometry

echo "Install gbm_drm_gen"
ssh-agent bash -c 'ssh-add ~/.ssh/gbm_drm_gen_sshkey; git clone  git@github.com:grburgess/gbm_drm_gen.git'
cd gbm_drm_gen && python setup.py install
cd ../ #&& rm -rf gbm_drm_gen

echo "Install GBM-background-model"
ssh-agent bash -c 'ssh-add ~/.ssh/gbm_background_sshkey; git clone git@github.com:grburgess/GBM-background-model.git'
cd GBM-background-model && python setup.py install
cd ../ # && rm -rf GBM-background-model

source deactivate

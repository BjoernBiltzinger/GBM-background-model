#!/bin/bash

# Bash script that loops over all detectors and fits/echans to create the output plots

declare -a dets=("n0" "n1" "n2" "n3" "n4" "n5" "n6" "n7" "n8" "n9" "na" "nb")

for det in "${dets[@]}"
do
    FITS=(`find "/home/fkunzwei/data1/gbm_data/fits/mn_out/test_2days_${det}_0" -mindepth 1 -maxdepth 1 -type d`)

    for fit_path in "${FITS[@]}"
    do
        echo "bash -c 'python plt_results.py -c config_plot_default --data_path $fit_path/data_for_plots.hdf5'"
        bash -c "python plt_results.py -c config_plot_default --data_path $fit_path/data_for_plots.hdf5"
    done
done

#!/bin/sh

# Bash script that loops over all detectors and energy channels and calls general_script
# Pass number of cores as argument

declare -a dets=("n0"
		"n1"
		"n2"
		"n3"
		"n4"
		"n5"
		"n6"
		"n7"
		"n8"
		"n9"
		"na"
		"nb")

for det in "${dets[@]}"
do
    for echan in 0 1 2 3 4 5 6 7
    do
	bash nice -19 mpiexec -n $1 python general_script.py  -d $det -e $echan
    done
done

#!/bin/bash
param=$1
# Grab reference parameter value
a=$(grep "$param" reference_parameters.txt)
arr=($a)
# One percent variation in the parameter
new_val=$(echo 1.01*"${arr[1]}"|bc -l)
# Create simulation directory
mkdir results/"$param"
# Generate parameter file
sed "s/${arr[0]} $PARTITION_COLUMN.*/${arr[0]} $new_val/" reference_parameters.txt > results/"$param"/modified_parameters.txt
# Run simulation
cd ../
python3 simulate_bulk.py sensitivity/results/"$param"/modified_parameters.txt sensitivity/dib_total_concs.txt ./sensitivity/results/"$param"/ sensitivity

#!/bin/bash

# List of arguments
N_OBS=(10 30 50 100 200 300 400 500 600)
RUNS=(0)

# Initialize empty data_out_PINNSR
python init_data_out_PINNSR.py

# Loop through each argument and run the Python script
for n_obs in "${N_OBS[@]}"; do
    for run in "${RUNS[@]}"; do
        python coefficients.py "$n_obs" "$run"
    done
done
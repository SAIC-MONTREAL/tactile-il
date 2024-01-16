#!/bin/bash

DATA_DIR_NAME=$1
GPUS=$2    # as comma separated string, e.g. "1,2,4"

gpu_arr=(${GPUS//,/ })
seed=1

for gpu in "${gpu_arr[@]}"; do
    echo "Starting parallel job on dataset ${DATA_DIR_NAME}, gpu ${gpu}, seed ${seed}"
    python -m contact_il.train_bc data_dir_name="${DATA_DIR_NAME}" device="cuda:${gpu}" $3 $4 $5 &
    seed=$((seed+1))
done
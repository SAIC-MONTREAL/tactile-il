#!/bin/bash

DATA_DIR_NAME=$1
NUM_SEEDS=$2

for (( seed=1; seed<=$NUM_SEEDS; seed++ )); do
    echo "Starting sequential job on dataset ${DATA_DIR_NAME}, seed ${seed}"
    python -m contact_il.train_bc data_dir_name="${DATA_DIR_NAME}" random_seed="${seed}" $3 $4 $5
done
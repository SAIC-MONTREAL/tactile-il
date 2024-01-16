#!/bin/bash

DATA_DIR_NAME=$1
NUM_SEEDS=$2

obs_list_strs=(
    'model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose']'
    'model_config.obs_key_list=['wrist_rgb','pose','prev_pose']'
    'model_config.obs_key_list=['sts_raw_image','pose','prev_pose']'
    'model_config.obs_key_list=['wrist_rgb']'
    'model_config.obs_key_list=['sts_raw_image']'
    'model_config.obs_key_list=['pose','prev_pose']'
)

id_strs=(
    'id=wrist_rgb-sts_raw_image-pose-prev_pose'
    'id=wrist_rgb-pose-prev_pose'
    'id=sts_raw_image-pose-prev_pose'
    'id=wrist_rgb'
    'id=sts_raw_image'
    'id=pose-prev_pose'
)

id_str_i=0
for id_str in "${id_strs[@]}"; do
    for (( seed=1; seed<=$NUM_SEEDS; seed++ )); do
        echo "Starting sequential job on dataset ${DATA_DIR_NAME}, seed ${seed}, observations ${id_str}"
        python -m contact_il.train_bc data_dir_name=${DATA_DIR_NAME} random_seed=${seed} ${id_str} ${obs_list_strs[id_str_i]} $3 $4 $5
    done
    id_str_i=$((id_str_i+1))
done



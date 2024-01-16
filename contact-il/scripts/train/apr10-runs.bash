#!/bin/bash

seed=$1
device=$2
ID="apr9-higher_quality"
DATA_DIR_NAMES=(
    'PandaTopGlassOrbNoSTSSwitch/higher_quality_tactile_only_1'
    'PandaTopGlassOrb/higher_quality_no_ft_adapt'
)

DATA_AMOUNTS=(
    '15'
    '10'
    '5'
)

obs_list_strs=(
    'model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose']'
)

obs_strs=(
    'obs_str=wrist_rgb-sts_raw_image-pose-prev_pose'
)


for data_dir_name in "${DATA_DIR_NAMES[@]}"; do

    obs_str_i=0
    for obs_str in "${obs_strs[@]}"; do

        for data_amt in "${DATA_AMOUNTS[@]}"; do

            echo "Training ${data_dir_name} with ${data_amt} eps with ${obs_str} obs on ${device}, seed ${seed}"

            python -m contact_il.train_bc data_dir_name=${data_dir_name} random_seed=${seed} ${obs_str} \
                ${obs_list_strs[obs_str_i]} dataset_config.n_max_episodes=${data_amt} id=${ID} device=${device}
        done
        obs_str_i=$((obs_str_i+1))
    done

done
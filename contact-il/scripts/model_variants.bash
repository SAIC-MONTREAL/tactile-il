#!/bin/bash

base_env=$1
sub_name=$2

MAIN_DATA_AMT=20
pre_data_strs=(
    "data_dir_name=${base_env}"
    "data_dir_name=${base_env}Close"
)

post_data_strs=(
    ""
    "Close"
)

data_dir_sub_strs=(
    "/${sub_name}"
    "TactileOnly/${sub_name}"
    "VisualOnly/${sub_name}"
    "/${sub_name}_simple_contact"
    "/${sub_name}_no_fa"
    "VisualOnly/${sub_name}_no_fa"
)

reset_data_dir_sub_strs=(
    "/${reset_sub_name}"
    "TactileOnly/${reset_sub_name}"
    "VisualOnly/${reset_sub_name}"
    "/${reset_sub_name}_simple_contact"
    "/${reset_sub_name}_no_fa"
    "VisualOnly/${reset_sub_name}_no_fa"
)

extra_data_amounts=(
    '15'
    '10'
    '5'
)

obs_list_strs_no_keys=(
    'wrist_rgb-sts_raw_image-pose-prev_pose'
    'wrist_rgb-pose-prev_pose'
    'sts_raw_image-pose-prev_pose'
    'wrist_rgb'
    'sts_raw_image'
    'pose-prev_pose'
    'wrist_rgb-sts_raw_image'
)

obs_keys_strs=(
    "['wrist_rgb','sts_raw_image','pose','prev_pose']"
    "['wrist_rgb','pose','prev_pose']"
    "['sts_raw_image','pose','prev_pose']"
    "['wrist_rgb']"
    "['sts_raw_image']"
    "['pose','prev_pose']"
    "['wrist_rgb','sts_raw_image']"
)

MAIN_VARIANT_IDX_COMBOS=(
    "0 0"
    "1 0"
    "2 0"
    "0 2"
    "1 2"
    "2 2"
    "4 0"
    "5 1"
)

DATA_VARIANT_IDX_COMBOS=(
    "0 0"
    "0 2"
    "5 1"
    "2 1"
)

OBS_VARIANT_IDX_COMBOS=(
    "0 0"
    "2 1"
    "0 2"
    "2 3"
    "0 4"
    "2 5"
    "0 6"
)
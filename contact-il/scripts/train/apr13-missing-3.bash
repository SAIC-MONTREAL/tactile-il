#!/bin/bash

# e.g. bash all-runs-one-seed.bash 1 cuda:5 PandaTopBlackFlatHandle double_env_lower_max_fix apr12-first-try

seed=$1
device=$2
base_env=$3
sub_name=$4
ID=$5

MAIN_DATA_AMT=20
pre_data_strs=(
    "data_dir_name=${base_env}Close"
)

data_dir_sub_strs=(
    "/${sub_name}"
    "TactileOnly/${sub_name}"
    "VisualOnly/${sub_name}"
    "/${sub_name}_simple_contact"
    "/${sub_name}_no_fa"
    "VisualOnly/${sub_name}_no_fa"
)

extra_data_amounts=(
    '15'
    '10'
    '5'
)

obs_list_strs=(
    'model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose'] obs_str=wrist_rgb-sts_raw_image-pose-prev_pose'
    'model_config.obs_key_list=['wrist_rgb','pose','prev_pose'] obs_str=wrist_rgb-pose-prev_pose'
    'model_config.obs_key_list=['sts_raw_image','pose','prev_pose'] obs_str=sts_raw_image-pose-prev_pose'
    'model_config.obs_key_list=['wrist_rgb'] obs_str=wrist_rgb'
    'model_config.obs_key_list=['sts_raw_image'] obs_str=sts_raw_image'
    'model_config.obs_key_list=['pose','prev_pose'] obs_str=pose-prev_pose'
)

MAIN_VARIANT_STRS=(
    "${data_dir_sub_strs[3]} ${obs_list_strs[0]}"
)

DATA_VARIANT_STRS=(
    "${data_dir_sub_strs[0]} ${obs_list_strs[0]}"
    "${data_dir_sub_strs[5]} ${obs_list_strs[1]}"
    "${data_dir_sub_strs[4]} ${obs_list_strs[0]}"
    "${data_dir_sub_strs[1]} ${obs_list_strs[0]}"
)

OBS_VARIANT_STRS=(
    "${data_dir_sub_strs[2]} ${obs_list_strs[1]}"
    "${data_dir_sub_strs[2]} ${obs_list_strs[3]}"
    "${data_dir_sub_strs[0]} ${obs_list_strs[4]}"
    "${data_dir_sub_strs[2]} ${obs_list_strs[5]}"
)

i=0
for pre_data_str in "${pre_data_strs}"; do
    combined=("${MAIN_VARIANT_STRS[@]}")
    for variant in "${combined[@]}"; do
        echo "Training ${pre_data_str}${variant} on device ${device}, seed ${seed}, id ${ID}, ${MAIN_DATA_AMT} eps"
        python -m contact_il.train_bc ${pre_data_str}${variant} random_seed=${seed} id=${ID} device=${device}\
            dataset_config.n_max_episodes=${MAIN_DATA_AMT}
        i=$((i+1))
    done

    # for variant in "${DATA_VARIANT_STRS[@]}"; do
    #     for data_amt in "${extra_data_amounts[@]}"; do
    #         echo "Training ${pre_data_str}${variant} on device ${device}, seed ${seed}, id ${ID}, ${data_amt} eps"
    #         python -m contact_il.train_bc ${pre_data_str}${variant} random_seed=${seed} id=${ID} device=${device}\
    #             dataset_config.n_max_episodes=${data_amt}
    #         i=$((i+1))
    #     done
    # done

    echo "Completed ${i} training runs."

done
#!/bin/bash

# DATA_DIR_NAME=$1
# NUM_SEEDS=$2

DATA_DIR_NAMES=(
    'PandaTopGlassOrb/real_data_1'
    'PandaTopGlassOrbNoSTSSwitch/tactile_only_1'
)
NUM_SEEDS=3
DATA_AMOUNTS=(
    '40'
    '20'
)

ID="apr9-test"


# obs_list_strs=(
#     'model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose']'
#     'model_config.obs_key_list=['wrist_rgb','pose','prev_pose']'
#     'model_config.obs_key_list=['sts_raw_image','pose','prev_pose']'
#     'model_config.obs_key_list=['wrist_rgb']'
#     'model_config.obs_key_list=['sts_raw_image']'
#     'model_config.obs_key_list=['pose','prev_pose']'
# )
obs_list_strs=(
    'model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose']'
    'model_config.obs_key_list=['wrist_rgb','pose','prev_pose']'
)

# obs_strs=(
#     'obs_str=wrist_rgb-sts_raw_image-pose-prev_pose'
#     'obs_str=wrist_rgb-pose-prev_pose'
#     'obs_str=sts_raw_image-pose-prev_pose'
#     'obs_str=wrist_rgb'
#     'obs_str=sts_raw_image'
#     'obs_str=pose-prev_pose'
# )
obs_strs=(
    'obs_str=wrist_rgb-sts_raw_image-pose-prev_pose'
    'obs_str=wrist_rgb-pose-prev_pose'
)

obs_str_i=0
for obs_str in "${obs_strs[@]}"; do
    for data_amt in "${DATA_AMOUNTS[@]}"; do
        for (( seed=1; seed<=$NUM_SEEDS; seed++ )); do
            echo "Starting sequential job on dataset ${DATA_DIR_NAMES[obs_str_i]}, seed ${seed}, observations ${obs_str}, data amount ${data_amt}"
            python -m contact_il.train_bc data_dir_name=${DATA_DIR_NAMES[obs_str_i]} random_seed=${seed} ${obs_str} \
                ${obs_list_strs[obs_str_i]} dataset_config.n_max_episodes=${data_amt} id=${ID}

            if [[ "${obs_str_i}" == "0" ]] && [[ "${data_amt}" == "40" ]]; then
                echo "For main training config, also training with 100k updates for seed ${seed}"
                python -m contact_il.train_bc data_dir_name=${DATA_DIR_NAMES[obs_str_i]} random_seed=${seed} ${obs_str} \
                    ${obs_list_strs[obs_str_i]} dataset_config.n_max_episodes=${data_amt} n_grad_updates=100000 \
                    n_scheduler_grad_updates=50000 id=${ID}
            fi

        done
    done
    obs_str_i=$((obs_str_i+1))
done

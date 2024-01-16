#!/bin/bash

# e.g. bash all-runs-one-seed-new.bash PandaTopBlackFlatHandle double_env_lower_max_fix apr12-first-try 1 cuda:5
# ex resulting command: python -m contact_il.train_bc data_dir_name=PandaTopBlackFlatHandle/double_env_lower_max_fix \
#                          random_seed=1 id=apr12-first-try device=cuda:5 dataset_config.n_max_episodes=20 \
#                          model_config.obs_key_list=['wrist_rgb','sts_raw_image','pose','prev_pose'] \
#                          obs_str=wrist_rgb-sts_raw_image-pose-prev_pose

base_env=$1
sub_name=$2
reset_sub_name=$3
train_id=$4
seed=$5
device=$6

# source all variables
. ../model_variants.bash ${base_env} ${sub_name} ${reset_sub_name}

post_data_str_i=0
for post_data_str in "${post_data_strs[@]}"; do
    combined=("${MAIN_VARIANT_IDX_COMBOS[@]}" "${OBS_VARIANT_IDX_COMBOS[@]}")
    for variant in "${combined[@]}"; do
        data_dir_sub_str_i=${variant:0:1}
        if [[ "${post_data_str_i}" == "0" ]]; then
            data_dir_sub_str=${data_dir_sub_strs[${data_dir_sub_str_i}]}
        else
            data_dir_sub_str=${reset_data_dir_sub_strs[${data_dir_sub_str_i}]}
        fi
        obs_list_str_i=${variant:2:3}
        obs_list_str=${obs_list_strs_no_keys[${obs_list_str_i}]}
        obs_key_list=${obs_keys_strs[${obs_list_str_i}]}

        echo "Training ${train_id}: ${MAIN_DATA_AMT} eps for dataset ${base_env}${post_data_str}${data_dir_sub_str}, "
        echo "observations ${obs_list_str}, seed ${seed}, device ${device}."

        python -m contact_il.train_bc \
            data_dir_name=${base_env}${post_data_str}${data_dir_sub_str} \
            random_seed=${seed} \
            id=${train_id} \
            device=${device} \
            dataset_config.n_max_episodes=${MAIN_DATA_AMT} \
            model_config.obs_key_list=${obs_key_list} \
            obs_str=${obs_list_str}

    done

    for variant in "${DATA_VARIANT_IDX_COMBOS[@]}"; do
        data_dir_sub_str_i=${variant:0:1}
        if [[ "${post_data_str_i}" == "0" ]]; then
            data_dir_sub_str=${data_dir_sub_strs[${data_dir_sub_str_i}]}
        else
            data_dir_sub_str=${reset_data_dir_sub_strs[${data_dir_sub_str_i}]}
        fi
        obs_list_str_i=${variant:2:3}
        obs_list_str=${obs_list_strs_no_keys[${obs_list_str_i}]}
        obs_key_list=${obs_keys_strs[${obs_list_str_i}]}

        for data_amt in "${extra_data_amounts[@]}"; do
            echo "Training ${train_id}: ${data_amt} eps for dataset ${base_env}${post_data_str}${data_dir_sub_str}, "
            echo "observations ${obs_list_str}, seed ${seed}, device ${device}."

            python -m contact_il.train_bc \
                data_dir_name=${base_env}${post_data_str}${data_dir_sub_str} \
                random_seed=${seed} \
                id=${train_id} \
                device=${device} \
                dataset_config.n_max_episodes=${data_amt} \
                model_config.obs_key_list=${obs_key_list} \
                obs_str=${obs_list_str}
        done
    done

    post_data_str_i=$((post_data_str_i+1))

done
#!/bin/bash

base_env=$1
sub_name=$2
reset_sub_name=$3
train_id=$4
test_id=$5

# ex from the hydra cfg file for training
# ${env:CIL_DATA_DIR}/models/${data_dir_name}/${dataset_config.n_max_episodes}_eps/${obs_str}/${id}_${hydra.job.override_dirname}/${random_seed}

MAIN_DATA_AMT=20
SEEDS=(1 2 3)
N_EPISODES=10
RENDER="false"
DEVICE="cuda"
SIM="false"
# STS_VID="$STS_SIM_VID"
STS_VID=""
# STS_CONFIG_DIR_OVERRIDE=""  # uses the one from the original dataset
STS_CONFIG_DIR_OVERRIDE="${STS_CONFIG}"

# source all variables
. ../model_variants.bash ${base_env} ${sub_name}

if [[ "${SIM}" == "true" ]]; then
    PYTHON_ARGS+=" --sim"
fi

combined=("${MAIN_VARIANT_IDX_COMBOS[@]}" "${OBS_VARIANT_IDX_COMBOS[@]}")
for variant in "${combined[@]}"; do
    data_dir_sub_str_i=${variant:0:1}
    data_dir_sub_str=${data_dir_sub_strs[${data_dir_sub_str_i}]}
    reset_data_dir_sub_str=${reset_data_dir_sub_strs[${data_dir_sub_str_i}]}
    obs_list_str_i=${variant:2:3}
    obs_list_str=${obs_list_strs_no_keys[${obs_list_str_i}]}

    for seed in "${SEEDS[@]}"; do
        # model_subdir="${CIL_DATA_DIR}/models/${base_env}${post_data_strs[0]}${data_dir_sub_str}/${MAIN_DATA_AMT}_eps/${obs_list_str}/${train_id}_/${seed}"
        model_subdir="${base_env}${post_data_strs[0]}${data_dir_sub_str}/${MAIN_DATA_AMT}_eps/${obs_list_str}/${train_id}_/${seed}"
        reset_model_subdir="${base_env}${post_data_strs[1]}${reset_data_dir_sub_str}/${MAIN_DATA_AMT}_eps/${obs_list_str}/${train_id}_/${seed}"

        echo "Running test ${test_id}: ${N_EPISODES} test eps for model at ${model_subdir}, reset model at ${reset_model_subdir} attached."

        python -m contact_il.test_model \
            --model_subdir=${model_subdir} \
            --n_episodes=${N_EPISODES} \
            --device=${DEVICE} \
            --sts_source_vid=${STS_VID} \
            --experiment_id=${test_id} \
            --reset_model_subdir=${reset_model_subdir} \
            --sts_config_dir_override=${STS_CONFIG_DIR_OVERRIDE}

    done
done

for data_amt in "${extra_data_amounts[@]}"; do
    for variant in "${DATA_VARIANT_IDX_COMBOS[@]}"; do
        data_dir_sub_str_i=${variant:0:1}
        data_dir_sub_str=${data_dir_sub_strs[${data_dir_sub_str_i}]}
        reset_data_dir_sub_str=${reset_data_dir_sub_strs[${data_dir_sub_str_i}]}
        obs_list_str_i=${variant:2:3}
        obs_list_str=${obs_list_strs_no_keys[${obs_list_str_i}]}

        for seed in "${SEEDS[@]}"; do
            model_subdir="${base_env}${post_data_strs[0]}${data_dir_sub_str}/${data_amt}_eps/${obs_list_str}/${train_id}_/${seed}"
            reset_model_subdir="${base_env}${post_data_strs[1]}${reset_data_dir_sub_str}/${data_amt}_eps/${obs_list_str}/${train_id}_/${seed}"

            echo "Running test ${test_id}: ${N_EPISODES} test eps for model at ${model_subdir}, rreset model at ${reset_model_subdir} attached."

            python -m contact_il.test_model \
                --model_subdir=${model_subdir} \
                --n_episodes=${N_EPISODES} \
                --device=${DEVICE} \
                --sts_source_vid=${STS_VID} \
                --experiment_id=${test_id} \
                --reset_model_subdir=${reset_model_subdir} \
                --sts_config_dir_override=${STS_CONFIG_DIR_OVERRIDE}

        done

    done
done
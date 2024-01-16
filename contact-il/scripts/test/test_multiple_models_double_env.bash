#!/bin/bash

MODEL_FILE="$1"
RESET_MODEL_FILE="$2"
EXPERIMENT_ID="$3"
N_EPISODES=10
RENDER="false"
DEVICE="cuda"
SIM="false"
# STS_VID="$STS_SIM_VID"
STS_VID=""


if [[ "${SIM}" == "true" ]]; then
    PYTHON_ARGS+=" --sim"
fi

readarray -t models < "${MODEL_FILE}"
readarray -t reset_models < "${RESET_MODEL_FILE}"

i=0
for model in "${models[@]}"; do
    reset_model=${reset_models[i]}
    echo "Starting ${N_EPISODES} test episodes for ${model}, reset ${reset_model}"

python_args=$(cat <<- END
    --model_subdir=${model}
    --n_episodes=${N_EPISODES}
    --device=${DEVICE}
    --sts_source_vid=${STS_VID}
    --experiment_id=${EXPERIMENT_ID}
    --reset_model_subdir=${reset_model}
END
)

    python -m contact_il.test_model ${python_args}

    i=$((i+1))

done

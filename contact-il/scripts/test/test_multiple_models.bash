#!/bin/bash

MODEL_FILE="$1"
EXPERIMENT_ID="$2"
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

for model in "${models[@]}"; do
    echo "Starting ${N_EPISODES} test episodes for ${model}"

python_args=$(cat <<- END
    --model_subdir=${model}
    --n_episodes=${N_EPISODES}
    --device=${DEVICE}
    --sts_source_vid=${STS_VID}
    --experiment_id=${EXPERIMENT_ID}
    --auto_reset
END
)

    python -m contact_il.test_model ${python_args}

done

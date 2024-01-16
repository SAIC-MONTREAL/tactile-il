#!/bin/bash

# DON'T USE THIS! JUST USE PYTHON DIRECTLY

ENVIRONMENT="$1"
REPLAY_DATASET_NAME="$2"
EXPERIMENT_NAME="$3"

if [[ "${ENVIRONMENT}" == "" ]]; then
    echo "ENVIRONMENT must be set. Exiting."
    exit 1
fi

if [[ "${EXPERIMENT_NAME}" == "" ]]; then
    echo "EXPERIMENT_NAME must be set. Exiting."
    exit 1
fi

REPLAY_DATASET_NAME="pid_new_raw_dataset"
REPLAY_DATASET_NAME=""
AUTO_RESET="false"

RENDER="false"

SAVE_DIR="$CIL_DATA_DIR/experiments"


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/test_ft_adapted_replay.py
${ENVIRONMENT}
${REPLAY_DATASET_NAME}
${STS_CONFIG_OLD}
${EXPERIMENT_NAME}
--save_dir=${SAVE_DIR}
--sts_config_dir=${STS_CONFIG_OLD}
END
)


if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi

if [[ "${AUTO_RESET}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --auto_reset"
fi

python ${PYTHON_TO_EXEC}
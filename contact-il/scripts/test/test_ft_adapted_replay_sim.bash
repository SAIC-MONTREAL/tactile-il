#!/bin/bash

EXPERIMENT_NAME="$1"

if [[ "${EXPERIMENT_NAME}" == "" ]]; then
    echo "EXPERIMENT_NAME must be set. Exiting."
    exit 1
fi

ENVIRONMENT="PandaCabinetOneFinger6DOF"    # for sim
STS_VID="../sts_example_vids/wipe-high-force-10fps.mp4"
REPLAY_DATASET_NAME="csv_test"
AUTO_RESET="false"

RENDER="false"


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/test_ft_adapted_replay.py
${ENVIRONMENT}
${REPLAY_DATASET_NAME}
${STS_CONFIG_OLD}
${EXPERIMENT_NAME}
--sts_source_vid=${STS_VID}
--sim
END
)


if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi

if [[ "${AUTO_RESET}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --auto_reset"
fi

python ${PYTHON_TO_EXEC}
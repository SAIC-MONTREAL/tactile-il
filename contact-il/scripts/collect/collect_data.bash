#!/bin/bash

DATASET_NAME="$1"
ENVIRONMENT="PandaCabinetOneFingerNoSTS6DOF"
N_EPISODES=5
COMPRESS="true"
RENDER="false"

SAVE_DIR="/home/t.ablett/datasets/contact-il/demonstrations"


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/collect_kin_teach_demos.py
--environment=${ENVIRONMENT}
--dataset_name=${DATASET_NAME}
--n_episodes=${N_EPISODES}
--save_dir=${SAVE_DIR}
END
)

if [[ "${COMPRESS}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --compress"
fi

if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi


python ${PYTHON_TO_EXEC}
#!/bin/bash

EXP_ID="$1"
N_EPISODES=10
RENDER="false"
# MODEL_SUBDIR="2022-12-16/_10-37-13"
MODEL_SUBDIR="$2"
DEVICE="cuda"
SIM="false"
# STS_VID="$STS_SIM_VID"
STS_VID=""


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/test_model.py
--experiment_id=${EXP_ID}
--model_subdir=${MODEL_SUBDIR}
--n_episodes=${N_EPISODES}
--device=${DEVICE}
--sts_source_vid=${STS_VID}
END
)

if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi

if [[ "${SIM}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --sim"
fi

python ${PYTHON_TO_EXEC}
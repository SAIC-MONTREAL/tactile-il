#!/bin/bash

ENVIRONMENT="$1"
DATASET_NAME="$2"
FT_ADAPTED_REPLAY="$3"

# ENVIRONMENT="PandaTestOneFinger6DOFRealRandomInit"    # real, test pose (not near door)
# ENVIRONMENT="PandaTopWipe"    # wipe door
# ENVIRONMENT="PandaTopGlassOrbOneFinger6DOFRealRandomInit"     # real, near door, random init
# ENVIRONMENT="PandaTopGlassOrbOneFinger6DOFRealNoRandomInit"     # real, near door, no random init
N_EPISODES=10
# N_EPISODES=3
COMPRESS="false"
RENDER="false"

SAVE_DIR="$CIL_DATA_DIR/demonstrations"


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/collect_kin_teach_demos.py
--environment=${ENVIRONMENT}
--dataset_name=${DATASET_NAME}
--n_episodes=${N_EPISODES}
--save_dir=${SAVE_DIR}
--sts_config_dir=${STS_CONFIG}
--sts_source_vid=${STS_VID}
--ft_adapted_replay=${FT_ADAPTED_REPLAY}
END
)

if [[ "${COMPRESS}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --compress"
fi

if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi

python ${PYTHON_TO_EXEC}
#!/bin/bash

DATASET_NAME="$1"

if [[ "${DATASET_NAME}" == "" ]]; then
    echo "DATASET_NAME must be set. Exiting."
    exit 1
fi

ENVIRONMENT="PandaCabinetOneFinger6DOF"    # for sim
# ENVIRONMENT="PandaCabinetOneFinger6DOFROS"    # for sim with ROS
# STS_VID="$STS_SIM_VID"
STS_VID="../../sts_example_vids/sts-drawer-high-force-10fps.mp4"
STS_REPLAY_VID="../../sts_example_vids/sts-drawer-low-force-10fps.mp4"
# REPLAY_DATASET_NAME="new_raw_sim_dataset"
REPLAY_DATASET_NAME=""
AUTO_RESET="false"
# FT_ADAPTED_REPLAY=""
FT_ADAPTED_REPLAY="forward_model"
# FT_ADAPTED_REPLAY="binary_contact"
# FT_ADAPTED_REPLAY="closed_loop_pid"
# FT_ADAPTED_REPLAY="open_loop_delft"
NO_REPLAY_DEMO="false"

N_EPISODES=1
COMPRESS="false"
RENDER="false"

SAVE_DIR="$CIL_DATA_DIR/demonstrations"


PYTHON_TO_EXEC=$(cat <<-END
../../contact_il/collect_kin_teach_demos.py ${ENVIRONMENT} ${DATASET_NAME} \
--n_episodes=${N_EPISODES}
--save_dir=${SAVE_DIR}
--sts_config_dir=${STS_CONFIG_OLD}
--sts_source_vid=${STS_VID}
--sts_replay_source_vid=${STS_REPLAY_VID}
--sim
--replay_dataset_name=${REPLAY_DATASET_NAME}
--ft_adapted_replay=${FT_ADAPTED_REPLAY}
END
)

if [[ "${COMPRESS}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --compress"
fi

if [[ "${RENDER}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --render"
fi

if [[ "${AUTO_RESET}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --auto_reset"
fi

if [[ "${NO_REPLAY_DEMO}" == "true" ]]; then
    PYTHON_TO_EXEC+=" --no_replay_demo"
fi

python ${PYTHON_TO_EXEC}
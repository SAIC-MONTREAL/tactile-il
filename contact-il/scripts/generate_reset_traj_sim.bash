#!/bin/bash


ENVIRONMENT="PandaCabinetOneFinger6DOF"    # for sim
SAVE_DIR="$CIL_DATA_DIR/reset_trajs"


PYTHON_TO_EXEC=$(cat <<-END
../contact_il/env_utils/generate_reset_traj.py
--environment=${ENVIRONMENT}
--save_dir=${SAVE_DIR}
--sim
END
)

python ${PYTHON_TO_EXEC}
# direnv
# eval "$(direnv hook bash)"

# IPs
export ROBOT_IP="173.16.0.3"
export NUC_IP="173.16.0.1"
export GRIPPER_IP="173.16.0.1"

# contact-il
export CIL_DATA_DIR="/mnt/nvme_data_drive/t.ablett/datasets/contact-il"

declare -a cil_repos=(
  "$HOME/projects/contact-panda-envs"
  "$HOME/projects/contact-il"
  "$HOME/projects/panda-polymetis"
  "$HOME/projects/pysts"
  "$HOME/projects/transform_utils"
  "$HOME/projects/place-from-pick-learning"
  "$HOME/panda_ros2_ws/src/sts-cam-ros2"
)

cil-pull-all () {
  for repo in "${cil_repos[@]}"; do
    echo "running git pull $repo"
    git -C "${repo}" pull &
  done
  wait
}

# pysts
export STS_PARENT_DIR="$HOME/panda_ros2_ws/src"
export STS_CONFIG="$HOME/panda_ros2_ws/src/sts-cam-ros2/configs/trevor_usb_sts_rectangular_robot"
export STS_CONFIG_OLD="$HOME/panda_ros2_ws/src/sts-cam-ros2/configs/trevor_usb_sts_rectangular_old"
alias sts-calibrate="python -m pysts.calibrate --config $STS_CONFIG --no_kalman_markers"
alias sts-calibrate-visual="python -m pysts.calibrate --config $STS_CONFIG --mode_dir visual --no_kalman_markers"
alias sts-view="python $HOME/projects/pysts/examples/view.py"

# matplotlib -- for conflicts with cv2 + pyqt
export MPLBACKEND=TkAgg

# polymetis
export PANDA_POLY_CONF="/user/$USER/projects/panda-polymetis/conf"
alias polykill="pkill -9 run_server"
alias polysim="polykill; bash $HOME/projects/panda-polymetis/launch/sim_robot.bash"
alias polysim-nogui="polykill; bash $HOME/projects/panda-polymetis/launch/sim_robot_no_gui.bash"
alias polygriplaunch="launch_gripper.py gripper=franka_hand gripper.cfg.robot_ip=$ROBOT_IP"
alias polykeyboard="python -m panda_polymetis.tools.keyboard_interface --server_ip $NUC_IP --gripper_ip $NUC_IP"
alias polyjointpos="python -m panda_polymetis.tools.get_joint_pos"

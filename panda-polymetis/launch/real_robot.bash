#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
launch_robot.py --config-dir $SCRIPT_DIR/../panda_polymetis/conf/ robot_client=contact-il_real timeout=20

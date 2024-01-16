# sts
This repository encapsulates high level functionality of the STS. 

Useful tools
- calibrate_camera.launch.py will calibrate the camera setting for either tactile or visual modes

Remote Hardware

This is highly dependent on a number of different things to be in place prior to operation.
- screen must be installed on both the local and remote (hardware) nodes
- login on the hardware node must properly source the sts-cam-ros2 install setup.bash file
- the sts launch file local_bringup.launch.py should work properly
- the values in the remote_sts_bringup.launch.py files must be configured for your hardware (especially remote_user and remote_config_dir). 
- Then remote_sts_bringup.launch.py should bring up the remote hardware. When you kill this launch process you will also bring down the remote nodes

Version History

- V1.1 Support for remote hardware
- V1.0 First porting to the ros2 environment

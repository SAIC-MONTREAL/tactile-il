# contact-panda-envs
A repository for robotic enviroments to be used with the Panda and the STS sensor.

This is both a ROS package and a pip installable python package. The ROS package allows for communication with ROS facilities, while the python package makes it convenient to install and use as part of other scripts (e.g. data collection or testing).

To use, first place in a `catkin_ws/src` folder, build with `catkin_make`, and then install it with `pip` in your python environment (which can be virtual).
Note that this will only work if all of the other dependencies listed in the _Installaion_ section below have also been installed.


# 1 Installation

**TODO add more here**
1. Install polymetis.
2. Install panda-polymetis.
3. Install sts.
4. Install pysts.

You're done!


<del>

## 1.1 ROS dependencies (DEPRECATED)
To run the environments defined here, you will need to have Ubuntu 20.04 running (either as a main OS or in docker). 
You will also need to have both ros1 (noetic) and ros2 (foxy) installed.

## 1.2 Workspace Structure
You should have three separate ros workspaces, for ros1 (`panda_ws` in this example), ros2 (`panda_ros2_ws` in this example), and ros bridge (`bridge_ws` in this example).
We'll assume the setup is (after cloning all dependenices mentioned in the lower sections):
```
$HOME/
└──panda_ws/
    └──src/
        ├──contact-panda-envs/  (this package)
        ├──panda-bm_ros_packages/
        └──sts-cam-ros1-msgs/
└──panda_ros2_ws/
    └──src/
        ├──panda-bm_ros2_packages/
        └──sts-cam-ros2/
└──bridge_ws/
    └──src/
        └──ros1_bridge/
```

## 1.3 ROS 1 Dependencies

### 1.3.1 [panda-bm_ros_packages](https://github.sec.samsung.net/o-limoyo/panda-bm_ros_packages)
To run things on the panda robots, we use the clients that have been defined in this package.
This package also uses the `with_latest_franka` branch of the package.
Clone the package into your ros1 workspace (possibly as defined in structure above), and also clone the submodules:
```bash
cd $HOME/panda_ws/src
git clone --recursive -b with_latest_franka git@github.sec.samsung.net:o-limoyo/panda-bm_ros_packages.git
```
You'll also need to install the requirements files via pip (possibly in your virtual python env):
```bash
cd $HOME/panda_ws/src/panda-bm_ros_packages
pip install -r requirements.txt
pip install -r sts_direct_requirements.txt
```

### 1.3.2 [sts-cam-ros1-msgs](https://github.sec.samsung.net/o-limoyo/panda-bm_ros_packages)
To translate messages and services from the `sts-cam-ros2` package to ros1, we have to use a corresponding ros1 version of the package.
Clone the package into your ros1 workspace (possibly as defined in structure above):
```bash
cd $HOME/panda_ws/src
git clone git@github.sec.samsung.net:SAIC-Montreal/sts-cam-ros1-msgs.git
```

### 1.3.3 Building
To build, use `catkin_make`:
```bash
cd $HOME/panda_ws
catkin_make
```

## 1.4 ROS 2 Dependencies

### 1.4.1 [sts-cam-ros2](https://github.sec.samsung.net/SAIC-Montreal/sts-cam-ros2) 
In addition to the sts client defined in `panda-bm_ros_packages`, we also use the `sts-cam-ros2` package, currently with the `rpi_camera` branch.
Clone the package into your ros2 workspace:
```bash
cd $HOME/panda_ros2_ws/src
git clone -b rpi_camera git@github.sec.samsung.net:SAIC-Montreal/sts-cam-ros2.git
```

Note that you _also_ need to have this branch of `sts-cam-ros2` set up on the device actually running the camera (if it is on a separate device, such as a raspberry pi).

### 1.4.2 [panda-bm_ros2_packages](https://github.sec.samsung.net/o-limoyo/panda-bm_ros2_packages)
For using the realsense and related packages, we use `panda-bm_ros2_packages`.

**Note**: this functionality is not yet used.

Clone the package into your ros2 workspace:
```bash
cd $HOME/panda_ros2_ws/src
git clone git@github.sec.samsung.net:o-limoyo/panda-bm_ros2_packages.git
```

### 1.4.3 Building
To build, use `colcon build`:
```bash
cd $HOME/panda_ros2_ws
colcon build --symlink-install
```

## 1.5 ROS Bridge Dependencies

### 1.5.1 [ros1_bridge](https://github.com/ros2/ros1_bridge)
Since the panda drivers are on ros1 and everything else we use is on ros2, we need to bridge ros1 and ros2 messages and services.
We do this using the `ros1_bridge` package.
To build this with support for the custom messages and services that we use, you need to follow a very specific instruction set.
First, clone the package:
```bash
cd $HOME/bridge_ws/src
git clone https://github.com/ros2/ros1_bridge.git
```
### 1.5.2 Building
we recommend following the instructions in the README for [sts-cam-ros1-msgs](https://github.sec.samsung.net/SAIC-Montreal/sts-cam-ros1-msgs) to actually get the bridge built properly.

# 2 Running
To run environments using this package, once you have followed all of the instructions defined above and pip installed `contact-panda-envs` (potentially in a virtual environment), you can follow these steps.
Each one of these commands should be run from separate terminals, and each terminal must be kept open, unless otherwise noted.

## 2.1 STS Setup
It is recommneded to set up an environment variable (in our case `STS_CONFIG`) that contains the directory with all of the sts config that you will be using.

### 2.1.1 Run STS stream (possibly from raspberry pi)
```bash
source $HOME/panda_ros2_ws/install/setup.bash
ros2 launch sts sts_bringup.launch.py config_dir:=$STS_CONFIG camera_resolution:=320x240 adjusted_resolution:=160x120
```

### 2.1.2 Run STS mode node
```bash
source $HOME/panda_ros2_ws/install/setup.bash
ros2 launch sts sts_mode_node.launch.py config_dir:=$STS_CONFIG
```

### 2.1.3 Calibrate Tactile and Visual Modes of STS
The calibration nodes should be run individually and can be shut down once calibration is complete.
#### 2.1.3.1 Tactile
```bash
source $HOME/panda_ros2_ws/install/setup.bash
ros2 launch sts calibrate_camera.launch.py config_dir:=$STS_CONFIG
```

#### 2.1.3.2 Visual
```bash
source $HOME/panda_ros2_ws/install/setup.bash
ros2 launch sts calibrate_camera.launch.py config_dir:=$STS_CONFIG mode:=visual
```

## 2.2 Panda Setup
Follow the instructions in [panda-bm_ros_packages](https://github.sec.samsung.net/o-limoyo/panda-bm_ros_packages) to bring up a panda arm in (gazebo) simulation or in the real world.

Note that both the regular panda bringup and panda moveit must both be running.

## 2.3 Bridge Setup
1. Source the ros1 environment variables, then bring up a roscore:
```bash
source /opt/ros/noetic/setup.bash
source $HOME/panda_ws/devel/setup.bash
roscore
```

2. In another terminal, start the ros bridge.
You can use the script that's defined in this package, but you will have to make sure to correctly define the aliases used in this bash script yourself for your own machine.
An example of the relevant aliases, along with many other potentially useful ones, can be found in the `shell_env_scripts/example_aliases` file in this package.
```bash
bash $HOME/panda_ws/src/contact-panda-envs/shell_env_scripts/start_bridge_all.sh
```

## 2.4 Environment
1. Source both the ros1 workspace and, potentially, your virtual python environment:
```bash
source /opt/ros/noetic/setup.bash
source $HOME/panda_ws/devel/setup.bash
source activate {optinal_virtual_env}
```
2. To verify that the environment will run properly, we can check to see if some of the relevant topics are being published:
```bash
rostopic list | grep sts
$ /sts/image
$ /sts/image_raw
```

```bash
rostopic list | grep joint_states
$ /franka_gripper/joint_states
$ /franka_state_controller/joint_states
$ /franka_state_controller/joint_states_desired
$ /joint_states
```

3. Open and use the environment as you would any other gym environment:
```python
from contact_panda_envs.envs.cabinet.cabinet_contact import PandaCabinetOneFingerNoSTS2DOF

env = PandaCabinetOneFingerNoSTS2DOF()
obs = env.reset()
done = False

while not done:
    next_obs, rew, done, info = env.step([0, 0])
```
   
</del>

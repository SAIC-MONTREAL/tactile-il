# contact-il
Imitation Learning with STS sensors.
The main functionalities that we're looking to demonstrate in this repository/with this paper are:
1. The use of STS data in end-to-end imitation learning for manipulation tasks
2. The use of an automatic switching algorithm to allow switching between visual and tactile modes (to benefit 1 above).
3. The use of force estimates to fix a crucial issue introduced by using kinesthetic teaching in contact-rich tasks with a pose-based impedance controller.

# Installation Instructions / Dependencies
Tested on Ubuntu 22.04 with conda + Python 3.8.
The conda + python version requirements are a result of using polymetis.
Recommended to install in the order shown here.

1. [polymetis](https://facebookresearch.github.io/fairo/polymetis/index.html)
    - Follow the [installation instructions](https://facebookresearch.github.io/fairo/polymetis/installation.html).
    For the time being, we're not installing fom source.
    - **Make sure that you install mamba!! (as suggested at the top of the installation page)**

For the rest of the dependencies, it is assumed that you are in your virtual env (`conda activate polymetis`).

2. Update a broken c library dependency:
  ```bash
  conda install -c conda-forge libstdcxx-ng=12
  ```

3. torchvision: needs to match this torch version
    - if torch from polymetis is 1.10.0, then torchvision: 0.11.1
    - if torch from polymetis is 1.13.1, then torchvision: (main/latest)
  ```bash
  pip install torchvision
  ```

4.  [panda-polymetis](https://github.sec.samsung.net/t-ablett/panda-polymetis) --> replaces panda-bm_ros_packages
  ```bash
  git clone git@github.sec.samsung.net:t-ablett/panda-polymetis.git
  pip install -e panda-polymetis
  ```

5. [contact-panda-envs](https://github.sec.samsung.net/t-ablett/contact-panda-envs) --> for gym envs
  ```bash
  git clone git@github.sec.samsung.net:t-ablett/contact-panda-envs.git
  pip install -e contact-panda-envs
  ```

6. [sts](https://github.sec.samsung.net/SAIC-Montreal/sts-cam-ros2)
    - Specifically, only the sts python package within this repo.
    - You do **not** need to have ROS installed!
  ```bash
  git clone -b kf_fixing git@github.sec.samsung.net:SAIC-Montreal/sts-cam-ros2.git
  pip install -e sts-cam-ros2/src/sts
  ```

7. [pysts](https://github.sec.samsung.net/t-ablett/pysts) --> replaces sensor interfacing from sts-cam-ros2
  ```bash
  git clone git@github.sec.samsung.net:t-ablett/pysts.git
  pip install -e pysts
  ```

8. [transform_utils](https://github.sec.samsung.net/f-hogan/transform_utils/tree/feature-rotation-representation) --> used for SE3 and SO3 transforms
  ```bash
  git clone -b feature-rotation-representation git@github.sec.samsung.net:f-hogan/transform_utils.git && cd transform_utils
  bash install.sh  # needed for some dependencies, instead of regular pip install
  ```

9. [place_from_pick_learning](https://github.sec.samsung.net/o-limoyo/place-from-pick-learning) --> used for learning
  ```bash
  git clone -b contact-il-feature-adds git@github.sec.samsung.net:o-limoyo/place-from-pick-learning.git
  pip install -e place-from-pick-learning
  ```

10. contact-il (this package).
  ```bash
  git clone git@github.sec.samsung.net:t-ablett/contact-il.git
  pip install -e contact-il
  ```

11. [realsense_wrapper](https://github.com/facebookresearch/fairo/tree/main/perception/realsense_driver)
    - you can clone fairo and install with `cd fairo/perception/realsense_driver; pip install .`, but fairo is pretty big.
    - alternatively, install like this:
  ```bash
  svn checkout https://github.com/facebookresearch/fairo/trunk/perception/realsense_driver
  pip install realsense_driver
  ```

12.  [inputs](https://github.com/trevorablett/inputs) --> used for non-blocking keyboard input reading
    - installed automatically by contact-il


13. Run a sim test: `python -c "import polysim"`.
  - If you see an error about numba/numpy versions, run the following:
      ```bash
      pip uninstall numba
      pip install --upgrade llvmlite --ignore-installed
      pip install --upgrade numba
      ```

# Environment Variables
Make sure to add something similar to the following to your `.bashrc` (with corrected locations for your specific install):
```bash
export NUC_IP="172.16.0.1"
export STS_CONFIG="$HOME/panda_ros2_ws/src/sts-cam-ros2/configs/trevor_usb_sts_rectangular_robot"
export STS_SIM_VID="$HOME/projects/pysts/examples/recordings/sts-visual-02-09-23-12_53_53.mp4"
```

# Bringup in Sim
Currently, only the arm is simulated.
If and when polymetis adds the ability to simulate the gripper in a robust way, I'll update this section.

For now, to bring up a simulated robot with a GUI, run:
```bash
pkill -9 run_server; bash /path/to/panda-polymetis/launch/sim_robot.bash
```

You can also bring one up without a GUI using:
```bash
pkill -9 run_server; bash /path/to/panda-polymetis/launch/sim_robot_no_gui.bash
```

The panda-polymetis package has a `FakePandaGripperClient` class to mimic the data that would be received from a real gripper for fully testing the environment.
This is automatically used when you use various `sim` options with the collection/testing scripts.

# Bringup on Robot
There are notes about the robots and networking info (e.g. ip addresses) here:
[panda_notes Networking and Credentials](https://github.sec.samsung.net/SAIC-Montreal/panda_notes/blob/master/networking_and_credentials.md).

1. Turn on bottom panda computer, wait for it to boot.
2. On the thinkstation, run panda desk in browser (`https://172.16.0.2/desk`), activate FCI.
3. ssh into the Panda 2 NUC:
    - run `source activate polymetis`
    - run `polylaunch` (an alias for `pkill -9 run_server; bash $HOME/projects/panda-polymetis/launch/real_robot.bash`).
4. On thinkstation (_not_ on the NUC!), run the gripper server:
    - `launch_gripper.py gripper=franka_hand`

# Testing Robot Functionality with Keyboard Interface
In both sim and the real robot, a useful tool is the keyboard interface.
This keyboard interface can also be used to quickly enable/disable freedrive when not running other modes.

To launch,
```bash
python -m panda_polymetis.tools.keyboard_interface
```

On the real robot, make sure to launch with `--server_ip $NUC_IP`.

# Collection/Training/Testing
All of these assume you have already run `cd /path/to/contact-il/scripts`.

As well, it is assumed that you have set a bash environment variable `CIL_DATA_DIR`, which will then have the subdirectories `demonstrations`, `models`, and `tests`, for data collection, training, and running models respectively.

The training code is currently set up to use hydra, while the dataset and testing code both use argparse + an optional bash script.
Arguably, all three could/should use hydra, but this is low priority for now.
One reason for not making the shift is because the options for data collection and testing should stay mostly fixed (apart from dataset/model/experiment names), while training will frequently have many changes for ablations.

## Data Collection
```bash
bash collect_sts_data.bash example_dataset_name
```
Can also see options/run with
```
python python -m contact_il.collect_kin_teach_demos -h
```

## Training
First, in `cfgs/cfg.yaml`, change `data_dir_name` to match the `example_dataset_name` from the data collection.
```bash
python -m contact_il.train_bc id=example_model_name
```
The options you can change here are determined by hydra; for example, `id` is in the `cfg.yaml` file mentioned before, and defaults to an empty string.
You can also see the options with `python -m contact_il.train_bc -h`.
By default, models are placed in a date folder (`YYYY-MM-DD`), and further placed in subfolders defined by id + time `id=example_model_name_HH-MM-SS`.
If other hydra parameters are modified on the command line, they are also added to the directory name.

## Testing Models
```bash
bash test_model.bash example_experiment_id model_subfolder
```
Can also see options/run with
```
python python -m contact_il.test_model -h
```

# Issues
If you get `AttributeError: module 'distutils' has no attribute 'version'`, run:
```
pip install setuptools==59.5.0
```

<del>

# Running things on Robot (DEPRECATED ROS METHOD)
Note that on the main computer we have aliases of `source-r1-all` and `source-r2-all` for sourcing noetic+ros1 ws and foxy + ros2 ws + bridge ws, respectively.

Need to have each of the following running:

1. panda desk needs to be running in browser, and activate FCI
2. `source-r1-all`, then run roscore on main computer.
3. ssh into robot nuc:
    - source '$HOME/trevor_ws/devel/setup.bash`
    - run `roslaunch control panda_control.launch robot_ip:=$ROBOT load_gripper:=true`, aliased as `launch_robot`.
4. _(optional)_ `source-r1-all`, then run `rosrun control keyboard_interface.py` to run freedrive, gripper, error handling, etc.

## STS things
1. `source-r2-all`, then run `launch_cam_res_mod 320x240 160x120` which is the function
```
ros2 launch sts sts_bringup.launch.py camera_resolution:=$1 config_dir:=${STS_CONFIG} adjusted_resolution:=$2
```
2. `source-r2-all`, then run `sts-mode-node`, which is an alias for `ros2 launch sts sts_mode_node.launch.py config_dir:=$STS_CONFIG`.

## Env things
1. `source-r1-all`, then run collection/control code.


# Note on dependencies
`place_from_pick_learning` uses PoseTransformer (optinally), and because PoseTransformer uses `geometry_msgs` and `std_msgs` from ROS, you either need to have ROS1 installed, or you can manually install both of these packages using the following (there's probably a cleaner way to do this via requirements.txt or setup.py)

```
pip install --extra-index-url https://rospypi.github.io/simple/_pre geometry-msgs
pip install --extra-index-url https://rospypi.github.io/simple/_pre std-msgs
```

</del>

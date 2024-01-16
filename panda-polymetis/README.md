# panda-polymetis
Tools for working with the Pandas using [Polymetis](https://facebookresearch.github.io/fairo/polymetis/) from FAIR. Meant to be a (mostly) drop-in replacement for panda-bm_ros_packages.

# Requirements
To use this package, you need to [install polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html).
Follow the conda-based instructions there to create a new conda env with polymetis installed.
You can also install from source, but this is a bit trickier (and thus far, nothing in this package requires it).

## Building polymetis from source
The instructions on the above page for building polymetis from source mostly work, but with one caveat in particular: if you're building on a system with cuda enabled, you will also have to install the cuda relevant cuda toolkit stuff.
Specifically, [uncomment out this line](https://github.com/facebookresearch/fairo/blob/0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2/polymetis/polymetis/environment.yml#L12), run `conda env update polymetis/environment.yml` and you may also need to run `conda install cuda-nvcc=11.3` or `conda install -c nvidia cuda-nvcc=11.3`.
I also had to update numpy manually (`pip install numpy==1.23.5`, the version from the environment.yml file above), which I did with pip instead of conda because conda wanted to change 5000 things.

# Usage

## Bringing up the robot with polymetis or polysim
The `launch_robot.py` and `launch_gripper.py` scripts are available from anywhere once you've run `conda activate polymetis`/`source activate polymetis`.

### Sim
*Note:* For now, simulating the gripper is not supported.
Polymetis sort of supports it, but it's experimental, and doesn't work all that well yet.

In one terminal, run
```
launch_robot.py robot_client=franka_sim use_real_time=false gui=true
```

### Real Robot
The robot client and gripper client are launched separately.

#### Arm
On the Nuc, for controlling the robot (assuming ROBOT_IP is set):
```
launch_robot.py robot_client=franka_hardware timeout=20 robot_client.executable_cfg.robot_ip=$ROBOT_IP
```

There are other command line arguments you can set using hydra.
See the [config file](https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/conf/robot_client/franka_hardware.yaml).

#### Gripper
On the host machine (_not_ the NUC, since the NUC should have as little running on it as possible),
```
launch_gripper.py gripper=franka_hand robot_client.executable_cfg.robot_ip=$ROBOT_IP
```

## Running our clients
Use the tests/utils to get an idea.

# States
Fields of states that are available are in [this proto file](https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/protos/polymetis.proto).
This is probably what we'll have to modify to expose things that aren't yet exposed (e.g. F_ext).

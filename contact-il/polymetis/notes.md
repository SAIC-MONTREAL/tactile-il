# Sim

To launch in sim:
```
launch_robot.py robot_client=franka_sim use_real_time=false gui=true
```

To get the pybullet gui to work, also needed to fix the C++ libraries that were present in anaconda.
Specifically, see this answer on ask ubuntu:

```
https://askubuntu.com/a/764572
```


# Updating the package
The conda package that is installed by default is missing some of the latest additions.

You need to follow the instructions from installing from source from the website, and possibly install some missing dependencies.
Specifically, a bunch of boost libraries may need to be installed, and also a few variables needed to have `std::` added in front of `size_t`.

# Updating the package -- manually, file by file...don't do this unless you really need to. Above method is much cleaner.
Had to copy the polymetis folder from the python subfolder over (check the files to confirm which one, and copy it into envs/polymetis/lib/python3.8/site-packages)

Then, also had to copy the launch_robot.py file p ~/projects/fairo/polymetis/polymetis/python/scripts/launch_robot.py.
# place-from-pick-learning


## Note on dependencies
`place_from_pick_learning` uses PoseTransformer (optinally), and because PoseTransformer uses `geometry_msgs` and `std_msgs` from ROS, you either need to have ROS1 installed, or you can manually install both of these packages using the following (there's probably a cleaner way to do this via requirements.txt or setup.py)

```
pip install --extra-index-url https://rospypi.github.io/simple/_pre geometry-msgs
pip install --extra-index-url https://rospypi.github.io/simple/_pre std-msgs
```
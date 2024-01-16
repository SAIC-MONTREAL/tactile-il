#!/bin/bash

shopt -s expand_aliases
source ~/.project_aliases

# first source ros 1 stuff and then built ros 1 workspace
source-ros1
source-pandaws

# then source ros 2 stuff and built ros 2 workspace, including built ros1_bridge
source-ros2
source-pandar2ws
source-bridgews

# now you can run the bridge -- bridge all if you want list/echo to work
ros2 run ros1_bridge dynamic_bridge
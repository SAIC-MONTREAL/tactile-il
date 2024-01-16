cmake_minimum_required(VERSION 3.4)
project(contact_panda_envs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

## Generate messages in the 'msg' folder
#  add_message_files(
#    FILES
#    KeyboardTrigger.msg
#    SModel_robot_input.msg
#    SModel_robot_output.msg
#    Marker.msg
#  )

## Generate services in the 'srv' folder
#  add_service_files(
#    FILES
#  )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#     DEPENDENCIES
#     std_msgs
#     sensor_msgs
# )

catkin_package(
   CATKIN_DEPENDS message_runtime
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
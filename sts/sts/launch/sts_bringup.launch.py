import os

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import json
import os

def generate_launch_description():
    # get working config directory
    config_dict = {}
    with open(os.environ["STS_PARENT_DIR"] + f"/sts-cam-ros2/configs/working_config_dir.json", "r") as fd:
        config_dict = json.load(fd)

    return LaunchDescription([
        DeclareLaunchArgument('camera_resolution', default_value = '640x480', description = 'Powerup camera resolution'),
        DeclareLaunchArgument('sts_name', default_value = 'sts', description = 'Namespace of this STS'),
        DeclareLaunchArgument('config_dir', default_value = config_dict['config_dir'], description = 'Configuration file directory'),
        DeclareLaunchArgument('pattern', default_value = 'white', description = 'LED light configuration'),

        Node(
            package = 'sts_tools',
            namespace = LaunchConfiguration('sts_name'),
            executable = 'camera_stream',
            name = 'camera_stream',
            parameters = [
                {'camera_resolution' : LaunchConfiguration('camera_resolution')},
                {'config_dir' : LaunchConfiguration('config_dir')}
            ],
        ),
        Node(
            package = 'sts_tools',
            namespace = LaunchConfiguration('sts_name'),
            executable = 'camera_parameters',
            name = 'camera_parameters',
            parameters = [
                {'config_dir' : LaunchConfiguration('config_dir')}
            ],
       ),
       Node(
            package='sts_tools',
            namespace = LaunchConfiguration('sts_name'),
            executable='led_node',
            name='led_node',
            parameters = [
                {'config_dir' : LaunchConfiguration('config_dir')}
            ],
        ),
        Node(
            package='sts_tools',
            namespace = LaunchConfiguration('sts_name'),
            executable='led_static',
            name='led_static',
            parameters=[
                {'config_dir':LaunchConfiguration('config_dir')},
                {'pattern':LaunchConfiguration('pattern')},
            ],
        )
    ])

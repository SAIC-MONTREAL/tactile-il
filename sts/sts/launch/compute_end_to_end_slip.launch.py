import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
import json
import os

def generate_launch_description():
    # get working config directory
    config_dict = {}
    with open(os.environ["STS_PARENT_DIR"] + f"/sts-cam-ros2/configs/working_config_dir.json", "r") as fd:
        config_dict = json.load(fd)

    return LaunchDescription([
        DeclareLaunchArgument('sts_name', default_value = 'sts', description = 'Namespace of this STS'),
        DeclareLaunchArgument('config_dir', default_value = config_dict['config_dir'], description='STS configuration directory.'),
        DeclareLaunchArgument('algorithm', default_value = 'hue', description='Localization algorithm [hue, uenet].'),
        DeclareLaunchArgument('display', default_value = "True", description = 'Display eye candy.'),

        Node(
            package = 'sts',
            namespace = LaunchConfiguration('sts_name'),
            executable = 'end_to_end_slip_detector',
            name = 'end_to_end_slip_detector',
            parameters = [
                {'config_dir' : LaunchConfiguration('config_dir')},
                {'display' : LaunchConfiguration('display')},
            ],
            remappings=[
                ('flow', 'tactile_flow'),
                ('flow_image', 'tactile_flow_image'),
            ],
        ),
    ])

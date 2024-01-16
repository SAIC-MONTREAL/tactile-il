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
        DeclareLaunchArgument('config_dir', default_value = config_dict['config_dir'], description = 'STS configuration directory.'),
        DeclareLaunchArgument('display', default_value = "True", description = 'Display eye candy.'),
        DeclareLaunchArgument('image_topic', default_value = "image", description = 'sts topic'),
        DeclareLaunchArgument('contact_image_topic', default_value = "contact_image", description = 'sts topic'),

        Node(
            package = 'sts',
            namespace = LaunchConfiguration('sts_name'),
            executable = 'contact_detection',
            name = 'contact_detection',
            parameters = [
                {'config_dir' : LaunchConfiguration('config_dir')},
                {'display': LaunchConfiguration('display')},
                {'image_topic': LaunchConfiguration('image_topic')},
                {'contact_image_topic': LaunchConfiguration('contact_image_topic')},
            ],
        ),
    ])

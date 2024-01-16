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
        DeclareLaunchArgument('algorithm', default_value = "hsv", description = 'Marker detection algorithm [hue, unet, hsv].'),
        DeclareLaunchArgument('marker_image_topic', default_value = "marker_image", description = 'marker image topic'),
        DeclareLaunchArgument('marker_topic', default_value = "marker", description = 'marker topic'),
        DeclareLaunchArgument('camera_topic', default_value = "image", description = 'sts topic'),

         Node(
            package = 'sts',
            namespace = LaunchConfiguration('sts_name'),
            executable = 'marker_detector',
            name = 'marker_detector',
            parameters = [
                {'config_dir' : LaunchConfiguration('config_dir')},
                {'algorithm' : LaunchConfiguration('algorithm')},
                {'display' : LaunchConfiguration('display')},
                {'marker_image_topic' : LaunchConfiguration('marker_image_topic')},
                {'marker_topic' : LaunchConfiguration('marker_topic')},
                {'camera_topic' : LaunchConfiguration('camera_topic')},
            ],
        ),

    ])

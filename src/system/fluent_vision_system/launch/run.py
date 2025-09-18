from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os


def _setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('fluent_vision_system')
    ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'fluent_vision_system.launch.py')
        ),
        launch_arguments={
            'config': LaunchConfiguration('config').perform(context),
            'update_camera_serials': LaunchConfiguration('update_camera_serials').perform(context),
            'camera_serials_path': LaunchConfiguration('camera_serials_path').perform(context),
            'use_serials_script': LaunchConfiguration('use_serials_script').perform(context),
            'serials_script_path': LaunchConfiguration('serials_script_path').perform(context),
        }.items(),
    )
    return [ld]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='/config/fluent_vision_system.yaml',
            description='Absolute path to fluent_vision_system yaml',
        ),
        DeclareLaunchArgument(
            'update_camera_serials',
            default_value='false',
            description='If true, detect RealSense models and write camera serials file before launching',
        ),
        DeclareLaunchArgument(
            'camera_serials_path',
            default_value='/config/camera_serials.yaml',
            description='Path to write/read camera_serials.yaml',
        ),
        DeclareLaunchArgument(
            'use_serials_script',
            default_value='false',
            description='Use external script to detect serials (preferred if available)',
        ),
        DeclareLaunchArgument(
            'serials_script_path',
            default_value='',
            description='Path to update_camera_serials.sh (or compatible)',
        ),
        OpaqueFunction(function=_setup),
    ])

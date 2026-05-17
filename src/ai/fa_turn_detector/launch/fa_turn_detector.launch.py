from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")

    return LaunchDescription([
        DeclareLaunchArgument(
            "node_name",
            default_value="fa_turn_detector",
            description="ノード名",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("fa_turn_detector"), "config", "default.yaml"]
            ),
            description="設定ファイルへのパス",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="",
            description="Smart Turn ONNX model path",
        ),
        Node(
            package="fa_turn_detector",
            executable="fa_turn_detector_node",
            name=node_name,
            output="screen",
            parameters=[config_file, {"backend.model_path": model_path}],
        ),
    ])

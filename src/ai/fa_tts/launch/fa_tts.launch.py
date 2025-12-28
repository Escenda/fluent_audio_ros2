from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fv_tts',
            executable='fv_tts_node',
            name='fv_tts',
            output='screen',
            parameters=[{
                'default_voice': '',
                'output_topic': 'audio/tts/frame',
                'playback_topic': 'audio/output/frame',
                'use_playback_topic': True,
                'cache_dir': '~/.fluent_voice_cache',
            }],
        )
    ])

"""Launch the image processor and the MSCKF VIO filter on FLA data.

The FLA dataset ships a fisheye (equidistant) stereo pair, so the calibration
in config/camchain-imucam-fla.yaml is used along with its own noise settings.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('msckf_vio'), 'config',
        'msckf_vio_fla_params.yaml'])

    imu_topic = LaunchConfiguration('imu_topic')
    cam0_topic = LaunchConfiguration('cam0_topic')
    cam1_topic = LaunchConfiguration('cam1_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'imu_topic', default_value='/sync/imu/imu',
            description='IMU topic for both nodes.'),
        DeclareLaunchArgument(
            'cam0_topic', default_value='/sync/cam0/image_raw',
            description='Left camera image topic.'),
        DeclareLaunchArgument(
            'cam1_topic', default_value='/sync/cam1/image_raw',
            description='Right camera image topic.'),

        Node(
            package='msckf_vio',
            executable='image_processor_node',
            name='image_processor',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('imu', imu_topic),
                ('cam0_image', cam0_topic),
                ('cam1_image', cam1_topic),
                ('features', '/image_processor/features'),
                ('tracking_info', '/image_processor/tracking_info'),
                ('debug_stereo_image', '/image_processor/debug_stereo_image'),
            ],
        ),

        Node(
            package='msckf_vio',
            executable='msckf_vio_node',
            name='vio',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('imu', imu_topic),
                ('features', '/image_processor/features'),
            ],
        ),
    ])

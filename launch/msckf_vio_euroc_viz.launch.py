"""EuRoC run with rviz2 started alongside the estimator.

Same as msckf_vio_euroc.launch.py, plus rviz2 with the EuRoC display config, so
the estimate can be watched while the bag is played. Set `rviz:=false` to bring
up the nodes alone.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('msckf_vio')

    nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch',
                                  'msckf_vio_euroc.launch.py'])),
        launch_arguments={
            'params_file': LaunchConfiguration('params_file'),
            'ground_truth': LaunchConfiguration('ground_truth'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'cam0_topic': LaunchConfiguration('cam0_topic'),
            'cam1_topic': LaunchConfiguration('cam1_topic'),
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution(
            [pkg_share, 'rviz', 'rviz_euroc_config.rviz'])],
        parameters=[{'use_sim_time': ParameterValue(
            LaunchConfiguration('use_sim_time'), value_type=bool)}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution([
            pkg_share, 'config', 'msckf_vio_euroc_params.yaml'])),
        DeclareLaunchArgument('ground_truth', default_value='',
                              description='Optional EuRoC ground-truth CSV or TUM file.'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'imu_topic', default_value='/imu0',
            description='IMU topic for both nodes.'),
        DeclareLaunchArgument(
            'cam0_topic', default_value='/cam0/image_raw',
            description='Left camera image topic.'),
        DeclareLaunchArgument(
            'cam1_topic', default_value='/cam1/image_raw',
            description='Right camera image topic.'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Also start rviz2 with the trajectory display.'),

        nodes,
        rviz,
    ])

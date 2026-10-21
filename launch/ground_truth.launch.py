"""Optional timestamp-matched ground-truth replay and trajectory comparison."""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_ground_truth(context):
    filename = LaunchConfiguration('ground_truth').perform(context).strip()
    if not filename:
        return []
    filename = Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise ValueError(f'Ground-truth file does not exist: {filename}')
    calibration = LaunchConfiguration('vio_params_file').perform(context)
    with open(calibration, encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['vio']['ros__parameters']
    return [Node(
        package='msckf_vio', executable='ground_truth_node.py',
        name='ground_truth', output='screen',
        parameters=[{
            'ground_truth': str(filename),
            'T_imu_body': parameters['T_imu_body'],
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
        remappings=[('odom', LaunchConfiguration('odom_topic')),
                    ('path', LaunchConfiguration('path_topic'))],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ground_truth', default_value='',
                              description='Optional EuRoC state ground-truth CSV or TUM file.'),
        DeclareLaunchArgument('vio_params_file', default_value=PathJoinSubstitution([
            FindPackageShare('msckf_vio'), 'config', 'msckf_vio_euroc_params.yaml'])),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('path_topic', default_value='/path'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=launch_ground_truth),
    ])

"""Call the VIO reset service.

This is not an anytime reset: the sensor suite must be standing still, since
the IMU bias and the initial orientation are re-estimated from the next 200
IMU samples. Equivalent to:

  ros2 service call /reset std_srvs/srv/Trigger
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    service_name = LaunchConfiguration('service_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'service_name', default_value='/reset',
            description='Name of the VIO reset service.'),

        ExecuteProcess(
            cmd=['ros2', 'service', 'call', service_name,
                 'std_srvs/srv/Trigger', '{}'],
            output='screen',
        ),
    ])

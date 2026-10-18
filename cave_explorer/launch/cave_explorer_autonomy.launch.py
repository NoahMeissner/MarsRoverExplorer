
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ld = LaunchDescription()
    config_path = [FindPackageShare('cave_explorer'), 'config']

    # Additional command line arguments
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )
    print_feedback_launch_arg = DeclareLaunchArgument(
        'print_feedback',
        default_value='False',
        description='Flag to enable print feedback from action server'
    )
    perception_args = [
        DeclareLaunchArgument('model', default_value='best.pt',
                              description='Detector weights in the config folder'),
        DeclareLaunchArgument('degrade', default_value='False',
                              description='Degrade the camera images before detection'),
        DeclareLaunchArgument('degrade_type', default_value='both',
                              description='dust, dark or both'),
        DeclareLaunchArgument('degrade_severity', default_value='1.0',
                              description='Maximum severity 0..1'),
        DeclareLaunchArgument('degrade_ramp_time', default_value='300.0',
                              description='Seconds until the maximum severity, 0 = constant'),
        DeclareLaunchArgument('geometry', default_value='True',
                              description='Start the cave geometry analysis node'),
    ]

    # Start Navigation Stack
    cave_explorer_node = Node(
        package='cave_explorer',
        executable='cave_explorer',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'print_feedback': LaunchConfiguration('print_feedback'),
                     'computer_vision_model_filename': PathJoinSubstitution(
                         config_path + [LaunchConfiguration('model')]),
                     'degrade': LaunchConfiguration('degrade'),
                     'degrade_type': LaunchConfiguration('degrade_type'),
                     'degrade_severity': LaunchConfiguration('degrade_severity'),
                     'degrade_ramp_time': LaunchConfiguration('degrade_ramp_time')}]
    )

    cave_geometry_node = Node(
        package='cave_explorer',
        executable='cave_geometry',
        output='screen',
        condition=IfCondition(LaunchConfiguration('geometry')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(print_feedback_launch_arg)
    for arg in perception_args:
        ld.add_action(arg)
    ld.add_action(cave_explorer_node)
    ld.add_action(cave_geometry_node)

    return ld

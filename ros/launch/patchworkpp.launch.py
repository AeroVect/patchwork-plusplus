from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import LoadComposableNodes, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# This configuration parameters are not exposed thorugh the launch system, meaning you can't modify
# those throw the ros launch CLI. If you need to change these values, you could write your own
# launch file and modify the 'parameters=' block from the Node class.
class config:
    # TBU. Examples are as follows:
    max_range: float = 80.0
    # deskew: bool = False


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    # tf tree configuration, these are the likely 3 parameters to change and nothing else
    base_frame = LaunchConfiguration("base_frame", default="base_link")

    # ROS configuration
    pointcloud_topic = LaunchConfiguration("cloud_topic")
    
    # Composable node container
    container = ComposableNodeContainer(
        name="patchworkpp_container",
        namespace="patchworkpp",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[],
        output="screen",
    )

    # Patchwork++ composable node
    patchworkpp_composable_node = ComposableNode(
        package="patchworkpp",
        plugin="patchworkpp_ros::GroundSegmentationServer",
        name="patchworkpp_ground_segmentation_node",
        parameters=[
            {
                # ROS node configuration
                "base_frame": base_frame,
                "use_sim_time": use_sim_time,
                # Patchwork++ configuration
                'sensor_height': 2.08,
                'num_iter': 3,  # Number of iterations for ground plane estimation using PCA.
                'num_lpr': 20,  # Maximum number of points to be selected as lowest points representative.
                'num_min_pts': 0,  # Minimum number of points to be estimated as ground plane in each patch.
                'th_seeds': 0.3,
                # threshold for lowest point representatives using in initial seeds selection of ground points.
                'th_dist': 0.125,  # threshold for thickness of ground.
                'th_seeds_v': 0.25,
                # threshold for lowest point representatives using in initial seeds selection of vertical structural points.
                'th_dist_v': 0.1,  # threshold for thickness of vertical structure.
                'max_range': 200.0,  # max_range of ground estimation area
                'min_range': 1.0,  # min_range of ground estimation area
                'uprightness_thr': 0.101,
                # threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
                'verbose': True  # display verbose info
            }
        ],
        extra_arguments=[{'use_intra_process_comms': LaunchConfiguration('use_intra_process_comms', default='false')}]
    )

    patchwork_composable_node = LoadComposableNodes(
        target_container=container.name,
        composable_node_descriptions=[patchworkpp_composable_node],
    )
            
    return LaunchDescription(
        [
            patchwork_composable_node
        ]
    )

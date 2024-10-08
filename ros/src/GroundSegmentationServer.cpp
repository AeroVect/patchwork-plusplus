#include <Eigen/Core>
#include <memory>
#include <utility>
#include <vector>

// Patchwork++-ROS
#include "GroundSegmentationServer.hpp"
#include "Utils.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

namespace patchworkpp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

GroundSegmentationServer::GroundSegmentationServer(const rclcpp::NodeOptions &options)
  : rclcpp::Node("patchworkpp_node", options) {

  patchwork::Params params;
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  params.sensor_height = declare_parameter<double>("sensor_height", params.sensor_height);
  params.num_iter      = declare_parameter<int>("num_iter", params.num_iter);
  params.num_lpr       = declare_parameter<int>("num_lpr", params.num_lpr);
  params.num_min_pts   = declare_parameter<int>("num_min_pts", params.num_min_pts);
  params.th_seeds      = declare_parameter<double>("th_seeds", params.th_seeds);

  params.th_dist    = declare_parameter<double>("th_dist", params.th_dist);
  params.th_seeds_v = declare_parameter<double>("th_seeds_v", params.th_seeds_v);
  params.th_dist_v  = declare_parameter<double>("th_dist_v", params.th_dist_v);

  params.max_range       = declare_parameter<double>("max_range", params.max_range);
  params.min_range       = declare_parameter<double>("min_range", params.min_range);
  params.uprightness_thr = declare_parameter<double>("uprightness_thr", params.uprightness_thr);

  params.verbose = get_parameter<bool>("verbose", params.verbose);

  // ToDo. Support intensity
  params.enable_RNR = false;

  // Construct the main Patchwork++ node
  Patchworkpp_ = std::make_unique<patchwork::PatchWorkpp>(params);

  // Initialize subscribers
  pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "pointcloud_topic", rclcpp::SensorDataQoS(),
    std::bind(&GroundSegmentationServer::EstimateGround, this, std::placeholders::_1));

  /*
   * We use the following QoS setting for reliable ground segmentation.
   * If you want to run Patchwork++ in real-time and real-world operation,
   * please change the QoS setting
   */
//  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

  cloud_publisher_     = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/cloud", qos);
  ground_publisher_    = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ground", qos);
  nonground_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonground", qos);
    
  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

void GroundSegmentationServer::EstimateGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  // Extract points and intensities from the incoming PointCloud2 message
  std::vector<Eigen::Vector3d> points;
  std::vector<float> intensities;
  PointCloud2ToEigen(msg, points, intensities);

  // Convert points to Eigen::MatrixX3f for processing
  Eigen::MatrixX3f cloud(points.size(), 3);
  for (size_t i = 0; i < points.size(); ++i) {
    cloud.row(i) = points[i].cast<float>();
  }

  // Estimate ground using Patchwork++
  Patchworkpp_->estimateGround(cloud);

  // Get indices of ground and nonground points
  Eigen::VectorXi ground_indices = Patchworkpp_->getGroundIndices();
  Eigen::VectorXi nonground_indices = Patchworkpp_->getNongroundIndices();


  // Extract ground points and intensities
  Eigen::MatrixX3f ground_points(ground_indices.size(), 3);
  std::vector<float> ground_intensities(ground_indices.size());

  for (int i = 0; i < ground_indices.size(); ++i) {
      int idx = ground_indices(i); 
      ground_points.row(i) = cloud.row(idx);
      ground_intensities[i] = intensities[idx];
  }

  // Extract nonground points and intensities
  Eigen::MatrixX3f nonground_points(nonground_indices.size(), 3);
  std::vector<float> nonground_intensities(nonground_indices.size());

  for (int i = 0; i < nonground_indices.size(); ++i) {
      int idx = nonground_indices(i); 
      nonground_points.row(i) = cloud.row(idx);
      nonground_intensities[i] = intensities[idx];
  }

  // Publish the clouds with intensity values
  PublishClouds(ground_points, ground_intensities, nonground_points, nonground_intensities, msg->header);
}

void GroundSegmentationServer::PublishClouds(const Eigen::MatrixX3f &est_ground,
                                             const std::vector<float> &ground_intensities,
                                             const Eigen::MatrixX3f &est_nonground,
                                             const std::vector<float> &nonground_intensities,
                                             const std_msgs::msg::Header &header_msg) {

  std_msgs::msg::Header header = header_msg;
  header.frame_id = base_frame_;

  ground_publisher_->publish(std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_ground, ground_intensities, header)));
  nonground_publisher_->publish(std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_nonground, nonground_intensities, header)));
}

}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)

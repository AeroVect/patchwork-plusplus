#include <Eigen/Core>
#include <memory>
#include <utility>
#include <vector>
#include <chrono>

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

  // Parameters for base link proximity filtering
  sensor_height_              = params.sensor_height;
  base_link_proximity_radius_ = declare_parameter<double>("base_link_proximity_radius", 10.0);
  max_ground_height_          = declare_parameter<double>("max_ground_height", 1.0);

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

  cloud_publisher_     = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/cloud", 1);
  ground_publisher_    = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ground", 1);
  nonground_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonground", 1);

  processing_time_publisher_ = create_publisher<std_msgs::msg::Float32>("/patchworkpp/processing_time", 1);
    
  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

void GroundSegmentationServer::EstimateGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  // Extract points and intensities from the incoming PointCloud2 message
  std::vector<Eigen::Vector3d> points;
  std::vector<float> intensities;

  // start timer
  auto start = std::chrono::high_resolution_clock::now();

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

  // end timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time and convert to milliseconds
  std::chrono::duration<double, std::milli> elapsed_seconds = end - start;

  // Publish the processing time
  std_msgs::msg::Float32 processing_time;
  processing_time.data = elapsed_seconds.count();
  processing_time_publisher_->publish(processing_time);
  // Reclassify the ground and nonground points
  std::pair<sensor_msgs::msg::PointCloud2::SharedPtr, sensor_msgs::msg::PointCloud2::SharedPtr> reclassified_clouds = ReclassifyGroundPoints(ground_points, ground_intensities, nonground_points, nonground_intensities, msg->header);
  // Publish the clouds with intensity values
  PublishClouds(reclassified_clouds.first, reclassified_clouds.second);
}

std::pair<sensor_msgs::msg::PointCloud2::SharedPtr, sensor_msgs::msg::PointCloud2::SharedPtr>
GroundSegmentationServer::ReclassifyGroundPoints(const Eigen::MatrixX3f &est_ground,
                                                 const std::vector<float> &ground_intensities,
                                                 const Eigen::MatrixX3f &est_nonground,
                                                 const std::vector<float> &nonground_intensities,
                                                 const std_msgs::msg::Header &header_msg) 
{
  std_msgs::msg::Header header = header_msg;
  header.frame_id = base_frame_;

  // Convert Eigen matrices to PointCloud2 and then to PCL clouds
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> ground_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  patchworkpp_ros::utils::EigenMatToPCL(est_ground, ground_intensities, *ground_pcl);

  std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> nonground_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  patchworkpp_ros::utils::EigenMatToPCL(est_nonground, nonground_intensities, *nonground_pcl);

  auto reclassified_ground_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  auto reclassified_nonground_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

  // Copy all nonground points to reclassified non-ground point cloud
  *reclassified_nonground_pcl = *nonground_pcl;

  // Use CropBox for filtering ground points based on proximity to the base link
  pcl::CropBox<pcl::PointXYZI> base_link_proximity_filter;
  base_link_proximity_filter.setMin(Eigen::Vector4f(-base_link_proximity_radius_, -base_link_proximity_radius_, max_ground_height_ - sensor_height_, 1.0)); 
  base_link_proximity_filter.setMax(Eigen::Vector4f(base_link_proximity_radius_, base_link_proximity_radius_, std::numeric_limits<float>::max(), 1.0));
  base_link_proximity_filter.setInputCloud(ground_pcl);

  // Extract the points inside these cropbox that need to be
  // reclassified from ground to nonground
  base_link_proximity_filter.setNegative(false); // Retain points in the cropbox
  auto inside_region_pointcloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  base_link_proximity_filter.filter(*inside_region_pointcloud);

  // Get the indices of the points inside the cropbox
  auto inside_region_indices = base_link_proximity_filter.getIndices();

  // Update reclassified_nonground_pcl by adding inside_region points
  *reclassified_nonground_pcl += *inside_region_pointcloud;
  // Extract the points outside the cropbox that need to be
  // kept as ground
  base_link_proximity_filter.setNegative(true); // Remove points in the cropbox
  auto outside_region_pointcloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  base_link_proximity_filter.filter(*outside_region_pointcloud);

  // Update reclassified_ground_pcl by adding outside_region points
  *reclassified_ground_pcl += *outside_region_pointcloud;

  // Convert reclassified PCL clouds to ROS PointCloud2 messages
  auto reclassified_ground_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(*reclassified_ground_pcl, *reclassified_ground_msg);
  reclassified_ground_msg->header = header;

  auto reclassified_nonground_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(*reclassified_nonground_pcl, *reclassified_nonground_msg);
  reclassified_nonground_msg->header = header;

  return {reclassified_ground_msg, reclassified_nonground_msg};
}

void GroundSegmentationServer::PublishClouds(sensor_msgs::msg::PointCloud2::SharedPtr ground_msg, sensor_msgs::msg::PointCloud2::SharedPtr nonground_msg) {

  ground_publisher_->publish(*ground_msg);
  nonground_publisher_->publish(*nonground_msg);
}

}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)

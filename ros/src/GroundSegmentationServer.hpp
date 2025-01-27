// Patchwork++
#include "patchwork/patchworkpp.h"

// pcl
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <string>

namespace patchworkpp_ros {

class GroundSegmentationServer : public rclcpp::Node {
 public:
  /// GroundSegmentationServer constructor
  GroundSegmentationServer() = delete;
  explicit GroundSegmentationServer(const rclcpp::NodeOptions &options);

 private:
  /// Register new frame
  void EstimateGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

  std::pair<sensor_msgs::msg::PointCloud2::SharedPtr, sensor_msgs::msg::PointCloud2::SharedPtr>
  ReclassifyGroundPoints(const Eigen::MatrixX3f &est_ground, 
                         const std::vector<float> &ground_intensities, 
                         const Eigen::MatrixX3f &est_nonground, 
                         const std::vector<float> &nonground_intensities, 
                         const std_msgs::msg::Header &header_msg); 

  /// Stream the point clouds for visualization
  void PublishClouds(sensor_msgs::msg::PointCloud2::SharedPtr ground_msg, sensor_msgs::msg::PointCloud2::SharedPtr nonground_msg);

 private:
  /// Data subscribers.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  /// Data publishers.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;

  // Timing publisher
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr processing_time_publisher_;
  

  /// Patchwork++
  std::unique_ptr<patchwork::PatchWorkpp> Patchworkpp_;

  std::string base_frame_{"base_link"};
  double sensor_height_{1.0};
  double base_link_proximity_radius_{10.0};
  double max_ground_height_{1.0};
};

}  // namespace patchworkpp_ros

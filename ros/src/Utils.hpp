#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

// ROS 2
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace patchworkpp_ros::utils {

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointField = sensor_msgs::msg::PointField;
using Header = std_msgs::msg::Header;

inline std::string FixFrameId(const std::string &frame_id) {
  return std::regex_replace(frame_id, std::regex("^/"), "");
}

inline std::optional<PointField> GetTimestampField(const PointCloud2::ConstSharedPtr msg) {
  PointField timestamp_field;
  for (const auto &field : msg->fields) {
    if ((field.name == "t" || field.name == "timestamp" || field.name == "time")) {
      timestamp_field = field;
    }
  }
  if (timestamp_field.count) return timestamp_field;
  RCLCPP_WARN_ONCE(rclcpp::get_logger("patchworkpp_node"),
                   "Field 't', 'timestamp', or 'time' does not exist. "
                   "Disabling scan deskewing");
  return {};
}

// Normalize timestamps from 0.0 to 1.0
inline auto NormalizeTimestamps(const std::vector<double> &timestamps) {
  const auto   [min_it, max_it] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
  const double min_timestamp    = *min_it;
  const double max_timestamp    = *max_it;

  std::vector<double> timestamps_normalized(timestamps.size());
  std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                 [&](const auto &timestamp) {
                   return (timestamp - min_timestamp) / (max_timestamp - min_timestamp);
                 });
  return timestamps_normalized;
}

inline auto ExtractTimestampsFromMsg(const PointCloud2::ConstSharedPtr msg,
                                     const PointField &timestamp_field) {
  auto extract_timestamps =
      [&msg]<typename T>(sensor_msgs::PointCloud2ConstIterator<T> &&it) -> std::vector<double> {
        const size_t        n_points = msg->height * msg->width;
        std::vector<double> timestamps;
        timestamps.reserve(n_points);
        for (size_t i = 0; i < n_points; ++i, ++it) {
          timestamps.emplace_back(static_cast<double>(*it));
        }
        return NormalizeTimestamps(timestamps);
      };

  // Determine the correct iterator type based on the timestamp field datatype
  using sensor_msgs::PointCloud2ConstIterator;
  if (timestamp_field.datatype == PointField::UINT32) {
    return extract_timestamps(PointCloud2ConstIterator<uint32_t>(*msg, timestamp_field.name));
  } else if (timestamp_field.datatype == PointField::FLOAT32) {
    return extract_timestamps(PointCloud2ConstIterator<float>(*msg, timestamp_field.name));
  } else if (timestamp_field.datatype == PointField::FLOAT64) {
    return extract_timestamps(PointCloud2ConstIterator<double>(*msg, timestamp_field.name));
  }

  throw std::runtime_error("Timestamp field type not supported");
}

inline std::unique_ptr<PointCloud2> CreatePointCloud2Msg(const size_t n_points,
                                                         const Header &header) {
  auto cloud_msg = std::make_unique<PointCloud2>();
  sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
  cloud_msg->header          = header;
  cloud_msg->header.frame_id = FixFrameId(cloud_msg->header.frame_id);
  cloud_msg->fields.clear();
  int offset = 0;
  offset = addPointField(*cloud_msg, "x", 1, PointField::FLOAT32, offset);
  offset = addPointField(*cloud_msg, "y", 1, PointField::FLOAT32, offset);
  offset = addPointField(*cloud_msg, "z", 1, PointField::FLOAT32, offset);
  offset = addPointField(*cloud_msg, "intensity", 1, PointField::FLOAT32, offset);
  offset = addPointField(*cloud_msg, "time", 1, PointField::FLOAT64, offset);

  // Calculate point step and resize data
  cloud_msg->point_step = offset;
  cloud_msg->row_step   = cloud_msg->width * cloud_msg->point_step;
  cloud_msg->data.resize(cloud_msg->height * cloud_msg->row_step);
  modifier.resize(n_points);
  return cloud_msg;
}

inline void FillPointCloud2XYZ(const std::vector<Eigen::Vector3d> &points, PointCloud2 &msg) {
  sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
  for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z) {
    const Eigen::Vector3d &point = points[i];
    *msg_x = static_cast<float>(point.x());
    *msg_y = static_cast<float>(point.y());
    *msg_z = static_cast<float>(point.z());
  }
}

inline void FillPointCloud2XYZ(const Eigen::MatrixX3f &points, PointCloud2 &msg) {
  sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
  for (size_t i = 0; i < points.rows(); ++i, ++msg_x, ++msg_y, ++msg_z) {
    *msg_x = points(i, 0);
    *msg_y = points(i, 1);
    *msg_z = points(i, 2);
  }
}

inline void FillPointCloud2Intensity(const std::vector<float> &intensities, PointCloud2 &msg) {
  sensor_msgs::PointCloud2Iterator<float> msg_intensity(msg, "intensity");
  for (size_t i = 0; i < intensities.size(); i++, ++msg_intensity) {
    *msg_intensity = intensities[i];
  }
}

inline void FillPointCloud2Timestamp(const std::vector<double> &timestamps, PointCloud2 &msg) {
  sensor_msgs::PointCloud2Iterator<double> msg_t(msg, "time");
  for (size_t i = 0; i < timestamps.size(); i++, ++msg_t) {
    *msg_t = timestamps[i];
  }
}

inline std::vector<double> GetTimestamps(const PointCloud2::ConstSharedPtr msg) {
  auto timestamp_field = GetTimestampField(msg);
  if (!timestamp_field.has_value()) return {};

  // Extract timestamps from cloud_msg
  std::vector<double> timestamps = ExtractTimestampsFromMsg(msg, timestamp_field.value());

  return timestamps;
}

inline void PointCloud2ToEigen(const PointCloud2::ConstSharedPtr msg,
                               std::vector<Eigen::Vector3d> &points,
                               std::vector<float> &intensities) {
  size_t n_points = msg->height * msg->width;
  points.reserve(n_points);
  intensities.reserve(n_points);
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> msg_intensity(*msg, "intensity");

  for (size_t i = 0; i < n_points; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity) {
    points.emplace_back(*msg_x, *msg_y, *msg_z);
    intensities.emplace_back(*msg_intensity);
  }
}

inline Eigen::MatrixXf PointCloud2ToEigenMat(const PointCloud2::ConstSharedPtr &msg,
                                             Eigen::VectorXf &intensities) {
  size_t n_points = msg->height * msg->width;
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> msg_intensity(*msg, "intensity");

  Eigen::MatrixXf points(n_points, 3);
  intensities.resize(n_points);

  for (size_t i = 0; i < n_points; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity) {
    points.row(i) << *msg_x, *msg_y, *msg_z;
    intensities(i) = *msg_intensity;
  }

  return points;
}

inline std::unique_ptr<PointCloud2> EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                                                       const std::vector<float> &intensities,
                                                       const Header &header) {
  if (points.size() != intensities.size()) {
    throw std::runtime_error("Points and intensities vector sizes do not match.");
  }
  auto msg = CreatePointCloud2Msg(points.size(), header);
  FillPointCloud2XYZ(points, *msg);
  FillPointCloud2Intensity(intensities, *msg);
  // Optionally, fill timestamps if needed
  return msg;
}

inline std::unique_ptr<PointCloud2> EigenMatToPointCloud2(const Eigen::MatrixX3f &points,
                                                          const std::vector<float> &intensities,
                                                          const Header &header) {
  if (points.rows() != intensities.size()) {
    throw std::runtime_error("Points and intensities vector sizes do not match.");
  }
  auto msg = CreatePointCloud2Msg(points.rows(), header);
  FillPointCloud2XYZ(points, *msg);
  FillPointCloud2Intensity(intensities, *msg);
  return msg;
}

inline void EigenMatToPCL(const Eigen::MatrixX3f &matrix, 
                          const std::vector<float> &intensities, 
                          pcl::PointCloud<pcl::PointXYZI> &cloud) {
  cloud.clear();
  for (int i = 0; i < matrix.rows(); ++i) {
    pcl::PointXYZI point;
    point.x = matrix(i, 0);
    point.y = matrix(i, 1);
    point.z = matrix(i, 2);
    point.intensity = intensities[i];
    cloud.push_back(point);
  }
}

}  // namespace patchworkpp_ros::utils

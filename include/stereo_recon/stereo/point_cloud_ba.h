#ifndef STEREO_RECON_STEREO_POINT_CLOUD_BA_H_
#define STEREO_RECON_STEREO_POINT_CLOUD_BA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "opencv2/core.hpp"

namespace stereo_recon {

struct PointCloudBaParameters {
  int window_size = 10;
  int sample_stride = 4;
  int max_samples_per_frame = 5000;
  int max_samples_per_cluster = 64;
  int min_points_per_plane = 24;
  int min_frames_per_plane = 3;
  int min_planes = 12;
  int max_planes = 200;
  int max_voxel_depth = 2;
  int max_iterations = 15;
  int num_threads = 1;
  double root_voxel_size = 0.8;
  double plane_eigenvalue_ratio = 0.1;
  double max_plane_rms = 0.05;
  double min_plane_spread = 0.02;
  double min_normal_eigenvalue = 0.01;
  double disparity_sigma = 0.5;
  double depth_edge_absolute = 0.05;
  double depth_edge_relative = 0.02;
  double noise_floor = 0.005;
  double huber_delta = 2.5;
  double vio_translation_sigma = 0.15;
  double vio_rotation_sigma_deg = 5.0;
  double max_translation_update = 0.2;
  double max_rotation_update_deg = 5.0;
  double max_vio_translation = 1.0;
  double max_vio_rotation_deg = 15.0;
  double max_solver_time_sec = 0.15;
};

struct BaPoint {
  Eigen::Vector3d position;
  Eigen::Matrix3d covariance;
};

// Poses map rectified camera coordinates into the respective reference frame.
// cv::Mat buffers must be owned by the frame, not borrowed ROS image storage.
struct MappingKeyframe {
  uint64_t id = 0;
  uint64_t sequence = 0;
  int64_t stamp_ns = 0;
  cv::Mat depth;
  cv::Mat color;
  cv::Matx33d intrinsics;
  double baseline = 0.0;
  Eigen::Isometry3d vio_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d map_pose = Eigen::Isometry3d::Identity();
  std::vector<BaPoint> samples;
};

struct BaSummary {
  bool accepted = false;
  size_t planes = 0;
  size_t clusters = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double normal_eigenvalue = 0.0;
  double time_ms = 0.0;
  std::string reason = "waiting_for_window";
};

// Joint pose/plane BA with fixed, depth-aware point-cluster weights. This is
// explicit plane least squares, not BALM2's analytical eigenvalue solver.
class PointCloudBundleAdjuster {
 public:
  explicit PointCloudBundleAdjuster(PointCloudBaParameters parameters);
  void PrepareKeyframe(MappingKeyframe* frame) const;
  // Mutates map_pose only when the entire solution passes acceptance checks.
  BaSummary Optimize(const std::vector<MappingKeyframe*>& frames) const;

 private:
  PointCloudBaParameters parameters_;
};

}  // namespace stereo_recon

#endif  // STEREO_RECON_STEREO_POINT_CLOUD_BA_H_

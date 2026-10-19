#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace msckf_vio::stereo {

// T_A_B maps coordinates from B to A throughout this module.
Eigen::Isometry3d transformFromRowMajor(const std::vector<double>& values);

struct CameraCalibration {
  cv::Size size;
  cv::Matx33d K;
  cv::Mat distortion;
  std::string distortion_model;
};

class StereoRectifier {
 public:
  StereoRectifier(const CameraCalibration& left, const CameraCalibration& right,
                  const Eigen::Isometry3d& T_right_left,
                  const Eigen::Isometry3d& T_body_left, double alpha);
  void rectify(const cv::Mat& left, const cv::Mat& right,
               cv::Mat& left_rect, cv::Mat& right_rect) const;
  cv::Size size() const { return size_; }
  const cv::Matx44d& Q() const { return Q_; }
  const cv::Matx34d& leftProjection() const { return P1_; }
  const cv::Matx34d& rightProjection() const { return P2_; }
  const cv::Mat& leftMask() const { return left_mask_; }
  const cv::Mat& rightMask() const { return right_mask_; }
  const Eigen::Isometry3d& bodyFromLeftRect() const { return T_body_left_rect_; }
  const Eigen::Isometry3d& bodyFromRightRect() const { return T_body_right_rect_; }

 private:
  cv::Size size_;
  cv::Mat left_map_x_, left_map_y_, right_map_x_, right_map_y_;
  cv::Mat left_mask_, right_mask_;
  cv::Matx44d Q_;
  cv::Matx34d P1_, P2_;
  Eigen::Isometry3d T_body_left_rect_, T_body_right_rect_;
};

struct ColorPoint {
  Eigen::Vector3d position;
  uint8_t r = 0, g = 0, b = 0;
};

struct FrameGeometry {
  cv::Mat depth;  // CV_32FC1, optical Z in metres, NaN where invalid.
  std::vector<ColorPoint> points;
};

FrameGeometry projectDisparity(const cv::Mat& disparity, const cv::Mat& left,
    const cv::Matx44d& Q, const cv::Mat& left_mask, const cv::Mat& right_mask,
    double min_depth, double max_depth, int pixel_stride);

class PoseBuffer {
 public:
  explicit PoseBuffer(size_t capacity);
  void add(int64_t stamp_ns, const Eigen::Isometry3d& pose);
  std::optional<Eigen::Isometry3d> find(int64_t stamp_ns, int64_t tolerance_ns) const;
  void clear() { poses_.clear(); }
 private:
  size_t capacity_;
  std::map<int64_t, Eigen::Isometry3d> poses_;
};

class KeyframeSelector {
 public:
  KeyframeSelector(double translation, double rotation);
  bool shouldInsert(const Eigen::Isometry3d& pose) const;
  void accept(const Eigen::Isometry3d& pose) { previous_ = pose; }
  void clear() { previous_.reset(); }
 private:
  double translation_, rotation_;
  std::optional<Eigen::Isometry3d> previous_;
};

class VoxelMap {
 public:
  VoxelMap(double voxel_size, size_t max_voxels);
  // Once full, update existing voxels but do not allocate new ones.
  size_t integrate(const std::vector<ColorPoint>& points,
                   const Eigen::Isometry3d& T_world_camera);
  std::vector<ColorPoint> points() const;
  size_t size() const { return voxels_.size(); }
  void clear() { voxels_.clear(); }

 private:
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct Hash { size_t operator()(const Key& key) const; };
  struct Voxel {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d color = Eigen::Vector3d::Zero();
    uint32_t observations = 0;
  };
  double voxel_size_;
  size_t max_voxels_;
  std::unordered_map<Key, Voxel, Hash> voxels_;
};

// ASCII PCD/PLY; throws on unsupported extensions and I/O failures.
void savePointCloud(const std::string& path, const std::vector<ColorPoint>& points);

}  // namespace msckf_vio::stereo

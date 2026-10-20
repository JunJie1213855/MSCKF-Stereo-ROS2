#ifndef MSCKF_VIO_STEREO_RECONSTRUCTION_H_
#define MSCKF_VIO_STEREO_RECONSTRUCTION_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Eigen/Geometry"
#include "opencv2/core.hpp"

namespace msckf_vio::stereo {

// t_a_b 表示把 B 坐标变换到 A；外参输入为按行排列的 4x4 刚体矩阵。
Eigen::Isometry3d TransformFromRowMajor(const std::vector<double>& values);

struct CameraCalibration {
  cv::Size size;
  cv::Matx33d intrinsics;
  cv::Mat distortion;
  std::string distortion_model;
};

// 启动时缓存校正映射；每帧仅 remap。保留逆校正旋转供相机位姿转换使用。
class StereoRectifier {
 public:
  StereoRectifier(const CameraCalibration& left, const CameraCalibration& right,
                  const Eigen::Isometry3d& t_right_left,
                  const Eigen::Isometry3d& t_body_left, double alpha);
  void Rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_rect,
               cv::Mat& right_rect) const;
  cv::Size size() const { return size_; }
  const cv::Matx44d& reprojection_matrix() const {
    return reprojection_matrix_;
  }
  const cv::Matx34d& left_projection() const { return left_projection_; }
  const cv::Matx34d& right_projection() const { return right_projection_; }
  const cv::Mat& left_mask() const { return left_mask_; }
  const cv::Mat& right_mask() const { return right_mask_; }
  const Eigen::Isometry3d& body_from_left_rect() const {
    return body_from_left_rect_;
  }
  const Eigen::Isometry3d& body_from_right_rect() const {
    return body_from_right_rect_;
  }

 private:
  cv::Size size_;
  cv::Mat left_map_x_, left_map_y_, right_map_x_, right_map_y_;
  cv::Mat left_mask_, right_mask_;
  cv::Matx44d reprojection_matrix_;
  cv::Matx34d left_projection_, right_projection_;
  Eigen::Isometry3d body_from_left_rect_, body_from_right_rect_;
};

struct ColorPoint {
  Eigen::Vector3d position;
  uint8_t r = 0, g = 0, b = 0;
};

struct FrameGeometry {
  cv::Mat depth;  // 左校正相机的光轴 Z，CV_32FC1，单位米，无效值为 NaN。
  std::vector<ColorPoint> points;
};

// 用 Q 矩阵齐次反投影；pixel_stride 只抽样点云，深度图仍为完整分辨率。
FrameGeometry ProjectDisparity(const cv::Mat& disparity, const cv::Mat& left,
                               const cv::Matx44d& reprojection_matrix,
                               const cv::Mat& left_mask,
                               const cv::Mat& right_mask, double min_depth,
                               double max_depth, int pixel_stride);

// 有界的时间戳缓存；容差为 0 时必须精确匹配，不做插值。调用者负责加锁。
class PoseBuffer {
 public:
  explicit PoseBuffer(size_t capacity);
  void Add(int64_t stamp_ns, const Eigen::Isometry3d& pose);
  std::optional<Eigen::Isometry3d> Find(int64_t stamp_ns,
                                        int64_t tolerance_ns) const;
  void Clear() { poses_.clear(); }

 private:
  size_t capacity_;
  std::map<int64_t, Eigen::Isometry3d> poses_;
};

// 与上次已融合位姿比较：平移或转角任意一个达到阈值即可插入。
class KeyframeSelector {
 public:
  KeyframeSelector(double translation, double rotation);
  bool ShouldInsert(const Eigen::Isometry3d& pose) const;
  void Accept(const Eigen::Isometry3d& pose) { previous_ = pose; }
  void Clear() { previous_.reset(); }

 private:
  double translation_, rotation_;
  std::optional<Eigen::Isometry3d> previous_;
};

// 世界坐标系下的体素质心地图，累计位置与颜色观测；当前实现不是 TSDF。
class VoxelMap {
 public:
  VoxelMap(double voxel_size, size_t max_voxels);
  // 满容量后仍更新已有体素，但拒绝新增；返回因容量不足跳过的点数。
  size_t Integrate(const std::vector<ColorPoint>& points,
                   const Eigen::Isometry3d& t_world_camera);
  std::vector<ColorPoint> points() const;
  size_t size() const { return voxels_.size(); }
  void Clear() { voxels_.clear(); }

 private:
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct Hash {
    size_t operator()(const Key& key) const;
  };
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
void SavePointCloud(const std::string& path,
                    const std::vector<ColorPoint>& points);

}  // namespace msckf_vio::stereo

#endif  // MSCKF_VIO_STEREO_RECONSTRUCTION_H_

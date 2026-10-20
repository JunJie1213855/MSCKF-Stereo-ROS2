#include "msckf_vio/stereo/reconstruction.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"

namespace msckf_vio::stereo {
namespace {
Eigen::Isometry3d InverseRectification(const cv::Mat& rotation) {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  Eigen::Matrix3d rotation_matrix;
  cv::cv2eigen(rotation, rotation_matrix);
  transform.linear() = rotation_matrix.transpose();
  return transform;
}

void ValidateCamera(const CameraCalibration& camera) {
  if (camera.size.width <= 0 || camera.size.height <= 0 ||
      !cv::checkRange(cv::Mat(camera.intrinsics)) ||
      camera.intrinsics(0, 0) <= 0 || camera.intrinsics(1, 1) <= 0 ||
      camera.distortion.empty() || !cv::checkRange(camera.distortion)) {
    throw std::invalid_argument("Invalid camera calibration");
  }
}
}  // namespace

Eigen::Isometry3d TransformFromRowMajor(const std::vector<double>& values) {
  if (values.size() != 16) {
    throw std::invalid_argument(
        "A calibration transform needs 16 row-major values");
  }
  using RowMajorMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
  const Eigen::Matrix4d matrix =
      Eigen::Map<const RowMajorMatrix4d>(values.data());
  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  if (!matrix.allFinite() ||
      (matrix.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9 ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() >
          1e-3 ||
      std::abs(rotation.determinant() - 1.0) > 1e-3) {
    throw std::invalid_argument(
        "Calibration transform is not a finite rigid transform");
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.matrix() = matrix;
  return result;
}

StereoRectifier::StereoRectifier(const CameraCalibration& left,
                                 const CameraCalibration& right,
                                 const Eigen::Isometry3d& t_right_left,
                                 const Eigen::Isometry3d& t_body_left,
                                 double alpha)
    : size_(left.size) {
  ValidateCamera(left);
  ValidateCamera(right);
  if (left.size != right.size ||
      left.distortion_model != right.distortion_model ||
      !std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0 ||
      t_right_left.translation().norm() < 1e-6) {
    throw std::invalid_argument(
        "Invalid stereo geometry, distortion models or rectification alpha");
  }
  cv::Mat rotation_matrix, translation, left_rect_rotation, right_rect_rotation,
      left_projection, right_projection, reprojection_matrix;
  cv::eigen2cv(Eigen::Matrix3d(t_right_left.linear()), rotation_matrix);
  cv::eigen2cv(Eigen::Vector3d(t_right_left.translation()), translation);
  if (left.distortion_model == "radtan" ||
      left.distortion_model == "plumb_bob") {
    cv::stereoRectify(left.intrinsics, left.distortion, right.intrinsics,
                      right.distortion, size_, rotation_matrix, translation,
                      left_rect_rotation, right_rect_rotation, left_projection,
                      right_projection, reprojection_matrix,
                      cv::CALIB_ZERO_DISPARITY, alpha, size_);
    cv::initUndistortRectifyMap(left.intrinsics, left.distortion,
                                left_rect_rotation, left_projection, size_,
                                CV_32FC1, left_map_x_, left_map_y_);
    cv::initUndistortRectifyMap(right.intrinsics, right.distortion,
                                right_rect_rotation, right_projection, size_,
                                CV_32FC1, right_map_x_, right_map_y_);
  } else if (left.distortion_model == "equidistant") {
    cv::fisheye::stereoRectify(
        left.intrinsics, left.distortion, right.intrinsics, right.distortion,
        size_, rotation_matrix, translation, left_rect_rotation,
        right_rect_rotation, left_projection, right_projection,
        reprojection_matrix, cv::CALIB_ZERO_DISPARITY, size_, alpha, 1.0);
    cv::fisheye::initUndistortRectifyMap(
        left.intrinsics, left.distortion, left_rect_rotation, left_projection,
        size_, CV_32FC1, left_map_x_, left_map_y_);
    cv::fisheye::initUndistortRectifyMap(
        right.intrinsics, right.distortion, right_rect_rotation,
        right_projection, size_, CV_32FC1, right_map_x_, right_map_y_);
  } else {
    throw std::invalid_argument("Unsupported distortion model: " +
                                left.distortion_model);
  }
  // 阶段 1：stereoRectify 使左右对应点落在同一行，约定左视差
  // d=x_left-x_right>0。 Q 把 [x,y,d,1]
  // 映射到齐次三维坐标；基线以米标定，深度自然以米输出。
  if (!cv::checkRange(reprojection_matrix) ||
      right_projection.at<double>(0, 3) >= -1e-6 ||
      std::abs(right_projection.at<double>(1, 3)) > 1e-6) {
    throw std::invalid_argument(
        "Stereo matcher needs horizontal stereo with right camera to the "
        "right");
  }
  reprojection_matrix_ = reprojection_matrix;
  left_projection_ = left_projection;
  right_projection_ = right_projection;
  // 阶段 2：反投影点位于“校正相机”坐标系，须先乘 R1^T 回到原相机，
  // 再经外参进入机体。只用原相机外参会把校正旋转遗漏，造成地图倾斜。
  body_from_left_rect_ = t_body_left * InverseRectification(left_rect_rotation);
  body_from_right_rect_ = t_body_left * t_right_left.inverse() *
                          InverseRectification(right_rect_rotation);

  // 阶段 3：检测 remap 的有效区域。黑色补边也可能产生正视差，不能直接融合。
  // 只有双线性采样涉及的原图像素都有效，校正像素才被标记为有效。
  cv::Mat valid_source(size_, CV_32FC1, cv::Scalar(1.0));
  cv::Mat left_valid, right_valid;
  cv::remap(valid_source, left_valid, left_map_x_, left_map_y_,
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::remap(valid_source, right_valid, right_map_x_, right_map_y_,
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  left_mask_ = left_valid >= 0.999f;
  right_mask_ = right_valid >= 0.999f;
}

void StereoRectifier::Rectify(const cv::Mat& left, const cv::Mat& right,
                              cv::Mat& left_rect, cv::Mat& right_rect) const {
  if (left.size() != size_ || right.size() != size_) {
    throw std::invalid_argument(
        "Image resolution differs from calibration; no implicit resizing is "
        "allowed");
  }
  cv::remap(left, left_rect, left_map_x_, left_map_y_, cv::INTER_LINEAR);
  cv::remap(right, right_rect, right_map_x_, right_map_y_, cv::INTER_LINEAR);
}

FrameGeometry ProjectDisparity(const cv::Mat& disparity, const cv::Mat& left,
                               const cv::Matx44d& reprojection_matrix,
                               const cv::Mat& left_mask,
                               const cv::Mat& right_mask, double min_depth,
                               double max_depth, int pixel_stride) {
  if (disparity.type() != CV_32FC1 || disparity.size() != left.size() ||
      (left.type() != CV_8UC1 && left.type() != CV_8UC3) ||
      left_mask.type() != CV_8UC1 || right_mask.type() != CV_8UC1 ||
      disparity.size() != left_mask.size() ||
      disparity.size() != right_mask.size() || !std::isfinite(min_depth) ||
      !std::isfinite(max_depth) || min_depth <= 0 || max_depth <= min_depth ||
      pixel_stride <= 0) {
    throw std::invalid_argument(
        "Invalid disparity, image, masks or depth limits");
  }
  FrameGeometry result;
  result.depth = cv::Mat(disparity.size(), CV_32FC1,
                         cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  result.points.reserve(disparity.total() / pixel_stride / pixel_stride);
  for (int y = 0; y < disparity.rows; ++y) {
    for (int x = 0; x < disparity.cols; ++x) {
      const float d = disparity.at<float>(y, x);
      if (!std::isfinite(d) || d <= 0 || !left_mask.at<uint8_t>(y, x)) {
        continue;
      }
      // 先验证 x_right=x_left-d 的落点及其采样邻域，避免使用右图校正边界。
      const double xr = x - static_cast<double>(d);
      if (xr < 0 || xr > disparity.cols - 1) {
        continue;
      }
      if (!right_mask.at<uint8_t>(y, static_cast<int>(std::floor(xr))) ||
          !right_mask.at<uint8_t>(y, static_cast<int>(std::ceil(xr)))) {
        continue;
      }
      // 齐次反投影后必须除以 W；深度取光轴 Z，而不是到相机的欧氏距离。
      const cv::Vec4d homogeneous = reprojection_matrix * cv::Vec4d(x, y, d, 1);
      if (!std::isfinite(homogeneous[3]) || std::abs(homogeneous[3]) < 1e-12) {
        continue;
      }
      const Eigen::Vector3d point(homogeneous[0] / homogeneous[3],
                                  homogeneous[1] / homogeneous[3],
                                  homogeneous[2] / homogeneous[3]);
      if (!point.allFinite() || point.z() < min_depth ||
          point.z() > max_depth) {
        continue;
      }
      result.depth.at<float>(y, x) = static_cast<float>(point.z());
      // 深度图保留所有有效像素，只有点云输出按步长抽样。
      if (x % pixel_stride != 0 || y % pixel_stride != 0) {
        continue;
      }
      ColorPoint sample;
      sample.position = point;
      if (left.channels() == 1) {
        sample.r = sample.g = sample.b = left.at<uint8_t>(y, x);
      } else {
        const auto color = left.at<cv::Vec3b>(y, x);
        sample.r = color[2];
        sample.g = color[1];
        sample.b = color[0];
      }
      result.points.push_back(sample);
    }
  }
  return result;
}

PoseBuffer::PoseBuffer(size_t capacity) : capacity_(capacity) {
  if (!capacity) {
    throw std::invalid_argument("Pose cache capacity must be positive");
  }
}

void PoseBuffer::Add(int64_t stamp_ns, const Eigen::Isometry3d& pose) {
  poses_[stamp_ns] = pose;
  while (poses_.size() > capacity_) {
    poses_.erase(poses_.begin());
  }
}

std::optional<Eigen::Isometry3d> PoseBuffer::Find(int64_t stamp_ns,
                                                  int64_t tolerance_ns) const {
  if (poses_.empty() || tolerance_ns < 0) {
    return std::nullopt;
  }
  // lower_bound 两侧最多各检查一个候选。容差只允许选取近邻，不做位姿插值。
  auto after = poses_.lower_bound(stamp_ns);
  auto best = after;
  if (after != poses_.begin()) {
    auto before = std::prev(after);
    if (after == poses_.end() ||
        stamp_ns - before->first <= after->first - stamp_ns) {
      best = before;
    }
  }
  if (best != poses_.end() &&
      std::abs(best->first - stamp_ns) <= tolerance_ns) {
    return best->second;
  }
  return std::nullopt;
}

KeyframeSelector::KeyframeSelector(double translation, double rotation)
    : translation_(translation), rotation_(rotation) {
  if (!std::isfinite(translation) || !std::isfinite(rotation) ||
      translation < 0 || rotation < 0) {
    throw std::invalid_argument("Invalid keyframe thresholds");
  }
}

bool KeyframeSelector::ShouldInsert(const Eigen::Isometry3d& pose) const {
  if (!previous_) {
    return true;
  }
  const double translation =
      (pose.translation() - previous_->translation()).norm();
  const double rotation =
      Eigen::AngleAxisd(previous_->linear().transpose() * pose.linear())
          .angle();
  return translation >= translation_ || rotation >= rotation_;
}

VoxelMap::VoxelMap(double voxel_size, size_t max_voxels)
    : voxel_size_(voxel_size), max_voxels_(max_voxels) {
  if (!std::isfinite(voxel_size) || voxel_size <= 0 || max_voxels == 0) {
    throw std::invalid_argument("Voxel size and map capacity must be positive");
  }
}

size_t VoxelMap::Hash::operator()(const Key& key) const {
  size_t seed = std::hash<int64_t>{}(key.x);
  for (int64_t v : {key.y, key.z}) {
    seed ^= std::hash<int64_t>{}(v) + size_t{0x9e3779b9} + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

size_t VoxelMap::Integrate(const std::vector<ColorPoint>& points,
                           const Eigen::Isometry3d& t_world_camera) {
  size_t skipped = 0;
  for (const auto& point : points) {
    const Eigen::Vector3d world = t_world_camera * point.position;
    // floor 而非向零截断：负坐标的体素归属必须与正坐标对称。
    const Eigen::Array3d coordinates = (world / voxel_size_).array().floor();
    // Reject coordinates outside a safely representable integer range.
    if (!coordinates.isFinite().all() || (coordinates.abs() > 9e18).any()) {
      continue;
    }
    const Key key{static_cast<int64_t>(coordinates.x()),
                  static_cast<int64_t>(coordinates.y()),
                  static_cast<int64_t>(coordinates.z())};
    auto it = voxels_.find(key);
    if (it == voxels_.end()) {
      if (voxels_.size() >= max_voxels_) {
        ++skipped;
        continue;
      }
      it = voxels_.emplace(key, Voxel{}).first;
    }
    auto& voxel = it->second;
    // 按点累计位置/颜色均值；S2M2 置信度只负责前面的筛选，不在这里加权。
    // 观测数封顶后，更新变为步长 1/1000 的缓慢平均，避免旧地图完全冻结。
    voxel.observations = std::min<uint32_t>(voxel.observations + 1, 1000);
    const double weight = 1.0 / voxel.observations;
    voxel.position += weight * (world - voxel.position);
    const Eigen::Vector3d color(point.r, point.g, point.b);
    voxel.color += weight * (color - voxel.color);
  }
  return skipped;
}

std::vector<ColorPoint> VoxelMap::points() const {
  std::vector<ColorPoint> result;
  result.reserve(voxels_.size());
  for (const auto& entry : voxels_) {
    const auto& voxel = entry.second;
    result.push_back({voxel.position,
                      static_cast<uint8_t>(std::lround(voxel.color.x())),
                      static_cast<uint8_t>(std::lround(voxel.color.y())),
                      static_cast<uint8_t>(std::lround(voxel.color.z()))});
  }
  return result;
}

void SavePointCloud(const std::string& path,
                    const std::vector<ColorPoint>& points) {
  const auto extension = std::filesystem::path(path).extension().string();
  if (extension != ".pcd" && extension != ".ply") {
    throw std::invalid_argument("Map output path must end in .pcd or .ply");
  }
  std::ofstream output(path);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.imbue(std::locale::classic());
  output << std::setprecision(9);
  if (extension == ".pcd") {
    output << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
              "TYPE F F F U\nCOUNT 1 1 1 1\nWIDTH "
           << points.size() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "
           << points.size() << "\nDATA ascii\n";
  } else {
    output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
           << "\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar "
              "blue\nend_header\n";
  }
  for (const auto& point : points) {
    output << point.position.x() << ' ' << point.position.y() << ' '
           << point.position.z() << ' ';
    if (extension == ".pcd") {
      const uint32_t rgb = (static_cast<uint32_t>(point.r) << 16) |
                           (static_cast<uint32_t>(point.g) << 8) | point.b;
      output << rgb << '\n';
    } else {
      output << unsigned(point.r) << ' ' << unsigned(point.g) << ' '
             << unsigned(point.b) << '\n';
    }
  }
  output.close();
}
}  // namespace msckf_vio::stereo

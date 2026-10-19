#include <msckf_vio/stereo/reconstruction.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

namespace msckf_vio::stereo {
namespace {
Eigen::Isometry3d inverseRectification(const cv::Mat& rotation) {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  Eigen::Matrix3d R;
  cv::cv2eigen(rotation, R);
  transform.linear() = R.transpose();
  return transform;
}

void validateCamera(const CameraCalibration& camera) {
  if (camera.size.width <= 0 || camera.size.height <= 0 ||
      !cv::checkRange(cv::Mat(camera.K)) || camera.K(0, 0) <= 0 ||
      camera.K(1, 1) <= 0 || camera.distortion.empty() ||
      !cv::checkRange(camera.distortion))
    throw std::invalid_argument("Invalid camera calibration");
}
}  // namespace

Eigen::Isometry3d transformFromRowMajor(const std::vector<double>& values) {
  if (values.size() != 16)
    throw std::invalid_argument("A calibration transform needs 16 row-major values");
  using RowMatrix = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
  const Eigen::Matrix4d matrix = Eigen::Map<const RowMatrix>(values.data());
  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  if (!matrix.allFinite() ||
      (matrix.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9 ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-3 ||
      std::abs(rotation.determinant() - 1.0) > 1e-3)
    throw std::invalid_argument("Calibration transform is not a finite rigid transform");
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.matrix() = matrix;
  return result;
}

StereoRectifier::StereoRectifier(const CameraCalibration& left,
    const CameraCalibration& right, const Eigen::Isometry3d& T_right_left,
    const Eigen::Isometry3d& T_body_left, double alpha) : size_(left.size) {
  validateCamera(left);
  validateCamera(right);
  if (left.size != right.size || left.distortion_model != right.distortion_model ||
      !std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0 ||
      T_right_left.translation().norm() < 1e-6)
    throw std::invalid_argument("Invalid stereo geometry, distortion models or rectification alpha");
  cv::Mat R, translation, R1, R2, P1, P2, Q;
  cv::eigen2cv(Eigen::Matrix3d(T_right_left.linear()), R);
  cv::eigen2cv(Eigen::Vector3d(T_right_left.translation()), translation);
  if (left.distortion_model == "radtan" || left.distortion_model == "plumb_bob") {
    cv::stereoRectify(left.K, left.distortion, right.K, right.distortion,
        size_, R, translation, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY,
        alpha, size_);
    cv::initUndistortRectifyMap(left.K, left.distortion, R1, P1, size_, CV_32FC1,
                               left_map_x_, left_map_y_);
    cv::initUndistortRectifyMap(right.K, right.distortion, R2, P2, size_, CV_32FC1,
                               right_map_x_, right_map_y_);
  } else if (left.distortion_model == "equidistant") {
    cv::fisheye::stereoRectify(left.K, left.distortion, right.K, right.distortion,
        size_, R, translation, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY,
        size_, alpha, 1.0);
    cv::fisheye::initUndistortRectifyMap(left.K, left.distortion, R1, P1,
        size_, CV_32FC1, left_map_x_, left_map_y_);
    cv::fisheye::initUndistortRectifyMap(right.K, right.distortion, R2, P2,
        size_, CV_32FC1, right_map_x_, right_map_y_);
  } else {
    throw std::invalid_argument("Unsupported distortion model: " + left.distortion_model);
  }
  // The matching contract is horizontal, positive left disparity.
  if (!cv::checkRange(Q) || P2.at<double>(0, 3) >= -1e-6 ||
      std::abs(P2.at<double>(1, 3)) > 1e-6)
    throw std::invalid_argument("Stereo matcher needs horizontal stereo with right camera to the right");
  Q_ = Q;
  P1_ = P1;
  P2_ = P2;
  T_body_left_rect_ = T_body_left * inverseRectification(R1);
  T_body_right_rect_ = T_body_left * T_right_left.inverse() * inverseRectification(R2);

  // Require all bilinear source samples to be inside the original image.
  cv::Mat valid_source(size_, CV_32FC1, cv::Scalar(1.0));
  cv::Mat left_valid, right_valid;
  cv::remap(valid_source, left_valid, left_map_x_, left_map_y_, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::remap(valid_source, right_valid, right_map_x_, right_map_y_, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(0));
  left_mask_ = left_valid >= 0.999f;
  right_mask_ = right_valid >= 0.999f;
}

void StereoRectifier::rectify(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& left_rect, cv::Mat& right_rect) const {
  if (left.size() != size_ || right.size() != size_)
    throw std::invalid_argument("Image resolution differs from calibration; no implicit resizing is allowed");
  cv::remap(left, left_rect, left_map_x_, left_map_y_, cv::INTER_LINEAR);
  cv::remap(right, right_rect, right_map_x_, right_map_y_, cv::INTER_LINEAR);
}

FrameGeometry projectDisparity(const cv::Mat& disparity, const cv::Mat& left,
    const cv::Matx44d& Q, const cv::Mat& left_mask, const cv::Mat& right_mask,
    double min_depth, double max_depth, int pixel_stride) {
  if (disparity.type() != CV_32FC1 || disparity.size() != left.size() ||
      (left.type() != CV_8UC1 && left.type() != CV_8UC3) ||
      left_mask.type() != CV_8UC1 || right_mask.type() != CV_8UC1 ||
      disparity.size() != left_mask.size() || disparity.size() != right_mask.size() ||
      !std::isfinite(min_depth) || !std::isfinite(max_depth) ||
      min_depth <= 0 || max_depth <= min_depth || pixel_stride <= 0)
    throw std::invalid_argument("Invalid disparity, image, masks or depth limits");
  FrameGeometry result;
  result.depth = cv::Mat(disparity.size(), CV_32FC1,
                         cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  result.points.reserve(disparity.total() / pixel_stride / pixel_stride);
  for (int y = 0; y < disparity.rows; ++y) {
    for (int x = 0; x < disparity.cols; ++x) {
      const float d = disparity.at<float>(y, x);
      if (!std::isfinite(d) || d <= 0 || !left_mask.at<uint8_t>(y, x)) continue;
      const double xr = x - static_cast<double>(d);
      if (xr < 0 || xr > disparity.cols - 1) continue;
      if (!right_mask.at<uint8_t>(y, static_cast<int>(std::floor(xr))) ||
          !right_mask.at<uint8_t>(y, static_cast<int>(std::ceil(xr)))) continue;
      const cv::Vec4d homogeneous = Q * cv::Vec4d(x, y, d, 1);
      if (!std::isfinite(homogeneous[3]) || std::abs(homogeneous[3]) < 1e-12) continue;
      const Eigen::Vector3d point(homogeneous[0] / homogeneous[3],
          homogeneous[1] / homogeneous[3], homogeneous[2] / homogeneous[3]);
      if (!point.allFinite() || point.z() < min_depth || point.z() > max_depth) continue;
      result.depth.at<float>(y, x) = static_cast<float>(point.z());
      if (x % pixel_stride != 0 || y % pixel_stride != 0) continue;
      ColorPoint sample;
      sample.position = point;
      if (left.channels() == 1) {
        sample.r = sample.g = sample.b = left.at<uint8_t>(y, x);
      } else {
        const auto color = left.at<cv::Vec3b>(y, x);
        sample.r = color[2]; sample.g = color[1]; sample.b = color[0];
      }
      result.points.push_back(sample);
    }
  }
  return result;
}

PoseBuffer::PoseBuffer(size_t capacity) : capacity_(capacity) {
  if (!capacity) throw std::invalid_argument("Pose cache capacity must be positive");
}

void PoseBuffer::add(int64_t stamp_ns, const Eigen::Isometry3d& pose) {
  poses_[stamp_ns] = pose;
  while (poses_.size() > capacity_) poses_.erase(poses_.begin());
}

std::optional<Eigen::Isometry3d> PoseBuffer::find(int64_t stamp_ns, int64_t tolerance_ns) const {
  if (poses_.empty() || tolerance_ns < 0) return std::nullopt;
  auto after = poses_.lower_bound(stamp_ns);
  auto best = after;
  if (after != poses_.begin()) {
    auto before = std::prev(after);
    if (after == poses_.end() || stamp_ns - before->first <= after->first - stamp_ns)
      best = before;
  }
  if (best != poses_.end() && std::abs(best->first - stamp_ns) <= tolerance_ns)
    return best->second;
  return std::nullopt;
}

KeyframeSelector::KeyframeSelector(double translation, double rotation)
    : translation_(translation), rotation_(rotation) {
  if (!std::isfinite(translation) || !std::isfinite(rotation) || translation < 0 || rotation < 0)
    throw std::invalid_argument("Invalid keyframe thresholds");
}

bool KeyframeSelector::shouldInsert(const Eigen::Isometry3d& pose) const {
  if (!previous_) return true;
  const double translation = (pose.translation() - previous_->translation()).norm();
  const double rotation = Eigen::AngleAxisd(previous_->linear().transpose() * pose.linear()).angle();
  return translation >= translation_ || rotation >= rotation_;
}

VoxelMap::VoxelMap(double voxel_size, size_t max_voxels)
    : voxel_size_(voxel_size), max_voxels_(max_voxels) {
  if (!std::isfinite(voxel_size) || voxel_size <= 0 || max_voxels == 0)
    throw std::invalid_argument("Voxel size and map capacity must be positive");
}

size_t VoxelMap::Hash::operator()(const Key& key) const {
  size_t seed = std::hash<int64_t>{}(key.x);
  for (int64_t v : {key.y, key.z})
    seed ^= std::hash<int64_t>{}(v) + size_t{0x9e3779b9} + (seed << 6) + (seed >> 2);
  return seed;
}

size_t VoxelMap::integrate(const std::vector<ColorPoint>& points,
                         const Eigen::Isometry3d& T_world_camera) {
  size_t skipped = 0;
  for (const auto& point : points) {
    const Eigen::Vector3d world = T_world_camera * point.position;
    const Eigen::Array3d coordinates = (world / voxel_size_).array().floor();
    // Reject coordinates outside a safely representable integer range.
    if (!coordinates.isFinite().all() || (coordinates.abs() > 9e18).any()) continue;
    const Key key{static_cast<int64_t>(coordinates.x()),
                  static_cast<int64_t>(coordinates.y()), static_cast<int64_t>(coordinates.z())};
    auto it = voxels_.find(key);
    if (it == voxels_.end()) {
      if (voxels_.size() >= max_voxels_) { ++skipped; continue; }
      it = voxels_.emplace(key, Voxel{}).first;
    }
    auto& voxel = it->second;
    // Saturate the weight to allow bounded, gradual updates on long recordings.
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
    result.push_back({voxel.position, static_cast<uint8_t>(std::lround(voxel.color.x())),
        static_cast<uint8_t>(std::lround(voxel.color.y())),
        static_cast<uint8_t>(std::lround(voxel.color.z()))});
  }
  return result;
}

void savePointCloud(const std::string& path, const std::vector<ColorPoint>& points) {
  const auto extension = std::filesystem::path(path).extension().string();
  if (extension != ".pcd" && extension != ".ply")
    throw std::invalid_argument("Map output path must end in .pcd or .ply");
  std::ofstream output(path);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.imbue(std::locale::classic());
  output << std::setprecision(9);
  if (extension == ".pcd") {
    output << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
              "TYPE F F F U\nCOUNT 1 1 1 1\nWIDTH " << points.size()
           << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << points.size() << "\nDATA ascii\n";
  } else {
    output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
           << "\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  }
  for (const auto& point : points) {
    output << point.position.x() << ' ' << point.position.y() << ' ' << point.position.z() << ' ';
    if (extension == ".pcd") {
      const uint32_t rgb = (uint32_t(point.r) << 16) | (uint32_t(point.g) << 8) | point.b;
      output << rgb << '\n';
    } else {
      output << unsigned(point.r) << ' ' << unsigned(point.g) << ' ' << unsigned(point.b) << '\n';
    }
  }
  output.close();
}
}  // namespace msckf_vio::stereo

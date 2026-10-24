#include "stereo_recon/stereo/tsdf_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace stereo_recon {
namespace {
constexpr float kZero = 1e-6f;

ColorPoint MakeSurfacePoint(const Eigen::Vector3d& position,
                            const Eigen::Vector3f& color) {
  return {position,
          static_cast<uint8_t>(std::lround(std::clamp(color.x(), 0.0f, 255.0f))),
          static_cast<uint8_t>(std::lround(std::clamp(color.y(), 0.0f, 255.0f))),
          static_cast<uint8_t>(std::lround(std::clamp(color.z(), 0.0f, 255.0f)))};
}
}  // namespace

TsdfMap::TsdfMap(const TsdfParameters& parameters) : parameters_(parameters) {
  if (!std::isfinite(parameters.voxel_size) || parameters.voxel_size <= 0 ||
      !std::isfinite(parameters.truncation_distance) ||
      parameters.truncation_distance < parameters.voxel_size ||
      !std::isfinite(parameters.max_weight) || parameters.max_weight < 1 ||
      !std::isfinite(parameters.min_weight) || parameters.min_weight <= 0 ||
      parameters.min_weight > parameters.max_weight ||
      !parameters.max_voxels || parameters.integration_stride <= 0) {
    throw std::invalid_argument("Invalid TSDF voxel size, truncation, weights, capacity or stride");
  }
}

size_t TsdfMap::Hash::operator()(const Key& key) const {
  size_t seed = std::hash<int64_t>{}(key.x);
  for (int64_t value : {key.y, key.z}) {
    seed ^= std::hash<int64_t>{}(value) + size_t{0x9e3779b9} + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

std::optional<TsdfMap::Key> TsdfMap::GridKey(const Eigen::Vector3d& world) const {
  const Eigen::Array3d grid = (world / parameters_.voxel_size).array().floor();
  if (!grid.isFinite().all() || (grid.abs() > 9e18).any()) return std::nullopt;
  return Key{static_cast<int64_t>(grid.x()), static_cast<int64_t>(grid.y()),
             static_cast<int64_t>(grid.z())};
}

Eigen::Vector3d TsdfMap::Center(const Key& key) const {
  return parameters_.voxel_size *
         Eigen::Vector3d(static_cast<double>(key.x) + 0.5,
                         static_cast<double>(key.y) + 0.5,
                         static_cast<double>(key.z) + 0.5);
}

// Amanatides-Woo traversal in world voxel coordinates. Parameter t is camera
// optical Z, so the allocation band has the same units as the depth residual.
bool TsdfMap::AllocateRay(const Eigen::Vector3d& origin,
                          const Eigen::Vector3d& direction,
                          double begin, double end) {
  const auto first = GridKey(origin + begin * direction);
  if (!first) return false;
  Key key = *first;
  std::array<int64_t*, 3> cells{&key.x, &key.y, &key.z};
  std::array<int, 3> step{};
  std::array<double, 3> next{}, delta{};
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(direction[axis]) < 1e-15) {
      next[axis] = delta[axis] = std::numeric_limits<double>::infinity();
      continue;
    }
    step[axis] = direction[axis] > 0 ? 1 : -1;
    const double boundary =
        (static_cast<double>(*cells[axis]) + (step[axis] > 0 ? 1.0 : 0.0)) *
        parameters_.voxel_size;
    next[axis] = (boundary - origin[axis]) / direction[axis];
    delta[axis] = parameters_.voxel_size / std::abs(direction[axis]);
  }
  for (;;) {
    if (voxels_.find(key) == voxels_.end()) {
      if (voxels_.size() >= parameters_.max_voxels) return true;
      voxels_.emplace(key, Voxel{});
    }
    const double next_t = std::min({next[0], next[1], next[2]});
    if (!std::isfinite(next_t) || next_t > end) break;
    // Advance all tied axes, avoiding cells touched only at a ray corner.
    for (int axis = 0; axis < 3; ++axis) {
      if (next[axis] <= next_t + 1e-12) {
        *cells[axis] += step[axis];
        next[axis] += delta[axis];
      }
    }
  }
  return false;
}

bool TsdfMap::Integrate(const cv::Mat& depth, const cv::Mat& color,
                        const cv::Matx33d& intrinsics,
                        const Eigen::Isometry3d& world_from_camera) {
  const Eigen::Matrix3d rotation = world_from_camera.linear();
  if (depth.empty() || depth.type() != CV_32FC1 || depth.size() != color.size() ||
      (color.type() != CV_8UC1 && color.type() != CV_8UC3) ||
      !cv::checkRange(cv::Mat(intrinsics)) || intrinsics(0, 0) <= 0 ||
      intrinsics(1, 1) <= 0 || std::abs(intrinsics(0, 1)) > 1e-9 ||
      std::abs(intrinsics(1, 0)) > 1e-9 ||
      std::abs(intrinsics(2, 0)) > 1e-9 || std::abs(intrinsics(2, 1)) > 1e-9 ||
      std::abs(intrinsics(2, 2) - 1.0) > 1e-9 ||
      !world_from_camera.matrix().allFinite() ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-3 ||
      std::abs(rotation.determinant() - 1.0) > 1e-3 ||
      (world_from_camera.matrix().row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9) {
    throw std::invalid_argument("Invalid TSDF depth, color, pinhole intrinsics or rigid pose");
  }
  const double fx = intrinsics(0, 0), fy = intrinsics(1, 1);
  const double cx = intrinsics(0, 2), cy = intrinsics(1, 2);
  const double mu = parameters_.truncation_distance;
  bool capacity_reached = false;
  // Allocation samples rays; the following voxel-centric pass always samples
  // the full-resolution depth and gives at most one observation per voxel.
  for (int y = 0; y < depth.rows && !capacity_reached; y += parameters_.integration_stride) {
    for (int x = 0; x < depth.cols && !capacity_reached; x += parameters_.integration_stride) {
      const double z = depth.at<float>(y, x);
      if (!std::isfinite(z) || z <= 0) continue;
      const Eigen::Vector3d ray((x - cx) / fx, (y - cy) / fy, 1.0);
      capacity_reached = AllocateRay(world_from_camera.translation(), rotation * ray,
                                     std::max(1e-6, z - mu), z + mu);
    }
  }
  const Eigen::Isometry3d camera_from_world = world_from_camera.inverse();
  for (auto it = voxels_.begin(); it != voxels_.end();) {
    auto& voxel = it->second;
    const Eigen::Vector3d camera = camera_from_world * Center(it->first);
    if (camera.z() > 0) {
      const double u = fx * camera.x() / camera.z() + cx;
      const double v = fy * camera.y() / camera.z() + cy;
      if (u >= -0.5 && u < depth.cols - 0.5 && v >= -0.5 && v < depth.rows - 0.5) {
        const int x = static_cast<int>(std::floor(u + 0.5));
        const int y = static_cast<int>(std::floor(v + 0.5));
        const double z = depth.at<float>(y, x);
        const double sdf = z - camera.z();
        if (std::isfinite(z) && z > 0 && sdf >= -mu) {
          const float distance = static_cast<float>(std::min(1.0, sdf / mu));
          const float old_weight = voxel.sdf.weight;
          voxel.sdf.distance = (old_weight * voxel.sdf.distance + distance) /
                               (old_weight + 1.0f);
          voxel.sdf.weight = std::min(old_weight + 1.0f, parameters_.max_weight);
          // Surface color only: distant free-space updates must not repaint
          // an unrelated surface. Crossing endpoints lie within one voxel.
          if (std::abs(sdf) <= parameters_.voxel_size) {
            Eigen::Vector3f rgb;
            if (color.channels() == 1) {
              rgb.setConstant(color.at<uint8_t>(y, x));
            } else {
              const auto bgr = color.at<cv::Vec3b>(y, x);
              rgb = Eigen::Vector3f(bgr[2], bgr[1], bgr[0]);
            }
            voxel.color = (voxel.color_weight * voxel.color + rgb) /
                          (voxel.color_weight + 1.0f);
            voxel.color_weight = std::min(voxel.color_weight + 1.0f,
                                           parameters_.max_weight);
          }
        }
      }
    }
    // Failed allocations at occlusion/frustum borders must not consume capacity.
    if (voxel.sdf.weight == 0) it = voxels_.erase(it);
    else ++it;
  }
  return capacity_reached;
}

std::optional<TsdfSample> TsdfMap::Sample(const Eigen::Vector3d& world) const {
  const auto key = GridKey(world);
  if (!key) return std::nullopt;
  const auto it = voxels_.find(*key);
  if (it == voxels_.end() || it->second.sdf.weight == 0) return std::nullopt;
  return it->second.sdf;
}

std::vector<ColorPoint> TsdfMap::points() const {
  std::vector<ColorPoint> result;
  result.reserve(voxels_.size() / 3);
  const auto observed = [this](const Key& key) -> const Voxel* {
    const auto it = voxels_.find(key);
    return it != voxels_.end() && it->second.sdf.weight >= parameters_.min_weight
               ? &it->second : nullptr;
  };
  for (const auto& entry : voxels_) {
    const auto& key = entry.first;
    const auto& voxel = entry.second;
    if (voxel.sdf.weight < parameters_.min_weight) continue;
    const Eigen::Vector3d center = Center(key);
    const std::array<Key, 3> positive{{{key.x + 1, key.y, key.z},
                                     {key.x, key.y + 1, key.z},
                                     {key.x, key.y, key.z + 1}}};
    if (std::abs(voxel.sdf.distance) <= kZero) {
      bool positive_seen = false, negative_seen = false;
      const std::array<Key, 6> neighbours{{positive[0], positive[1], positive[2],
          {key.x - 1, key.y, key.z}, {key.x, key.y - 1, key.z},
          {key.x, key.y, key.z - 1}}};
      for (const auto& neighbour : neighbours) {
        const auto* other = observed(neighbour);
        if (other) {
          positive_seen |= other->sdf.distance > kZero;
          negative_seen |= other->sdf.distance < -kZero;
        }
      }
      if (positive_seen && negative_seen) result.push_back(MakeSurfacePoint(center, voxel.color));
      continue;
    }
    for (const auto& neighbour : positive) {
      const auto* other = observed(neighbour);
      if (!other || std::abs(other->sdf.distance) <= kZero ||
          (voxel.sdf.distance > 0) == (other->sdf.distance > 0)) continue;
      const double t = static_cast<double>(voxel.sdf.distance) /
                       (voxel.sdf.distance - other->sdf.distance);
      const Eigen::Vector3d position = center + t * (Center(neighbour) - center);
      const float color_t = static_cast<float>(t);
      Eigen::Vector3f rgb = (1.0f - color_t) * voxel.color + color_t * other->color;
      if (voxel.color_weight == 0) rgb = other->color;
      if (other->color_weight == 0) rgb = voxel.color;
      result.push_back(MakeSurfacePoint(position, rgb));
    }
  }
  return result;
}

}  // namespace stereo_recon

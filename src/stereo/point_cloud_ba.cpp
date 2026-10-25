#include "stereo_recon/stereo/point_cloud_ba.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "Eigen/Eigenvalues"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace stereo_recon {
namespace {

constexpr double kRadiansPerDegree = 0.017453292519943295;

struct PoseBlock {
  // Ceres quaternion ordering is w,x,y,z, followed by translation.
  std::array<double, 7> values;
  explicit PoseBlock(const Eigen::Isometry3d& pose) {
    const Eigen::Quaterniond q(pose.linear());
    values = {q.w(),
              q.x(),
              q.y(),
              q.z(),
              pose.translation().x(),
              pose.translation().y(),
              pose.translation().z()};
  }
  Eigen::Isometry3d Pose() const {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() =
        Eigen::Quaterniond(values[0], values[1], values[2], values[3])
            .normalized()
            .toRotationMatrix();
    pose.translation() = Eigen::Vector3d(values[4], values[5], values[6]);
    return pose;
  }
};

struct SampleReference {
  size_t frame;
  const BaPoint* point;
  Eigen::Vector3d world;
};

struct PlaneCluster {
  size_t frame;
  Eigen::Matrix4d square_root;
};

struct PlaneFeature {
  std::array<double, 4> values;
  std::vector<PlaneCluster> clusters;
};

struct ClusterPlaneResidual {
  Eigen::Matrix4d square_root;
  template <typename T>
  bool operator()(const T* pose, const T* plane, T* residual) const {
    const T inverse_q[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
    T local_plane[4];
    ceres::QuaternionRotatePoint(inverse_q, plane, local_plane);
    local_plane[3] =
        plane[0] * pose[4] + plane[1] * pose[5] + plane[2] * pose[6] + plane[3];
    for (int row = 0; row < 4; ++row) {
      residual[row] = T(0);
      for (int col = 0; col < 4; ++col) {
        residual[row] += T(square_root(row, col)) * local_plane[col];
      }
    }
    return true;
  }
};

struct RelativePoseResidual {
  Eigen::Quaterniond rotation;
  Eigen::Vector3d translation;
  double translation_sigma;
  double rotation_sigma;

  template <typename T>
  bool operator()(const T* first, const T* second, T* residual) const {
    const T inverse_first[4] = {first[0], -first[1], -first[2], -first[3]};
    T relative_q[4];
    ceres::QuaternionProduct(inverse_first, second, relative_q);
    const T inverse_measured[4] = {T(rotation.w()), T(-rotation.x()),
                                   T(-rotation.y()), T(-rotation.z())};
    T error_q[4];
    ceres::QuaternionProduct(inverse_measured, relative_q, error_q);
    // q and -q encode the same orientation. Use the shortest small-angle error.
    const T sign = error_q[0] < T(0) ? T(-1) : T(1);
    const T delta[3] = {second[4] - first[4], second[5] - first[5],
                        second[6] - first[6]};
    T local_delta[3];
    ceres::QuaternionRotatePoint(inverse_first, delta, local_delta);
    for (int i = 0; i < 3; ++i) {
      residual[i] = (local_delta[i] - T(translation[i])) / T(translation_sigma);
      residual[i + 3] = T(2) * sign * error_q[i + 1] / T(rotation_sigma);
    }
    return true;
  }
};

double RotationAngle(const Eigen::Isometry3d& pose) {
  return std::abs(Eigen::AngleAxisd(pose.linear()).angle());
}

bool IsRigid(const Eigen::Isometry3d& pose) {
  return pose.matrix().allFinite() &&
         (pose.linear().transpose() * pose.linear() -
          Eigen::Matrix3d::Identity())
                 .norm() < 1e-3 &&
         std::abs(pose.linear().determinant() - 1.0) < 1e-3;
}

}  // namespace

PointCloudBundleAdjuster::PointCloudBundleAdjuster(
    PointCloudBaParameters parameters)
    : parameters_(std::move(parameters)) {
  const auto& p = parameters_;
  if (p.window_size < 3 || p.window_size > 100 || p.sample_stride < 1 ||
      p.max_samples_per_frame < 24 || p.max_samples_per_cluster < 4 ||
      p.min_points_per_plane < 12 || p.min_frames_per_plane < 2 ||
      p.min_frames_per_plane > p.window_size || p.min_planes < 3 ||
      p.max_planes < p.min_planes || p.max_voxel_depth < 0 ||
      p.max_voxel_depth > 8 || p.max_iterations < 1 || p.num_threads < 1) {
    throw std::invalid_argument("Invalid point-cloud BA counts");
  }
  for (double value :
       {p.root_voxel_size, p.plane_eigenvalue_ratio, p.max_plane_rms,
        p.min_plane_spread, p.min_normal_eigenvalue, p.disparity_sigma,
        p.depth_edge_absolute, p.depth_edge_relative, p.noise_floor,
        p.huber_delta, p.vio_translation_sigma, p.vio_rotation_sigma_deg,
        p.max_translation_update, p.max_rotation_update_deg,
        p.max_vio_translation, p.max_vio_rotation_deg, p.max_solver_time_sec}) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument("Point-cloud BA scales must be positive");
    }
  }
  if (p.plane_eigenvalue_ratio >= 1 || p.min_normal_eigenvalue >= 1.0 / 3.0) {
    throw std::invalid_argument("Invalid BA geometry thresholds");
  }
}

void PointCloudBundleAdjuster::PrepareKeyframe(MappingKeyframe* frame) const {
  if (frame == nullptr || frame->depth.type() != CV_32FC1 ||
      frame->depth.empty() || frame->color.size() != frame->depth.size() ||
      (frame->color.type() != CV_8UC1 && frame->color.type() != CV_8UC3) ||
      !IsRigid(frame->vio_pose) || !IsRigid(frame->map_pose) ||
      !cv::checkRange(cv::Mat(frame->intrinsics)) ||
      frame->intrinsics(0, 0) <= 0 || frame->intrinsics(1, 1) <= 0 ||
      !std::isfinite(frame->baseline) || frame->baseline <= 0) {
    throw std::invalid_argument("Invalid BA keyframe geometry");
  }
  const auto& p = parameters_;
  const double fx = frame->intrinsics(0, 0), fy = frame->intrinsics(1, 1);
  const double cx = frame->intrinsics(0, 2), cy = frame->intrinsics(1, 2);
  int stride = p.sample_stride;
  while (static_cast<int64_t>((frame->depth.cols + stride - 1) / stride) *
             ((frame->depth.rows + stride - 1) / stride) >
         p.max_samples_per_frame) {
    ++stride;
  }
  frame->samples.clear();
  for (int y = 1; y < frame->depth.rows - 1; y += stride) {
    for (int x = 1; x < frame->depth.cols - 1; x += stride) {
      const double z = frame->depth.at<float>(y, x);
      if (!std::isfinite(z) || z <= 0) continue;
      bool edge = false;
      for (const auto& offset :
           std::array<cv::Point, 4>{cv::Point(-1, 0), cv::Point(1, 0),
                                    cv::Point(0, -1), cv::Point(0, 1)}) {
        const double adjacent =
            frame->depth.at<float>(y + offset.y, x + offset.x);
        if (!std::isfinite(adjacent) || adjacent <= 0 ||
            std::abs(adjacent - z) >
                p.depth_edge_absolute + p.depth_edge_relative * z) {
          edge = true;
          break;
        }
      }
      if (edge) continue;
      const Eigen::Vector3d point((x - cx) * z / fx, (y - cy) * z / fy, z);
      const double disparity = fx * frame->baseline / z;
      Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
      jacobian(0, 0) = z / fx;
      jacobian(1, 1) = z / fy;
      jacobian.col(2) = -point / disparity;
      Eigen::Matrix3d measurement = Eigen::Matrix3d::Identity() * 0.25;
      measurement(2, 2) = p.disparity_sigma * p.disparity_sigma;
      const Eigen::Matrix3d covariance =
          jacobian * measurement * jacobian.transpose() +
          Eigen::Matrix3d::Identity() * p.noise_floor * p.noise_floor;
      if (point.allFinite() && covariance.allFinite()) {
        frame->samples.push_back({point, covariance});
      }
    }
  }
}

BaSummary PointCloudBundleAdjuster::Optimize(
    const std::vector<MappingKeyframe*>& frames) const {
  BaSummary result;
  const auto start = std::chrono::steady_clock::now();
  const auto finish = [&](const std::string& reason) {
    result.reason = reason;
    result.time_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    return result;
  };
  const auto& p = parameters_;
  if (frames.size() < static_cast<size_t>(p.min_frames_per_plane)) {
    return finish("waiting_for_window");
  }
  if (frames.size() > static_cast<size_t>(p.window_size)) {
    throw std::invalid_argument("BA window exceeds configured bound");
  }
  std::vector<PoseBlock> poses;
  std::map<std::array<int64_t, 3>, std::vector<SampleReference>> roots;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (!frames[i] || !IsRigid(frames[i]->map_pose) ||
        !IsRigid(frames[i]->vio_pose)) {
      throw std::invalid_argument("Invalid BA input pose");
    }
    poses.emplace_back(frames[i]->map_pose);
    for (const auto& point : frames[i]->samples) {
      const Eigen::Vector3d world = frames[i]->map_pose * point.position;
      if (!world.allFinite() || world.cwiseAbs().maxCoeff() > 1e8) continue;
      std::array<int64_t, 3> key;
      for (int axis = 0; axis < 3; ++axis) {
        key[axis] =
            static_cast<int64_t>(std::floor(world[axis] / p.root_voxel_size));
      }
      roots[key].push_back({i, &point, world});
    }
  }

  std::vector<PlaneFeature> planes;
  // Deterministic octree traversal makes archive replay reproducible.
  std::function<void(const std::vector<SampleReference>&,
                     const Eigen::Vector3d&, double, int)>
      partition;
  partition = [&](const std::vector<SampleReference>& points,
                  const Eigen::Vector3d& origin, double size, int level) {
    if (planes.size() >= static_cast<size_t>(p.max_planes) ||
        points.size() < static_cast<size_t>(p.min_points_per_plane))
      return;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto& ref : points) center += ref.world;
    center /= points.size();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& ref : points) {
      const Eigen::Vector3d delta = ref.world - center;
      covariance.noalias() += delta * delta.transpose();
    }
    covariance /= points.size();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
    const auto values = eigen.eigenvalues();
    if (eigen.info() == Eigen::Success &&
        values[1] > p.min_plane_spread * p.min_plane_spread &&
        values[0] < p.plane_eigenvalue_ratio * values[1] &&
        values[0] < p.max_plane_rms * p.max_plane_rms) {
      const Eigen::Vector3d normal = eigen.eigenvectors().col(0);
      std::map<size_t, std::vector<const BaPoint*>> groups;
      for (const auto& ref : points) groups[ref.frame].push_back(ref.point);
      for (auto it = groups.begin(); it != groups.end();) {
        if (it->second.size() < 4)
          it = groups.erase(it);
        else
          ++it;
      }
      if (groups.size() < static_cast<size_t>(p.min_frames_per_plane)) return;
      PlaneFeature plane{
          {normal.x(), normal.y(), normal.z(), -normal.dot(center)}, {}};
      for (const auto& [frame_index, samples] : groups) {
        const size_t count = std::min(
            samples.size(), static_cast<size_t>(p.max_samples_per_cluster));
        const Eigen::Vector3d local_normal =
            frames[frame_index]->map_pose.linear().transpose() * normal;
        Eigen::Matrix4d moment = Eigen::Matrix4d::Zero();
        for (size_t k = 0; k < count; ++k) {
          const auto& point = *samples[k * samples.size() / count];
          const double variance =
              local_normal.dot(point.covariance * local_normal);
          const Eigen::Vector4d homogeneous(
              point.position.x(), point.position.y(), point.position.z(), 1.0);
          // Equal total contribution per frame and feature. Weights stay fixed
          // throughout this solve; normal-dependent whitening is not rederived.
          moment.noalias() += homogeneous * homogeneous.transpose() /
                              (variance * count * groups.size());
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> factor(moment);
        if (factor.info() != Eigen::Success) return;
        const Eigen::Matrix4d square_root =
            factor.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
            factor.eigenvectors().transpose();
        plane.clusters.push_back({frame_index, square_root});
      }
      planes.push_back(std::move(plane));
      return;
    }
    if (level >= p.max_voxel_depth) return;
    const double half = size * 0.5;
    std::array<std::vector<SampleReference>, 8> children;
    for (const auto& ref : points) {
      int index = 0;
      for (int axis = 0; axis < 3; ++axis) {
        if (ref.world[axis] >= origin[axis] + half) index |= 1 << axis;
      }
      children[index].push_back(ref);
    }
    for (int index = 0; index < 8; ++index) {
      Eigen::Vector3d child_origin = origin;
      for (int axis = 0; axis < 3; ++axis) {
        if (index & (1 << axis)) child_origin[axis] += half;
      }
      partition(children[index], child_origin, half, level + 1);
    }
  };
  for (const auto& [key, points] : roots) {
    partition(points,
              Eigen::Vector3d(key[0], key[1], key[2]) * p.root_voxel_size,
              p.root_voxel_size, 0);
  }
  result.planes = planes.size();
  if (planes.size() < static_cast<size_t>(p.min_planes)) {
    return finish("insufficient_shared_planes");
  }
  Eigen::Matrix3d normal_information = Eigen::Matrix3d::Zero();
  std::vector<int> support(frames.size(), 0);
  std::vector<size_t> parent(frames.size());
  std::iota(parent.begin(), parent.end(), 0);
  const auto root = [&](size_t i) {
    while (parent[i] != i) i = parent[i];
    return i;
  };
  for (const auto& plane : planes) {
    const Eigen::Vector3d n(plane.values[0], plane.values[1], plane.values[2]);
    normal_information.noalias() += n * n.transpose();
    for (const auto& cluster : plane.clusters) {
      ++support[cluster.frame];
      parent[root(cluster.frame)] = root(plane.clusters.front().frame);
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> normals(normal_information /
                                                         planes.size());
  result.normal_eigenvalue = normals.eigenvalues()[0];
  if (normals.info() != Eigen::Success ||
      result.normal_eigenvalue < p.min_normal_eigenvalue) {
    return finish("degenerate_plane_directions");
  }
  for (size_t i = 0; i < frames.size(); ++i) {
    if (support[i] < 3 || root(i) != root(0)) {
      return finish("disconnected_or_unsupported_frame");
    }
  }

  ceres::Problem problem;
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (auto& pose : poses) {
    problem.AddParameterBlock(pose.values.data(), 7,
                              new ceres::ProductParameterization(
                                  new ceres::QuaternionParameterization,
                                  new ceres::IdentityParameterization(3)));
    ordering->AddElementToGroup(pose.values.data(), 1);
  }
  problem.SetParameterBlockConstant(poses.front().values.data());
  std::vector<ceres::ResidualBlockId> geometry_blocks;
  for (auto& plane : planes) {
    problem.AddParameterBlock(
        plane.values.data(), 4,
        new ceres::ProductParameterization(
            new ceres::HomogeneousVectorParameterization(3),
            new ceres::IdentityParameterization(1)));
    ordering->AddElementToGroup(plane.values.data(), 0);
    for (const auto& cluster : plane.clusters) {
      auto* cost =
          new ceres::AutoDiffCostFunction<ClusterPlaneResidual, 4, 7, 4>(
              new ClusterPlaneResidual{cluster.square_root});
      geometry_blocks.push_back(problem.AddResidualBlock(
          cost, new ceres::HuberLoss(p.huber_delta),
          poses[cluster.frame].values.data(), plane.values.data()));
    }
  }
  result.clusters = geometry_blocks.size();
  for (size_t i = 1; i < frames.size(); ++i) {
    const Eigen::Isometry3d measured =
        frames[i - 1]->vio_pose.inverse() * frames[i]->vio_pose;
    auto* cost = new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 7, 7>(
        new RelativePoseResidual{Eigen::Quaterniond(measured.linear()),
                                 measured.translation(),
                                 p.vio_translation_sigma,
                                 p.vio_rotation_sigma_deg * kRadiansPerDegree});
    problem.AddResidualBlock(cost, nullptr, poses[i - 1].values.data(),
                             poses[i].values.data());
  }
  ceres::Problem::EvaluateOptions evaluation;
  evaluation.residual_blocks = geometry_blocks;
  if (!problem.Evaluate(evaluation, &result.initial_cost, nullptr, nullptr,
                        nullptr))
    return finish("invalid_initial_cost");
  ceres::Solver::Options options;
  options.max_num_iterations = p.max_iterations;
  options.max_solver_time_in_seconds = p.max_solver_time_sec;
  options.num_threads = p.num_threads;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable() ||
      !problem.Evaluate(evaluation, &result.final_cost, nullptr, nullptr,
                        nullptr) ||
      !std::isfinite(result.final_cost) ||
      result.final_cost > result.initial_cost + 1e-8 ||
      summary.final_cost > summary.initial_cost + 1e-8) {
    return finish("solver_or_cost_rejected");
  }
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto pose = poses[i].Pose();
    const Eigen::Isometry3d update = frames[i]->map_pose.inverse() * pose;
    const Eigen::Isometry3d total = frames[i]->vio_pose.inverse() * pose;
    if (!IsRigid(pose) ||
        update.translation().norm() > p.max_translation_update ||
        RotationAngle(update) > p.max_rotation_update_deg * kRadiansPerDegree ||
        total.translation().norm() > p.max_vio_translation ||
        RotationAngle(total) > p.max_vio_rotation_deg * kRadiansPerDegree) {
      return finish("pose_update_rejected");
    }
  }
  for (size_t i = 0; i < frames.size(); ++i)
    frames[i]->map_pose = poses[i].Pose();
  result.accepted = true;
  return finish("accepted");
}

}  // namespace stereo_recon

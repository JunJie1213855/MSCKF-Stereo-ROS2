#include "stereo_recon/stereo/submap_mapper.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "opencv2/imgcodecs.hpp"
#include "yaml-cpp/yaml.h"

namespace stereo_recon {
namespace {

template <typename Visitor>
void VisitBaFields(PointCloudBaParameters* parameters, Visitor visitor) {
  auto& p = *parameters;
  for (const auto& [name, field] : std::vector<std::pair<std::string, int*>>{
           {"window_size", &p.window_size},
           {"sample_stride", &p.sample_stride},
           {"max_samples_per_frame", &p.max_samples_per_frame},
           {"max_samples_per_cluster", &p.max_samples_per_cluster},
           {"min_points_per_plane", &p.min_points_per_plane},
           {"min_frames_per_plane", &p.min_frames_per_plane},
           {"min_planes", &p.min_planes},
           {"max_planes", &p.max_planes},
           {"max_voxel_depth", &p.max_voxel_depth},
           {"max_iterations", &p.max_iterations},
           {"num_threads", &p.num_threads}}) {
    visitor(name, field);
  }
  for (const auto& [name, field] : std::vector<std::pair<std::string, double*>>{
           {"root_voxel_size", &p.root_voxel_size},
           {"plane_eigenvalue_ratio", &p.plane_eigenvalue_ratio},
           {"max_plane_rms", &p.max_plane_rms},
           {"min_plane_spread", &p.min_plane_spread},
           {"min_normal_eigenvalue", &p.min_normal_eigenvalue},
           {"disparity_sigma", &p.disparity_sigma},
           {"depth_edge_absolute", &p.depth_edge_absolute},
           {"depth_edge_relative", &p.depth_edge_relative},
           {"noise_floor", &p.noise_floor},
           {"huber_delta", &p.huber_delta},
           {"vio_translation_sigma", &p.vio_translation_sigma},
           {"vio_rotation_sigma_deg", &p.vio_rotation_sigma_deg},
           {"max_translation_update", &p.max_translation_update},
           {"max_rotation_update_deg", &p.max_rotation_update_deg},
           {"max_vio_translation", &p.max_vio_translation},
           {"max_vio_rotation_deg", &p.max_vio_rotation_deg},
           {"max_solver_time_sec", &p.max_solver_time_sec}}) {
    visitor(name, field);
  }
}

template <typename Visitor>
void VisitMappingFields(SubmapMapperParameters* parameters, Visitor visitor) {
  auto& p = *parameters;
  VisitBaFields(&p.ba, [&](const std::string& name, auto* value) {
    visitor("ba", name, value);
  });
  visitor("ba", "enabled", &p.enable_optimization);
  visitor("ba", "submap_keyframes", &p.submap_keyframes);
  visitor("ba", "overlap_keyframes", &p.overlap_keyframes);
  visitor("ba", "optimize_every_n_keyframes", &p.optimize_every_n_keyframes);
  visitor("ba", "max_trajectory_poses", &p.max_trajectory_poses);
  visitor("", "voxel_size", &p.tsdf.voxel_size);
  // yaml-cpp supports size_t directly, retaining the configured capacity.
  visitor("", "max_voxels", &p.tsdf.max_voxels);
  visitor("tsdf", "truncation_distance", &p.tsdf.truncation_distance);
  visitor("tsdf", "max_weight", &p.tsdf.max_weight);
  visitor("tsdf", "min_weight", &p.tsdf.min_weight);
  visitor("tsdf", "integration_stride", &p.tsdf.integration_stride);
}

void WriteMappingParameters(const std::string& path,
                            SubmapMapperParameters parameters) {
  YAML::Node root;
  auto config = root["stereo_reconstruction"]["ros__parameters"];
  VisitMappingFields(&parameters, [&](const std::string& section,
                                      const std::string& name, auto* value) {
    if (section.empty())
      config[name] = *value;
    else
      config[section][name] = *value;
  });
  std::ofstream stream(path);
  stream << root;
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write mapping parameters");
}

std::string FrameName(uint64_t id) {
  std::ostringstream stream;
  stream << "frame_" << std::setfill('0') << std::setw(10) << id;
  return stream.str();
}

cv::Mat PoseMatrix(const Eigen::Isometry3d& pose) {
  cv::Mat result(4, 4, CV_64F);
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      result.at<double>(row, col) = pose.matrix()(row, col);
    }
  }
  return result;
}

}  // namespace

SubmapMapper::SubmapMapper(SubmapMapperParameters parameters)
    : parameters_(std::move(parameters)), optimizer_(parameters_.ba) {
  if (parameters_.submap_keyframes < parameters_.ba.window_size ||
      parameters_.submap_keyframes > 200 || parameters_.overlap_keyframes < 1 ||
      parameters_.overlap_keyframes >= parameters_.ba.window_size ||
      parameters_.optimize_every_n_keyframes < 1 ||
      parameters_.max_trajectory_poses < parameters_.submap_keyframes) {
    throw std::invalid_argument("Invalid BA submap/trajectory bounds");
  }
  if (!parameters_.archive_directory.empty()) {
    const auto suffix = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const auto directory =
        std::filesystem::path(parameters_.archive_directory) /
        ("session_" + std::to_string(suffix));
    if (!std::filesystem::create_directories(directory)) {
      throw std::runtime_error("Cannot create unique BA archive directory");
    }
    archive_path_ = directory.string();
    WriteMappingParameters((directory / "mapping_parameters.yaml").string(),
                           parameters_);
  }
  Clear();
}

void SubmapMapper::Clear() {
  ++sequence_;
  active_.clear();
  trajectory_.clear();
  active_tsdf_ = std::make_unique<TsdfMap>(parameters_.tsdf);
  frozen_surface_ = std::make_unique<VoxelMap>(parameters_.tsdf.voxel_size,
                                               parameters_.tsdf.max_voxels);
  map_from_vio_.setIdentity();
  frozen_submaps_ = 0;
  capacity_reached_ = false;
  rebuild_required_ = false;
  // IDs remain monotonic across clears so recorded files are never overwritten.
}

std::optional<MappingSnapshot> SubmapMapper::Process(
    MappingKeyframe frame, std::vector<ColorPoint> current_cloud,
    const std::function<bool()>& cancelled) {
  const auto start = std::chrono::steady_clock::now();
  if (cancelled()) return std::nullopt;
  if (!active_.empty() && frame.stamp_ns <= active_.back().stamp_ns) {
    throw std::invalid_argument(
        "Mapping keyframes must have increasing stamps");
  }
  frame.id = next_id_++;
  frame.sequence = sequence_;
  frame.map_pose = map_from_vio_ * frame.vio_pose;
  optimizer_.PrepareKeyframe(&frame);
  Archive(frame);
  if (cancelled()) return std::nullopt;

  if (active_.size() >= static_cast<size_t>(parameters_.submap_keyframes)) {
    const size_t freeze_count = active_.size() - parameters_.overlap_keyframes;
    TsdfMap freezing(parameters_.tsdf);
    for (size_t i = 0; i < freeze_count; ++i) {
      if (cancelled()) return std::nullopt;
      const auto& old = active_[i];
      capacity_reached_ |= freezing.Integrate(old.depth, old.color,
                                              old.intrinsics, old.map_pose);
    }
    capacity_reached_ |=
        frozen_surface_->Integrate(freezing.points(),
                                   Eigen::Isometry3d::Identity()) > 0;
    active_.erase(active_.begin(), active_.begin() + freeze_count);
    ++frozen_submaps_;
    rebuild_required_ = true;
  }
  active_.push_back(std::move(frame));
  auto& latest = active_.back();
  trajectory_.push_back({latest.id, latest.stamp_ns, latest.map_pose});
  while (trajectory_.size() >
         static_cast<size_t>(parameters_.max_trajectory_poses)) {
    trajectory_.pop_front();
  }

  BaSummary optimization;
  if (parameters_.enable_optimization &&
      latest.id % parameters_.optimize_every_n_keyframes == 0) {
    std::vector<MappingKeyframe*> window;
    const size_t begin =
        active_.size() > static_cast<size_t>(parameters_.ba.window_size)
            ? active_.size() - parameters_.ba.window_size
            : 0;
    std::vector<Eigen::Isometry3d> before;
    for (size_t i = begin; i < active_.size(); ++i) {
      window.push_back(&active_[i]);
      before.push_back(active_[i].map_pose);
    }
    optimization = optimizer_.Optimize(window);
    if (optimization.accepted) {
      for (size_t i = 0; i < window.size(); ++i) {
        rebuild_required_ |=
            (window[i]->map_pose.matrix() - before[i].matrix()).norm() > 1e-8;
      }
    }
  } else {
    optimization.reason = parameters_.enable_optimization
                              ? "trigger_interval"
                              : "optimization_disabled";
  }
  if (cancelled()) return std::nullopt;

  if (rebuild_required_) {
    auto replacement = std::make_unique<TsdfMap>(parameters_.tsdf);
    for (const auto& keyframe : active_) {
      if (cancelled()) return std::nullopt;
      capacity_reached_ |=
          replacement->Integrate(keyframe.depth, keyframe.color,
                                 keyframe.intrinsics, keyframe.map_pose);
    }
    active_tsdf_ = std::move(replacement);
    rebuild_required_ = false;
  } else {
    capacity_reached_ |= active_tsdf_->Integrate(
        latest.depth, latest.color, latest.intrinsics, latest.map_pose);
  }
  if (cancelled()) return std::nullopt;
  map_from_vio_ = latest.map_pose * latest.vio_pose.inverse();
  for (const auto& keyframe : active_) {
    const auto it = std::lower_bound(
        trajectory_.begin(), trajectory_.end(), keyframe.id,
        [](const MappingPose& pose, uint64_t id) { return pose.id < id; });
    if (it != trajectory_.end() && it->id == keyframe.id) {
      it->pose = keyframe.map_pose;
    }
  }

  // Bound the published union too. TSDF extraction may emit several crossings
  // per grid cell; merge only those surface samples at the configured scale.
  VoxelMap surface(parameters_.tsdf.voxel_size, parameters_.tsdf.max_voxels);
  capacity_reached_ |= surface.Integrate(active_tsdf_->points(),
                                         Eigen::Isometry3d::Identity()) > 0;
  capacity_reached_ |= surface.Integrate(frozen_surface_->points(),
                                         Eigen::Isometry3d::Identity()) > 0;
  MappingSnapshot snapshot;
  snapshot.stamp_ns = latest.stamp_ns;
  snapshot.map_from_vio = map_from_vio_;
  snapshot.camera_pose = latest.map_pose;
  snapshot.points = surface.points();
  for (auto& point : current_cloud) {
    point.position = latest.map_pose * point.position;
  }
  snapshot.current_points = std::move(current_cloud);
  snapshot.trajectory.assign(trajectory_.begin(), trajectory_.end());
  snapshot.optimization = std::move(optimization);
  snapshot.active_keyframes = active_.size();
  snapshot.frozen_submaps = frozen_submaps_;
  snapshot.capacity_reached = capacity_reached_;
  snapshot.total_time_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  return snapshot;
}

void SubmapMapper::Archive(const MappingKeyframe& frame) const {
  if (archive_path_.empty()) return;
  const auto directory = std::filesystem::path(archive_path_);
  const std::string name = FrameName(frame.id);
  const auto depth_path = directory / (name + "_depth.tiff");
  const auto color_path = directory / (name + "_color.png");
  if (!cv::imwrite(depth_path.string(), frame.depth) ||
      !cv::imwrite(color_path.string(), frame.color)) {
    throw std::runtime_error("Cannot write BA keyframe images");
  }
  const auto temporary = directory / (name + ".tmp.yml");
  cv::FileStorage storage(temporary.string(), cv::FileStorage::WRITE);
  if (!storage.isOpened()) throw std::runtime_error("Cannot write BA metadata");
  storage << "version" << 1 << "id" << std::to_string(frame.id) << "sequence"
          << std::to_string(frame.sequence) << "stamp_ns"
          << std::to_string(frame.stamp_ns) << "depth"
          << depth_path.filename().string() << "color"
          << color_path.filename().string() << "intrinsics"
          << cv::Mat(frame.intrinsics) << "baseline" << frame.baseline
          << "vio_pose" << PoseMatrix(frame.vio_pose);
  storage.release();
  std::filesystem::rename(temporary, directory / (name + ".yml"));
}

MappingKeyframe ReadArchivedKeyframe(const std::string& metadata_path) {
  const auto directory = std::filesystem::path(metadata_path).parent_path();
  cv::FileStorage storage(metadata_path, cv::FileStorage::READ);
  if (!storage.isOpened() || static_cast<int>(storage["version"]) != 1) {
    throw std::invalid_argument("Invalid BA archive metadata: " +
                                metadata_path);
  }
  MappingKeyframe frame;
  frame.id = std::stoull(static_cast<std::string>(storage["id"]));
  frame.sequence = std::stoull(static_cast<std::string>(storage["sequence"]));
  frame.stamp_ns = std::stoll(static_cast<std::string>(storage["stamp_ns"]));
  frame.depth = cv::imread(
      (directory / static_cast<std::string>(storage["depth"])).string(),
      cv::IMREAD_UNCHANGED);
  frame.color = cv::imread(
      (directory / static_cast<std::string>(storage["color"])).string(),
      cv::IMREAD_UNCHANGED);
  cv::Mat intrinsics, pose;
  storage["intrinsics"] >> intrinsics;
  storage["vio_pose"] >> pose;
  storage["baseline"] >> frame.baseline;
  if (intrinsics.rows != 3 || intrinsics.cols != 3 ||
      intrinsics.type() != CV_64F || pose.rows != 4 || pose.cols != 4 ||
      pose.type() != CV_64F) {
    throw std::invalid_argument("Invalid BA archive calibration/pose");
  }
  std::vector<double> values;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      values.push_back(pose.at<double>(row, col));
    }
  }
  frame.vio_pose = TransformFromRowMajor(values);
  frame.map_pose = frame.vio_pose;
  frame.intrinsics = cv::Matx33d(intrinsics);
  return frame;
}

SubmapMapperParameters ReadMappingParameters(const std::string& path) {
  const auto root = YAML::LoadFile(path);
  const auto config = root["stereo_reconstruction"]["ros__parameters"];
  if (!config.IsMap()) {
    throw std::invalid_argument(
        "Expected stereo_reconstruction/ros__parameters");
  }
  SubmapMapperParameters parameters;
  VisitMappingFields(&parameters, [&](const std::string& section,
                                      const std::string& name, auto* value) {
    const auto assign = [&](const YAML::Node& node) {
      if (!node.IsDefined()) return;
      using Value = std::remove_reference_t<decltype(*value)>;
      *value = node.template as<Value>();
    };
    if (section.empty()) {
      assign(config[name]);
    } else {
      const auto group = config[section];
      if (!group.IsDefined()) return;
      assign(group[name]);
    }
  });
  // Offline replay never records a second copy of the input archive.
  parameters.archive_directory.clear();
  return parameters;
}

void SaveTrajectory(const std::string& path,
                    const std::vector<MappingPose>& trajectory) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open trajectory: " + path);
  stream << "# timestamp tx ty tz qx qy qz qw\n" << std::setprecision(17);
  for (const auto& entry : trajectory) {
    if (entry.stamp_ns < 0)
      throw std::invalid_argument("Negative trajectory time");
    const Eigen::Quaterniond q(entry.pose.linear());
    stream << entry.stamp_ns / 1000000000LL << '.' << std::setfill('0')
           << std::setw(9) << entry.stamp_ns % 1000000000LL << ' '
           << entry.pose.translation().transpose() << ' ' << q.x() << ' '
           << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
  stream.close();
  if (!stream) throw std::runtime_error("Cannot finish trajectory: " + path);
}

}  // namespace stereo_recon

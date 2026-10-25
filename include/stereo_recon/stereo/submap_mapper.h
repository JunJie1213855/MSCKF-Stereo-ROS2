#ifndef STEREO_RECON_STEREO_SUBMAP_MAPPER_H_
#define STEREO_RECON_STEREO_SUBMAP_MAPPER_H_

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "stereo_recon/stereo/point_cloud_ba.h"
#include "stereo_recon/stereo/reconstruction.h"
#include "stereo_recon/stereo/tsdf_map.h"

namespace stereo_recon {

struct SubmapMapperParameters {
  bool enable_optimization = true;
  PointCloudBaParameters ba;
  TsdfParameters tsdf;
  int submap_keyframes = 20;
  int overlap_keyframes = 4;
  int optimize_every_n_keyframes = 1;
  int max_trajectory_poses = 20000;
  std::string archive_directory;
};

struct MappingPose {
  uint64_t id;
  int64_t stamp_ns;
  Eigen::Isometry3d pose;
};

struct MappingSnapshot {
  int64_t stamp_ns = 0;
  Eigen::Isometry3d map_from_vio = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d camera_pose = Eigen::Isometry3d::Identity();
  std::vector<ColorPoint> points;
  std::vector<ColorPoint> current_points;
  std::vector<MappingPose> trajectory;
  BaSummary optimization;
  size_t active_keyframes = 0;
  size_t frozen_submaps = 0;
  bool capacity_reached = false;
  double total_time_ms = 0.0;
};

// Single-owner mapping engine. Call only from one worker; consumers receive
// immutable snapshots. Frozen storage contains compressed TSDF surface samples,
// not resumable distance fields. Active depth remains available for rebuilding.
class SubmapMapper {
 public:
  explicit SubmapMapper(SubmapMapperParameters parameters);
  void Clear();
  std::optional<MappingSnapshot> Process(
      MappingKeyframe frame, std::vector<ColorPoint> current_cloud,
      const std::function<bool()>& cancelled);
  const std::string& archive_path() const { return archive_path_; }

 private:
  void Archive(const MappingKeyframe& frame) const;
  SubmapMapperParameters parameters_;
  PointCloudBundleAdjuster optimizer_;
  std::deque<MappingKeyframe> active_;
  std::deque<MappingPose> trajectory_;
  std::unique_ptr<TsdfMap> active_tsdf_;
  std::unique_ptr<VoxelMap> frozen_surface_;
  Eigen::Isometry3d map_from_vio_ = Eigen::Isometry3d::Identity();
  std::string archive_path_;
  uint64_t next_id_ = 0;
  uint64_t sequence_ = 0;
  size_t frozen_submaps_ = 0;
  bool capacity_reached_ = false;
  bool rebuild_required_ = false;
};

// Archive metadata is per frame so a partial recording has no broken manifest.
MappingKeyframe ReadArchivedKeyframe(const std::string& metadata_path);
SubmapMapperParameters ReadMappingParameters(const std::string& path);
void SaveTrajectory(const std::string& path,
                    const std::vector<MappingPose>& trajectory);

}  // namespace stereo_recon

#endif  // STEREO_RECON_STEREO_SUBMAP_MAPPER_H_

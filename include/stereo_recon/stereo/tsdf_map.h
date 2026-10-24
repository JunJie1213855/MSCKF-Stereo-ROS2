#ifndef STEREO_RECON_STEREO_TSDF_MAP_H_
#define STEREO_RECON_STEREO_TSDF_MAP_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "stereo_recon/stereo/reconstruction.h"

namespace stereo_recon {

struct TsdfParameters {
  double voxel_size = 0.05;
  double truncation_distance = 0.15;  // Projective optical-Z distance, metres.
  float max_weight = 100.0f;
  float min_weight = 1.0f;
  size_t max_voxels = 500000;
  int integration_stride = 2;  // Allocation rays only; integration uses full depth.
};

struct TsdfSample {
  float distance = 1.0f;  // Normalized signed distance in [-1,1].
  float weight = 0.0f;
};

// Sparse projective TSDF. Allocates only around measured surfaces and updates
// each observed voxel once per keyframe, including previously allocated free
// space. Caller serializes access. Pose is world <- rectified left camera.
class TsdfMap {
 public:
  explicit TsdfMap(const TsdfParameters& parameters);
  // depth: optical Z in metres, CV_32FC1, NaN/zero invalid; color: mono8/BGR8.
  // Returns true if capacity prevented an allocation. Existing voxels still fuse.
  bool Integrate(const cv::Mat& depth, const cv::Mat& color,
                 const cv::Matx33d& intrinsics,
                 const Eigen::Isometry3d& world_from_camera);
  // Interpolate sign crossings between observed neighbouring voxel centres.
  std::vector<ColorPoint> points() const;
  std::optional<TsdfSample> Sample(const Eigen::Vector3d& world) const;
  size_t size() const { return voxels_.size(); }
  void Clear() { voxels_.clear(); }

 private:
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct Hash { size_t operator()(const Key& key) const; };
  struct Voxel {
    TsdfSample sdf;
    Eigen::Vector3f color = Eigen::Vector3f::Zero();
    float color_weight = 0.0f;
  };
  std::optional<Key> GridKey(const Eigen::Vector3d& world) const;
  Eigen::Vector3d Center(const Key& key) const;
  bool AllocateRay(const Eigen::Vector3d& origin,
                   const Eigen::Vector3d& direction, double begin, double end);
  TsdfParameters parameters_;
  std::unordered_map<Key, Voxel, Hash> voxels_;
};

}  // namespace stereo_recon
#endif  // STEREO_RECON_STEREO_TSDF_MAP_H_

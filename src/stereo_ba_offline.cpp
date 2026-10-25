#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stereo_recon/stereo/submap_mapper.h"

namespace stereo_recon {
namespace {

void PrintUsage() {
  std::cerr
      << "Usage: stereo_ba_offline ARCHIVE OUTPUT_PREFIX [options]\n"
      << "  --no-ba                 Fixed VIO poses, same TSDF pipeline\n"
      << "  --sequence N            Select one recorded clear/reset segment\n"
      << "  --params FILE           Override archived mapping parameters\n"
      << "  --window-size N         BA window (default 10)\n"
      << "  --max-plane-rms M       Plane thickness threshold (default .05)\n"
      << "  --disparity-sigma PX    Assumed disparity noise (default .5)\n"
      << "  --voxel-size M          TSDF/output voxel size (default .05)\n"
      << "  --solver-time SEC       Per-solve budget (default .15)\n"
      << "  --max-frames N          Limit the number of replayed frames\n";
}

int Run(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage();
    return 2;
  }
  SubmapMapperParameters parameters;
  std::filesystem::path parameter_path =
      std::filesystem::path(argv[1]) / "mapping_parameters.yaml";
  bool explicit_parameters = false;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--params") {
      if (++i >= argc) throw std::invalid_argument("Missing --params path");
      parameter_path = argv[i];
      explicit_parameters = true;
    }
  }
  if (explicit_parameters || std::filesystem::exists(parameter_path)) {
    parameters = ReadMappingParameters(parameter_path.string());
  }
  std::optional<uint64_t> requested_sequence;
  int max_frames = std::numeric_limits<int>::max();
  bool window_overridden = false;
  bool voxel_overridden = false;
  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--no-ba") {
      parameters.enable_optimization = false;
      continue;
    }
    if (i + 1 >= argc) throw std::invalid_argument("Missing value: " + option);
    const std::string value = argv[++i];
    if (option == "--sequence")
      requested_sequence = std::stoull(value);
    else if (option == "--params")
      continue;
    else if (option == "--window-size") {
      parameters.ba.window_size = std::stoi(value);
      window_overridden = true;
    } else if (option == "--max-plane-rms")
      parameters.ba.max_plane_rms = std::stod(value);
    else if (option == "--disparity-sigma")
      parameters.ba.disparity_sigma = std::stod(value);
    else if (option == "--voxel-size") {
      parameters.tsdf.voxel_size = std::stod(value);
      voxel_overridden = true;
    } else if (option == "--solver-time")
      parameters.ba.max_solver_time_sec = std::stod(value);
    else if (option == "--max-frames")
      max_frames = std::stoi(value);
    else
      throw std::invalid_argument("Unknown option: " + option);
  }
  if (max_frames <= 0)
    throw std::invalid_argument("max-frames must be positive");
  if (window_overridden) {
    if (parameters.ba.window_size < 3 || parameters.ba.window_size > 100) {
      throw std::invalid_argument("window-size must be in [3, 100]");
    }
    parameters.submap_keyframes =
        std::max(parameters.submap_keyframes, parameters.ba.window_size * 2);
    parameters.overlap_keyframes =
        std::min(parameters.overlap_keyframes, parameters.ba.window_size - 1);
  }
  if (voxel_overridden) {
    parameters.tsdf.truncation_distance = std::max(
        parameters.tsdf.truncation_distance, 3.0 * parameters.tsdf.voxel_size);
  }

  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
    const auto name = entry.path().filename().string();
    if (entry.is_regular_file() && name.rfind("frame_", 0) == 0 &&
        entry.path().extension() == ".yml" &&
        name.find(".tmp.") == std::string::npos) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty())
    throw std::invalid_argument("Archive has no committed frames");
  if (files.size() > 1000000)
    throw std::invalid_argument("Archive exceeds replay bound");
  parameters.max_trajectory_poses =
      static_cast<int>(files.size()) + parameters.submap_keyframes;
  SubmapMapper mapper(parameters);
  const std::string prefix = argv[2];
  const auto parent = std::filesystem::path(prefix).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream metrics(prefix + ".metrics.csv");
  if (!metrics) throw std::runtime_error("Cannot open metrics output");
  metrics
      << "id,stamp_ns,accepted,reason,planes,clusters,initial_cost,final_cost,"
         "normal_eigenvalue,ba_ms,mapping_ms,capacity_reached\n";
  std::optional<MappingSnapshot> result;
  std::vector<MappingPose> raw_trajectory;
  std::optional<uint64_t> sequence;
  int processed = 0;
  for (const auto& file : files) {
    auto frame = ReadArchivedKeyframe(file.string());
    if (requested_sequence && frame.sequence != *requested_sequence) continue;
    if (sequence && *sequence != frame.sequence) {
      throw std::invalid_argument(
          "Archive contains multiple reset segments; select one with "
          "--sequence");
    }
    sequence = frame.sequence;
    const uint64_t archived_id = frame.id;
    raw_trajectory.push_back({frame.id, frame.stamp_ns, frame.vio_pose});
    result = mapper.Process(std::move(frame), {}, [] { return false; });
    const auto& ba = result->optimization;
    metrics << archived_id << ',' << result->stamp_ns << ',' << ba.accepted
            << ',' << ba.reason << ',' << ba.planes << ',' << ba.clusters << ','
            << ba.initial_cost << ',' << ba.final_cost << ','
            << ba.normal_eigenvalue << ',' << ba.time_ms << ','
            << result->total_time_ms << ',' << result->capacity_reached << '\n';
    std::cout << "Frame " << ++processed << ": " << ba.reason
              << ", planes=" << ba.planes << ", BA=" << ba.time_ms
              << " ms, mapping=" << result->total_time_ms << " ms\n";
    if (processed >= max_frames) break;
  }
  if (!result)
    throw std::invalid_argument("No frames match the selected sequence");
  SavePointCloud(prefix + ".ply", result->points);
  SaveTrajectory(prefix + ".tum", result->trajectory);
  SaveTrajectory(prefix + ".vio.tum", raw_trajectory);
  metrics.close();
  if (!metrics) throw std::runtime_error("Cannot finish metrics output");
  std::cout << "Saved " << prefix << ".ply, .tum, .vio.tum, .metrics.csv\n";
  return 0;
}

}  // namespace
}  // namespace stereo_recon

int main(int argc, char** argv) {
  try {
    return stereo_recon::Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "Offline reconstruction failed: " << error.what() << '\n';
    return 1;
  }
}

#include <msckf_vio/stereo/stereo_matcher.h>
#include <stdexcept>

namespace msckf_vio::stereo {
class IgevMatcher::Impl {};
namespace {
[[noreturn]] void unavailable() {
  throw std::runtime_error("IGEV TensorRT backend was not built. Rebuild with "
                           "-DMSCKF_ENABLE_TENSORRT=ON and specify TensorRT_ROOT/CUDA_ROOT");
}
}  // namespace
IgevMatcher::IgevMatcher(const IgevParameters&) { unavailable(); }
IgevMatcher::~IgevMatcher() = default;
cv::Mat IgevMatcher::compute(const cv::Mat&, const cv::Mat&) { unavailable(); }
}  // namespace msckf_vio::stereo

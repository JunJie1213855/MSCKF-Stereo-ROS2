#include <stdexcept>

#include "stereo_recon/stereo/stereo_matcher.h"

namespace stereo_recon {
class IgevMatcher::Impl {};
class S2m2Matcher::Impl {};
namespace {
[[noreturn]] void ThrowUnavailable() {
  throw std::runtime_error(
      "TensorRT stereo backends was not built. Rebuild with "
      "-DSTEREO_RECON_ENABLE_TENSORRT=ON and specify TensorRT_ROOT/CUDA_ROOT");
}
}  // namespace
IgevMatcher::IgevMatcher(const IgevParameters&) { ThrowUnavailable(); }
void IgevMatcher::ValidateInputSize(cv::Size) const { ThrowUnavailable(); }
IgevMatcher::~IgevMatcher() = default;
cv::Mat IgevMatcher::Compute(const cv::Mat&, const cv::Mat&) {
  ThrowUnavailable();
}
S2m2Matcher::S2m2Matcher(const S2m2Parameters&) { ThrowUnavailable(); }
S2m2Matcher::~S2m2Matcher() = default;
cv::Mat S2m2Matcher::Compute(const cv::Mat&, const cv::Mat&) {
  ThrowUnavailable();
}
void S2m2Matcher::ValidateInputSize(cv::Size) const { ThrowUnavailable(); }
}  // namespace stereo_recon

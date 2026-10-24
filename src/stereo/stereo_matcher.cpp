#include "stereo_recon/stereo/stereo_matcher.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "opencv2/imgproc.hpp"

namespace stereo_recon {
namespace {
cv::Mat MakeGrayscaleImage(const cv::Mat& image) {
  if (image.type() == CV_8UC1) {
    return image;
  }
  if (image.type() != CV_8UC3) {
    throw std::invalid_argument("StereoMatcher expects mono8 or BGR8 images");
  }
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  return gray;
}
}  // namespace

SgbmMatcher::SgbmMatcher(const SgbmParameters& p) : parameters_(p) {
  if (p.min_disparity < 0 || p.num_disparities <= 0 ||
      p.num_disparities % 16 != 0 || p.block_size < 3 || p.block_size > 255 ||
      p.block_size % 2 == 0 || p.uniqueness_ratio < 0 ||
      p.speckle_window_size < 0 || p.speckle_range < 0 ||
      p.pre_filter_cap <= 0) {
    throw std::invalid_argument(
        "Invalid SGBM parameters: disparities must be a "
        "positive multiple of 16 and block_size odd >= 3");
  }
  const int area = p.block_size * p.block_size;
  matcher_ = cv::StereoSGBM::create(
      p.min_disparity, p.num_disparities, p.block_size, 8 * area, 32 * area,
      p.disp12_max_diff, p.pre_filter_cap, p.uniqueness_ratio,
      p.speckle_window_size, p.speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);
}

cv::Mat SgbmMatcher::Compute(const cv::Mat& left, const cv::Mat& right) {
  if (left.empty() || left.size() != right.size() ||
      left.cols <= parameters_.min_disparity + parameters_.num_disparities +
                       parameters_.block_size) {
    throw std::invalid_argument(
        "Stereo images are empty, mismatched or too narrow for SGBM");
  }
  cv::Mat raw, disparity;
  matcher_->compute(MakeGrayscaleImage(left), MakeGrayscaleImage(right), raw);
  // OpenCV SGBM 返回放大 16 倍的定点视差；统一接口要求原图像素单位。
  raw.convertTo(disparity, CV_32F, 1.0 / 16.0);
  const int invalid = (parameters_.min_disparity - 1) * 16;
  for (int y = 0; y < raw.rows; ++y) {
    const auto* source = raw.ptr<int16_t>(y);
    auto* destination = disparity.ptr<float>(y);
    for (int x = 0; x < raw.cols; ++x) {
      if (source[x] <= invalid || destination[x] <= 0.0f) {
        destination[x] = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return disparity;
}

std::unique_ptr<StereoMatcher> CreateStereoMatcher(
    const std::string& backend, const SgbmParameters& parameters,
    const IgevParameters& igev_parameters,
    const S2m2Parameters& s2m2_parameters) {
  if (backend == "sgbm") {
    return std::make_unique<SgbmMatcher>(parameters);
  }
  if (backend == "igev") {
    return std::make_unique<IgevMatcher>(igev_parameters);
  }
  if (backend == "s2m2") {
    return std::make_unique<S2m2Matcher>(s2m2_parameters);
  }
  throw std::invalid_argument("Unsupported stereo backend '" + backend +
                              "'; available backends: sgbm, igev, s2m2");
}
}  // namespace stereo_recon

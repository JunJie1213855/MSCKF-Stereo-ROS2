#pragma once

#include <memory>
#include <string>
#include <opencv2/calib3d.hpp>

namespace msckf_vio::stereo {

// Backends receive equally sized, rectified mono8 or BGR8 images. They must
// return CV_32FC1 left disparity, in input-image pixels (x_left - x_right),
// at the input resolution, with NaN for invalid measurements. TensorRT
// backends own their color conversion, normalization, padding and unpadding.
class StereoMatcher {
 public:
  virtual ~StereoMatcher() = default;
  virtual cv::Mat compute(const cv::Mat& left, const cv::Mat& right) = 0;
  virtual std::string name() const = 0;
};

struct SgbmParameters {
  int min_disparity = 0;
  int num_disparities = 128;
  int block_size = 5;
  int uniqueness_ratio = 10;
  int speckle_window_size = 100;
  int speckle_range = 2;
  int disp12_max_diff = 1;
  int pre_filter_cap = 31;
};

class SgbmMatcher final : public StereoMatcher {
 public:
  explicit SgbmMatcher(const SgbmParameters& parameters);
  cv::Mat compute(const cv::Mat& left, const cv::Mat& right) override;
  std::string name() const override { return "sgbm"; }

 private:
  SgbmParameters parameters_;
  cv::Ptr<cv::StereoSGBM> matcher_;
};

std::unique_ptr<StereoMatcher> createStereoMatcher(
    const std::string& backend, const SgbmParameters& parameters);

}  // namespace msckf_vio::stereo

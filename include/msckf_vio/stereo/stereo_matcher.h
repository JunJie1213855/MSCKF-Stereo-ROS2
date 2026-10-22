#ifndef MSCKF_VIO_STEREO_STEREO_MATCHER_H_
#define MSCKF_VIO_STEREO_STEREO_MATCHER_H_

#include <memory>
#include <string>

#include "opencv2/calib3d.hpp"

namespace msckf_vio::stereo {

// 视差后端的统一边界：输入为已校正且等大的 mono8/BGR8 图像，输出为
// 原始分辨率的 CV_32FC1 左视差 d=x_left-x_right，单位像素，无效值为 NaN。
// 模型自身的颜色布局、归一化和补边由后端处理，几何层不感知这些差异。
class StereoMatcher {
 public:
  virtual ~StereoMatcher() = default;
  virtual cv::Mat Compute(const cv::Mat& left, const cv::Mat& right) = 0;
  virtual std::string name() const = 0;
  // 启动时用标定分辨率检查 engine，避免收到第一帧后才发现高宽不匹配。
  virtual void ValidateInputSize(cv::Size) const {}
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
  cv::Mat Compute(const cv::Mat& left, const cv::Mat& right) override;
  std::string name() const override { return "sgbm"; }

 private:
  SgbmParameters parameters_;
  cv::Ptr<cv::StereoSGBM> matcher_;
};

struct IgevParameters {
  std::string engine_path;
  std::string left_input_name = "left";
  std::string right_input_name = "right";
  std::string output_name = "disparity";
  // Zero uses the static engine dimension, or pads the image to a multiple
  // of padding_multiple for a dynamic dimension. No image resizing is done.
  int input_width = 0;
  int input_height = 0;
  int padding_multiple = 32;
  std::string padding_mode = "right_bottom";
  int device_id = 0;
};

class IgevMatcher final : public StereoMatcher {
 public:
  explicit IgevMatcher(const IgevParameters& parameters);
  ~IgevMatcher() override;
  cv::Mat Compute(const cv::Mat& left, const cv::Mat& right) override;
  std::string name() const override { return "igev"; }
  void ValidateInputSize(cv::Size size) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct S2m2Parameters {
  std::string engine_path;
  std::string left_input_name = "input_left";
  std::string right_input_name = "input_right";
  std::string output_name = "output_disp";
  std::string visibility_output_name = "output_occ";  // 值越大，越可见。
  std::string confidence_output_name = "output_conf";
  int input_width = 0;
  int input_height = 0;
  int device_id = 0;
  double min_confidence = 0.7;
  double min_visibility = 0.7;
};

class S2m2Matcher final : public StereoMatcher {
 public:
  explicit S2m2Matcher(const S2m2Parameters& parameters);
  ~S2m2Matcher() override;
  cv::Mat Compute(const cv::Mat& left, const cv::Mat& right) override;
  std::string name() const override { return "s2m2"; }
  void ValidateInputSize(cv::Size size) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<StereoMatcher> CreateStereoMatcher(
    const std::string& backend, const SgbmParameters& parameters,
    const IgevParameters& igev_parameters = IgevParameters{},
    const S2m2Parameters& s2m2_parameters = S2m2Parameters{});

}  // namespace msckf_vio::stereo

#endif  // MSCKF_VIO_STEREO_STEREO_MATCHER_H_

#include "stereo_recon/stereo/igev_processing.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

namespace stereo_recon {
namespace {
cv::Mat MakeColorImage(const cv::Mat& image) {
  if (image.type() == CV_8UC3) {
    return image;
  }
  if (image.type() != CV_8UC1) {
    throw std::invalid_argument("IGEV expects mono8 or BGR8 images");
  }
  cv::Mat color;
  cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
  return color;
}
}  // namespace

IgevInput PrepareIgevInput(const cv::Mat& left, const cv::Mat& right,
                           cv::Size network_size,
                           const std::string& padding_mode) {
  if (left.empty() || left.size() != right.size() ||
      network_size.width < left.cols || network_size.height < left.rows) {
    throw std::invalid_argument(
        "IGEV inputs must have equal, nonempty sizes fitting the engine; "
        "image resizing is not implicit");
  }
  // 阶段 1：只补边以满足 engine 尺寸，记录原图 ROI，不改变相机内参与视差尺度。
  const int pad_x = network_size.width - left.cols;
  const int pad_y = network_size.height - left.rows;
  int x = 0, y = 0, border = cv::BORDER_CONSTANT;
  if (padding_mode == "sintel") {
    x = pad_x / 2;
    y = pad_y / 2;
    border = cv::BORDER_REPLICATE;
  } else if (padding_mode != "right_bottom") {
    throw std::invalid_argument(
        "IGEV padding_mode must be right_bottom or sintel");
  }
  cv::Mat left_padded, right_padded;
  cv::copyMakeBorder(MakeColorImage(left), left_padded, y, pad_y - y, x,
                     pad_x - x, border | cv::BORDER_ISOLATED,
                     cv::Scalar(0, 0, 0));
  cv::copyMakeBorder(MakeColorImage(right), right_padded, y, pad_y - y, x,
                     pad_x - x, border | cv::BORDER_ISOLATED,
                     cv::Scalar(0, 0, 0));
  // 阶段 2：BGR→RGB、HWC→NCHW、uint8→float32，值域仍为 [0,255]。
  // 模型图内已有 2*(image/255)-1；这里再次归一化会直接改变推理结果。
  return {cv::dnn::blobFromImage(left_padded, 1.0, cv::Size(), cv::Scalar(),
                                 true, false),
          cv::dnn::blobFromImage(right_padded, 1.0, cv::Size(), cv::Scalar(),
                                 true, false),
          network_size, cv::Rect(x, y, left.cols, left.rows)};
}

cv::Mat RestoreIgevDisparity(const cv::Mat& output, const IgevInput& input) {
  if (output.empty() || output.channels() != 1 ||
      (output.depth() != CV_32F && output.depth() != CV_16F) ||
      output.dims < 2 || output.dims > 4 ||
      output.size[output.dims - 2] != input.network_size.height ||
      output.size[output.dims - 1] != input.network_size.width) {
    throw std::invalid_argument(
        "IGEV final disparity must be FP32/FP16 [H,W], [1,H,W] "
        "or [1,1,H,W] at engine input resolution");
  }
  for (int dimension = 0; dimension < output.dims - 2; ++dimension) {
    if (output.size[dimension] != 1) {
      throw std::invalid_argument(
          "IGEV output must contain exactly one disparity image");
    }
  }
  const cv::Rect bounds(0, 0, input.network_size.width,
                        input.network_size.height);
  if (input.image_roi.empty() ||
      (bounds & input.image_roi) != input.image_roi) {
    throw std::invalid_argument("Invalid IGEV unpadding region");
  }
  // 阶段 3：恢复 FP32 并裁回原始区域。IGEV 输出已是像素视差，不要再除以 16。
  cv::Mat floats;
  output.convertTo(floats, CV_32F);
  if (!floats.isContinuous()) {
    floats = floats.clone();
  }
  cv::Mat disparity =
      floats.reshape(1, input.network_size.height)(input.image_roi).clone();
  for (int y = 0; y < disparity.rows; ++y) {
    auto* values = disparity.ptr<float>(y);
    for (int x = 0; x < disparity.cols; ++x) {
      if (!std::isfinite(values[x]) || values[x] <= 0) {
        values[x] = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return disparity;
}
}  // namespace stereo_recon

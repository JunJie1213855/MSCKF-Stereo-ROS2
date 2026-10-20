#include "msckf_vio/stereo/s2m2_processing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "opencv2/imgproc.hpp"

namespace msckf_vio::stereo {
namespace {
// 自适应池化用 floor/ceil 窗口，尺寸不可整除时相邻窗口允许重叠。
// INTER_AREA 的采样窗口不同，不能替代 PyTorch adaptive_avg_pool2d。
cv::Mat AdaptiveAveragePool(const cv::Mat& padded, int pool_h, int pool_w) {
  // PyTorch adaptive_avg_pool2d uses floor(start) and ceil(end), including
  // overlapping bins for non-divisible sizes. cv::INTER_AREA is not equivalent.
  cv::Mat pooled(pool_h, pool_w, CV_32FC3);
  for (int y = 0; y < pool_h; ++y) {
    const int y0 =
        static_cast<int>(static_cast<int64_t>(y) * padded.rows / pool_h);
    const int y1 = static_cast<int>(
        (static_cast<int64_t>(y + 1) * padded.rows + pool_h - 1) / pool_h);
    for (int x = 0; x < pool_w; ++x) {
      const int x0 =
          static_cast<int>(static_cast<int64_t>(x) * padded.cols / pool_w);
      const int x1 = static_cast<int>(
          (static_cast<int64_t>(x + 1) * padded.cols + pool_w - 1) / pool_w);
      cv::Vec3f sum(0, 0, 0);
      for (int py = y0; py < y1; ++py) {
        for (int px = x0; px < x1; ++px) {
          const auto v = padded.at<cv::Vec3b>(py, px);
          for (int c = 0; c < 3; ++c) {
            sum[c] += v[c];
          }
        }
      }
      pooled.at<cv::Vec3f>(y, x) =
          sum / static_cast<float>((y1 - y0) * (x1 - x0));
    }
  }
  return pooled;
}

cv::Mat PrepareImage(const cv::Mat& image, cv::Size size, const cv::Rect& roi) {
  // 阶段 1：先在 BGR/灰度空间居中补边，最后统一转成 RGB NCHW。
  cv::Mat color;
  if (image.type() == CV_8UC1) {
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
  } else if (image.type() == CV_8UC3) {
    color = image;
  } else {
    throw std::invalid_argument("S2M2 expects mono8 or BGR8 images");
  }
  cv::Mat padded;
  cv::copyMakeBorder(color, padded, roi.y, size.height - roi.y - image.rows,
                     roi.x, size.width - roi.x - image.cols,
                     cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar());
  // 阶段 2：用低分辨率图像填充边缘，原图区域稍后恢复。
  const auto pooled =
      AdaptiveAveragePool(padded, image.rows / 32, image.cols / 32);
  cv::Mat filled;
  cv::resize(pooled, filled, size, 0, 0, cv::INTER_LINEAR);
  // 阶段 3：恢复原图像素，边缘保留插值结果；同时交换 BGR/RGB 通道。
  // torch.to(uint8) 截断小数，而 OpenCV convertTo(CV_8U) 会四舍五入。
  const int shape[] = {1, 3, size.height, size.width};
  cv::Mat blob(4, shape, CV_8UC1);
  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      const bool original = roi.contains(cv::Point(x, y));
      for (int c = 0; c < 3; ++c) {
        const float value =
            original ? color.at<cv::Vec3b>(y - roi.y, x - roi.x)[2 - c]
                     : filled.at<cv::Vec3f>(y, x)[2 - c];
        blob.ptr<uint8_t>(0, c)[y * size.width + x] =
            static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
      }
    }
  }
  return blob;
}

cv::Mat CropOutput(const cv::Mat& output, const StereoTensorInput& input) {
  if (output.empty() || output.channels() != 1 ||
      (output.depth() != CV_32F && output.depth() != CV_16F) ||
      output.dims < 2 || output.dims > 4 ||
      output.size[output.dims - 2] != input.network_size.height ||
      output.size[output.dims - 1] != input.network_size.width) {
    throw std::invalid_argument(
        "S2M2 outputs must be FP32/FP16 single images at engine input "
        "resolution");
  }
  for (int i = 0; i < output.dims - 2; ++i) {
    if (output.size[i] != 1) {
      throw std::invalid_argument("S2M2 output must contain one image");
    }
  }
  const cv::Rect bounds(0, 0, input.network_size.width,
                        input.network_size.height);
  if (input.image_roi.empty() ||
      (bounds & input.image_roi) != input.image_roi) {
    throw std::invalid_argument("Invalid S2M2 unpadding region");
  }
  cv::Mat floats;
  output.convertTo(floats, CV_32F);
  if (!floats.isContinuous()) {
    floats = floats.clone();
  }
  return floats.reshape(1, input.network_size.height)(input.image_roi).clone();
}
}  // namespace

StereoTensorInput PrepareS2m2Input(const cv::Mat& left, const cv::Mat& right,
                                   cv::Size size) {
  if (left.empty() || left.size() != right.size() || left.cols < 32 ||
      left.rows < 32 || size.width < left.cols || size.height < left.rows ||
      size.width % 32 || size.height % 32) {
    throw std::invalid_argument(
        "S2M2 expects equal images >=32 pixels fitting a multiple-of-32 "
        "engine; "
        "no implicit resize or transpose is allowed");
  }
  const cv::Rect roi((size.width - left.cols) / 2,
                     (size.height - left.rows) / 2, left.cols, left.rows);
  return {PrepareImage(left, size, roi), PrepareImage(right, size, roi), size,
          roi};
}

cv::Mat RestoreS2m2Disparity(const cv::Mat& output, const cv::Mat& visibility,
                             const cv::Mat& confidence,
                             const StereoTensorInput& input,
                             double min_confidence, double min_visibility) {
  if (!std::isfinite(min_confidence) || !std::isfinite(min_visibility) ||
      min_confidence < 0 || min_confidence > 1 || min_visibility < 0 ||
      min_visibility > 1) {
    throw std::invalid_argument(
        "S2M2 confidence/visibility thresholds must be in [0,1]");
  }
  // 阶段 4：三个输出必须裁同一个 ROI，否则置信度会与视差错位。
  // 只裁补边而不缩放，因此视差仍是原始图像的像素单位。
  cv::Mat disparity = CropOutput(output, input);
  const auto occ = CropOutput(visibility, input),
             conf = CropOutput(confidence, input);
  // output_occ 实际表示可见性；阈值失败统一写 NaN，几何层会跳过该像素。
  for (int y = 0; y < disparity.rows; ++y) {
    for (int x = 0; x < disparity.cols; ++x) {
      auto& d = disparity.at<float>(y, x);
      const float o = occ.at<float>(y, x), c = conf.at<float>(y, x);
      if (!std::isfinite(d) || d <= 0 || !std::isfinite(o) ||
          !std::isfinite(c) || o <= static_cast<float>(min_visibility) ||
          c <= static_cast<float>(min_confidence)) {
        d = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return disparity;
}
}  // namespace msckf_vio::stereo

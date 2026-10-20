#ifndef MSCKF_VIO_STEREO_STEREO_TENSOR_INPUT_H_
#define MSCKF_VIO_STEREO_STEREO_TENSOR_INPUT_H_

#include "opencv2/core.hpp"

namespace msckf_vio::stereo {

// 连续 RGB NCHW 输入：IGEV 使用 FP32，S2M2 使用 UINT8，值域均为 [0,255]。
// 归一化已包含在模型中；此结构只描述模型输入及恢复原始分辨率所需的 ROI。
struct StereoTensorInput {
  cv::Mat left_blob;
  cv::Mat right_blob;
  cv::Size network_size;
  cv::Rect image_roi;
};

}  // namespace msckf_vio::stereo

#endif  // MSCKF_VIO_STEREO_STEREO_TENSOR_INPUT_H_

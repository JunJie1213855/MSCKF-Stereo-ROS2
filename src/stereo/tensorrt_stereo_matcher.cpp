#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
// CUDA SDK 使用平铺的头文件名，没有可添加的项目子目录。
#include "cuda_runtime_api.h"  // NOLINT(build/include_subdir)

#include "stereo_recon/stereo/igev_processing.h"
#include "stereo_recon/stereo/s2m2_processing.h"
#include "stereo_recon/stereo/stereo_matcher.h"

namespace stereo_recon {
static_assert(NV_TENSORRT_MAJOR == 10 || NV_TENSORRT_MAJOR == 11,
              "The TensorRT stereo backend supports TensorRT 10.x and 11.x");
namespace {
void CheckCuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("TensorRT stereo ") + operation +
                             ": " + cudaGetErrorString(result));
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity > Severity::kWARNING) {
      return;
    }
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (severity <= Severity::kERROR) {
        last_error_ = message;
      }
      std::cerr << "[Stereo TensorRT] " << message << '\n';
    } catch (...) {
    }  // TensorRT log callbacks must not throw.
  }
  std::string ErrorMessage(const std::string& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    return "TensorRT stereo " + operation +
           (last_error_.empty() ? "" : ": " + last_error_);
  }

 private:
  std::mutex mutex_;
  std::string last_error_;
};

class DeviceBuffer {
 public:
  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void Ensure(size_t bytes) {
    if (bytes <= capacity_) {
      return;
    }
    if (data_) {
      CheckCuda(cudaFree(data_), "free device buffer");
    }
    data_ = nullptr;
    capacity_ = 0;
    CheckCuda(cudaMalloc(&data_, bytes), "allocate device buffer");
    capacity_ = bytes;
  }
  void* data() const { return data_; }

 private:
  void* data_ = nullptr;
  size_t capacity_ = 0;
};

struct Stream {
  cudaStream_t value = nullptr;
  ~Stream() {
    if (value) {
      cudaStreamDestroy(value);
    }
  }
};

// On any inference exception, finish pending transfers before local host
// tensors can be released or reused on the next call.
struct StreamFence {
  cudaStream_t stream;
  ~StreamFence() { cudaStreamSynchronize(stream); }
};

int GetCvTensorType(nvinfer1::DataType type) {
  if (type == nvinfer1::DataType::kFLOAT) {
    return CV_32F;
  }
  if (type == nvinfer1::DataType::kHALF) {
    return CV_16F;
  }
  if (type == nvinfer1::DataType::kUINT8) {
    return CV_8U;
  }
  throw std::invalid_argument(
      "TensorRT stereo supports UINT8/FP32/FP16 tensor I/O only");
}

size_t ComputeTensorBytes(const nvinfer1::Dims& dimensions, int type) {
  if (dimensions.nbDims < 1 || dimensions.nbDims > nvinfer1::Dims::MAX_DIMS) {
    throw std::invalid_argument(
        "TensorRT stereo encountered an unsupported tensor rank");
  }
  size_t bytes = CV_ELEM_SIZE(type);
  for (int i = 0; i < dimensions.nbDims; ++i) {
    if (dimensions.d[i] <= 0 ||
        dimensions.d[i] > std::numeric_limits<int>::max() ||
        bytes > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(dimensions.d[i])) {
      throw std::invalid_argument(
          "TensorRT stereo output shape is unresolved or too large");
    }
    bytes *= dimensions.d[i];
  }
  return bytes;
}
}  // namespace

// 一个 matcher 拥有一个 context/stream；互斥锁保证同一实例不会并发 enqueue。
// 析构顺序很重要：缓冲区先释放，然后 context/engine/runtime，logger 最后析构。
class TensorRtStereoRuntime {
 public:
  explicit TensorRtStereoRuntime(const IgevParameters& parameters,
                                 const S2m2Parameters& s2m2 = S2m2Parameters{},
                                 bool s2m2_mode = false)
      : parameters_(parameters), s2m2_(s2m2), is_s2m2_(s2m2_mode) {
    // 构造阶段只做一次：检查配置、反序列化 engine、核对张量接口。
    ValidateParameters();
    LoadEngine();
    ValidateEngineIo();
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error(
          logger_.ErrorMessage("failed to create execution context"));
    }
    CheckCuda(cudaStreamCreate(&stream_.value), "create CUDA stream");
    std::clog << (is_s2m2_ ? "[S2M2] Loaded " : "[IGEV] Loaded ")
              << parameters.engine_path << " on CUDA device "
              << parameters.device_id
              << "; inputs=" << parameters.left_input_name << ','
              << parameters.right_input_name
              << "; output=" << parameters.output_name
              << "; padding=" << parameters.padding_mode << '\n';
  }

  ~TensorRtStereoRuntime() {
    // ROS constructs this object on the main thread but inference runs on its
    // worker thread; the CUDA device must be selected on both threads.
    cudaSetDevice(parameters_.device_id);
    if (stream_.value) {
      cudaStreamSynchronize(stream_.value);
    }
  }

  cv::Mat Compute(const cv::Mat& left, const cv::Mat& right) {
    // One context/stream per matcher; serialize direct callers as well as ROS.
    std::lock_guard<std::mutex> lock(inference_mutex_);
    CheckCuda(cudaSetDevice(parameters_.device_id),
              "select inference CUDA device");
    ValidateInputSize(left.size());
    const auto size = ResolveInputSize(left.size());
    // 阶段 1：模型专用预处理；只补边，不缩放，不重复执行模型内归一化。
    const auto prepared = is_s2m2_ ? PrepareS2m2Input(left, right, size)
                                   : PrepareIgevInput(left, right, size,
                                                      parameters_.padding_mode);
    // 阶段 2：解析运行时形状，并为所有 I/O 复用缓冲区、绑定设备地址。
    const auto selected = SelectedOutputs();
    PrepareBindings(size, selected);

    auto& left_tensor = tensors_.at(parameters_.left_input_name);
    auto& right_tensor = tensors_.at(parameters_.right_input_name);
    prepared.left_blob.convertTo(left_tensor.host, left_tensor.host.type());
    prepared.right_blob.convertTo(right_tensor.host, right_tensor.host.type());
    // 阶段 3：H2D → enqueueV3 → D2H 在同一 stream 上顺序执行。
    // fence 在离开作用域时同步，确保 host 缓冲区被释放/复用前传输已结束。
    StreamFence fence{stream_.value};
    for (auto* input : {&left_tensor, &right_tensor}) {
      CheckCuda(
          cudaMemcpyAsync(input->device.data(), input->host.data, input->bytes,
                          cudaMemcpyHostToDevice, stream_.value),
          "upload input");
    }
    if (!context_->enqueueV3(stream_.value)) {
      throw std::runtime_error(logger_.ErrorMessage("enqueueV3 failed"));
    }
    for (const auto& name : selected) {
      auto& output = tensors_.at(name);
      CheckCuda(
          cudaMemcpyAsync(output.host.data, output.device.data(), output.bytes,
                          cudaMemcpyDeviceToHost, stream_.value),
          "download stereo output");
    }
    CheckCuda(cudaStreamSynchronize(stream_.value), "synchronize inference");
    // 阶段 4：同步完成才读取 CPU 输出，裁边并应用 S2M2 质量掩码。
    const auto& output = tensors_.at(parameters_.output_name).host;
    if (is_s2m2_) {
      return RestoreS2m2Disparity(
          output, tensors_.at(s2m2_.visibility_output_name).host,
          tensors_.at(s2m2_.confidence_output_name).host, prepared,
          s2m2_.min_confidence, s2m2_.min_visibility);
    }
    return RestoreIgevDisparity(output, prepared);
  }

  void ValidateInputSize(cv::Size image) const {
    const auto size = ResolveInputSize(image);
    if (image.width <= 0 || image.height <= 0 || image.width > size.width ||
        image.height > size.height ||
        (is_s2m2_ && (image.width < 32 || image.height < 32 ||
                      size.width % 32 || size.height % 32))) {
      throw std::invalid_argument(
          std::string(is_s2m2_ ? "S2M2" : "IGEV") + " image WxH=" +
          std::to_string(image.width) + "x" + std::to_string(image.height) +
          " does not fit engine WxH=" + std::to_string(size.width) + "x" +
          std::to_string(size.height) +
          "; no implicit resize/transpose. EuRoC requires NCHW [1,3,480,768]. "
          "Re-export the engine.");
    }
  }

 private:
  void ValidateParameters() const {
    const auto& parameters = parameters_;
    if (parameters.engine_path.empty() || parameters.device_id < 0 ||
        parameters.input_width < 0 || parameters.input_height < 0 ||
        parameters.padding_multiple <= 0 ||
        parameters.left_input_name.empty() ||
        parameters.right_input_name.empty() || parameters.output_name.empty() ||
        parameters.left_input_name == parameters.right_input_name ||
        (parameters.padding_mode != "right_bottom" &&
         parameters.padding_mode != "sintel" && !is_s2m2_)) {
      throw std::invalid_argument(
          "Invalid TensorRT stereo engine path, tensor names, dimensions, "
          "device or padding mode");
    }
    if (is_s2m2_ &&
        (s2m2_.visibility_output_name.empty() ||
         s2m2_.confidence_output_name.empty() ||
         s2m2_.visibility_output_name == s2m2_.confidence_output_name ||
         s2m2_.visibility_output_name == parameters.output_name ||
         s2m2_.confidence_output_name == parameters.output_name ||
         !std::isfinite(s2m2_.min_confidence) ||
         !std::isfinite(s2m2_.min_visibility) || s2m2_.min_confidence < 0 ||
         s2m2_.min_confidence > 1 || s2m2_.min_visibility < 0 ||
         s2m2_.min_visibility > 1)) {
      throw std::invalid_argument(
          "Invalid S2M2 quality output names or thresholds");
    }
  }

  void LoadEngine() {
    const auto& parameters = parameters_;
    std::ifstream file(parameters.engine_path,
                       std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) {
      throw std::runtime_error(
          "TensorRT stereo engine is missing or empty (conversion may still be "
          "running): " +
          parameters.engine_path);
    }
    const auto length = static_cast<size_t>(file.tellg());
    std::vector<char> serialized(length);
    file.seekg(0);
    if (!file.read(serialized.data(), static_cast<std::streamsize>(length))) {
      throw std::runtime_error("Cannot read TensorRT stereo engine: " +
                               parameters.engine_path);
    }
    CheckCuda(cudaSetDevice(parameters.device_id), "select CUDA device");
    if (!initLibNvInferPlugins(&logger_, "")) {
      throw std::runtime_error(
          logger_.ErrorMessage("failed to initialize TensorRT plugins"));
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error(
          logger_.ErrorMessage("failed to create runtime"));
    }
    engine_.reset(
        runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!engine_) {
      throw std::runtime_error(
          logger_.ErrorMessage("failed to deserialize engine"));
    }
  }

  // IGEV 与 S2M2 复用运行时，但输入 dtype、张量名称与质量输出约定不同。
  void ValidateEngineIo() const {
    const auto& parameters = parameters_;
    int inputs = 0;
    bool found_left = false, found_right = false, found_output = false;
    std::vector<std::string> outputs;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const std::string name = engine_->getIOTensorName(i);
      const bool is_input = engine_->getTensorIOMode(name.c_str()) ==
                            nvinfer1::TensorIOMode::kINPUT;
      const int type =
          GetCvTensorType(engine_->getTensorDataType(name.c_str()));
      const bool floating = type == CV_32F || type == CV_16F;
      const bool s2m2_uint8 = is_input && is_s2m2_ && type == CV_8U;
      if (!floating && !s2m2_uint8) {
        throw std::invalid_argument(
            "Unexpected tensor type: S2M2 inputs must be UINT8/FP32/FP16; "
            "IGEV inputs and all outputs must be FP32/FP16: " +
            name);
      }
      if (engine_->getTensorFormat(name.c_str()) !=
              nvinfer1::TensorFormat::kLINEAR ||
          engine_->getTensorLocation(name.c_str()) !=
              nvinfer1::TensorLocation::kDEVICE) {
        throw std::invalid_argument(
            "TensorRT stereo requires linear device tensor I/O: " + name);
      }
      if (is_input) {
        ++inputs;
        const auto shape = engine_->getTensorShape(name.c_str());
        if (shape.nbDims != 4 || (shape.d[0] != 1 && shape.d[0] != -1) ||
            (shape.d[1] != 3 && shape.d[1] != -1)) {
          throw std::invalid_argument(
              "TensorRT stereo inputs must be NCHW [1,3,H,W]: " + name);
        }
        for (int axis : {2, 3}) {
          if (shape.d[axis] != -1 &&
              (shape.d[axis] <= 0 ||
               shape.d[axis] > std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "Invalid TensorRT stereo input spatial dimensions: " + name);
          }
        }
        found_left |= name == parameters.left_input_name;
        found_right |= name == parameters.right_input_name;
      } else {
        found_output |= name == parameters.output_name;
        outputs.push_back(name);
      }
    }
    if (inputs != 2 || !found_left || !found_right || !found_output) {
      throw std::invalid_argument(
          "TensorRT stereo engine tensor names differ from the configured "
          "left/right/disparity names");
    }
    if (is_s2m2_) {
      for (const auto& name :
           {s2m2_.visibility_output_name, s2m2_.confidence_output_name}) {
        if (std::find(outputs.begin(), outputs.end(), name) == outputs.end()) {
          throw std::invalid_argument("Missing S2M2 quality output: " + name);
        }
      }
    }
  }

  std::vector<std::string> SelectedOutputs() const {
    std::vector<std::string> selected{parameters_.output_name};
    if (is_s2m2_) {
      selected.push_back(s2m2_.visibility_output_name);
      selected.push_back(s2m2_.confidence_output_name);
    }
    return selected;
  }

  void PrepareBindings(cv::Size size,
                       const std::vector<std::string>& selected) {
    const nvinfer1::Dims4 dimensions(1, 3, size.height, size.width);
    for (const auto& name :
         {parameters_.left_input_name, parameters_.right_input_name}) {
      if (!context_->setInputShape(name.c_str(), dimensions)) {
        throw std::invalid_argument(
            "TensorRT stereo input dimensions are outside the engine "
            "shape/profile: " +
            name);
      }
    }
    if (context_->inferShapes(0, nullptr) != 0) {
      throw std::runtime_error(
          logger_.ErrorMessage("could not resolve all engine tensor shapes"));
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const std::string name = engine_->getIOTensorName(i);
      auto& tensor = tensors_[name];
      const auto shape = context_->getTensorShape(name.c_str());
      const int type =
          GetCvTensorType(engine_->getTensorDataType(name.c_str()));
      tensor.bytes = ComputeTensorBytes(shape, type);
      // TensorRT 10 dimensions are int64; OpenCV uses int dimensions.
      std::vector<int> cv_shape;
      for (int axis = 0; axis < shape.nbDims; ++axis) {
        cv_shape.push_back(static_cast<int>(shape.d[axis]));
      }
      tensor.host.create(shape.nbDims, cv_shape.data(), type);
      tensor.device.Ensure(tensor.bytes);
      if (!context_->setTensorAddress(name.c_str(), tensor.device.data())) {
        throw std::runtime_error(
            logger_.ErrorMessage("could not bind tensor " + name));
      }
    }
    // Validate final output before enqueue; never guess a low-resolution
    // output's disparity scale or select an arbitrary output tensor.
    for (const auto& name : selected) {
      const auto shape = context_->getTensorShape(name.c_str());
      if (shape.nbDims < 2 || shape.nbDims > 4 ||
          shape.d[shape.nbDims - 2] != size.height ||
          shape.d[shape.nbDims - 1] != size.width) {
        throw std::invalid_argument("Stereo output must be full-resolution: " +
                                    name);
      }
      for (int i = 0; i < shape.nbDims - 2; ++i) {
        if (shape.d[i] != 1) {
          throw std::invalid_argument("Stereo output must be one image: " +
                                      name);
        }
      }
    }
  }

  cv::Size ResolveInputSize(cv::Size image) const {
    const auto left =
        engine_->getTensorShape(parameters_.left_input_name.c_str());
    const auto right =
        engine_->getTensorShape(parameters_.right_input_name.c_str());
    const auto resolve = [&](int original, int requested, int left_dim,
                             int right_dim) {
      int target = requested;
      if (target == 0 && left_dim > 0) {
        target = left_dim;
      }
      if (target == 0 && right_dim > 0) {
        target = right_dim;
      }
      if (target == 0) {
        const int64_t rounded = ((static_cast<int64_t>(original) +
                                  parameters_.padding_multiple - 1) /
                                 parameters_.padding_multiple) *
                                parameters_.padding_multiple;
        if (rounded <= 0 || rounded > std::numeric_limits<int>::max()) {
          throw std::invalid_argument(
              "TensorRT stereo padded image dimension overflow");
        }
        target = static_cast<int>(rounded);
      }
      if ((left_dim > 0 && target != left_dim) ||
          (right_dim > 0 && target != right_dim)) {
        throw std::invalid_argument(
            "Configured TensorRT stereo dimensions differ from the static "
            "engine input shapes");
      }
      return target;
    };
    return cv::Size(
        resolve(image.width, parameters_.input_width, left.d[3], right.d[3]),
        resolve(image.height, parameters_.input_height, left.d[2], right.d[2]));
  }

  struct Tensor {
    DeviceBuffer device;
    cv::Mat host;
    size_t bytes = 0;
  };
  Logger logger_;  // Must outlive all TensorRT objects.
  IgevParameters parameters_;
  S2m2Parameters s2m2_;
  bool is_s2m2_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  Stream stream_;
  std::map<std::string, Tensor> tensors_;
  std::mutex inference_mutex_;
};

class IgevMatcher::Impl : public TensorRtStereoRuntime {
 public:
  explicit Impl(const IgevParameters& p) : TensorRtStereoRuntime(p) {}
};

namespace {
IgevParameters MakeS2m2RuntimeParameters(const S2m2Parameters& p) {
  IgevParameters result;
  result.engine_path = p.engine_path;
  result.left_input_name = p.left_input_name;
  result.right_input_name = p.right_input_name;
  result.output_name = p.output_name;
  result.input_width = p.input_width;
  result.input_height = p.input_height;
  result.padding_mode = "s2m2";
  result.device_id = p.device_id;
  return result;
}
}  // namespace

class S2m2Matcher::Impl : public TensorRtStereoRuntime {
 public:
  explicit Impl(const S2m2Parameters& p)
      : TensorRtStereoRuntime(MakeS2m2RuntimeParameters(p), p, true) {}
};

IgevMatcher::IgevMatcher(const IgevParameters& parameters)
    : impl_(std::make_unique<Impl>(parameters)) {}
IgevMatcher::~IgevMatcher() = default;
cv::Mat IgevMatcher::Compute(const cv::Mat& left, const cv::Mat& right) {
  return impl_->Compute(left, right);
}
void IgevMatcher::ValidateInputSize(cv::Size size) const {
  impl_->ValidateInputSize(size);
}

S2m2Matcher::S2m2Matcher(const S2m2Parameters& parameters)
    : impl_(std::make_unique<Impl>(parameters)) {}
S2m2Matcher::~S2m2Matcher() = default;
cv::Mat S2m2Matcher::Compute(const cv::Mat& left, const cv::Mat& right) {
  return impl_->Compute(left, right);
}
void S2m2Matcher::ValidateInputSize(cv::Size size) const {
  impl_->ValidateInputSize(size);
}
}  // namespace stereo_recon

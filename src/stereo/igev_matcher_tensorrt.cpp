#include <msckf_vio/stereo/igev_processing.h>
#include <msckf_vio/stereo/stereo_matcher.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace msckf_vio::stereo {
static_assert(NV_TENSORRT_MAJOR == 10, "The IGEV backend requires TensorRT 10.x");
namespace {
void cudaCheck(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string("IGEV ") + operation + ": " + cudaGetErrorString(result));
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity > Severity::kWARNING) return;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (severity <= Severity::kERROR) last_error_ = message;
      std::cerr << "[IGEV TensorRT] " << message << '\n';
    } catch (...) {}  // TensorRT log callbacks must not throw.
  }
  std::string error(const std::string& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    return "IGEV " + operation + (last_error_.empty() ? "" : ": " + last_error_);
  }
 private:
  std::mutex mutex_;
  std::string last_error_;
};

class DeviceBuffer {
 public:
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void ensure(size_t bytes) {
    if (bytes <= capacity_) return;
    if (data_) cudaCheck(cudaFree(data_), "free device buffer");
    data_ = nullptr;
    capacity_ = 0;
    cudaCheck(cudaMalloc(&data_, bytes), "allocate device buffer");
    capacity_ = bytes;
  }
  void* data() const { return data_; }
 private:
  void* data_ = nullptr;
  size_t capacity_ = 0;
};

struct Stream {
  cudaStream_t value = nullptr;
  ~Stream() { if (value) cudaStreamDestroy(value); }
};

// On any inference exception, finish pending transfers before local host
// tensors can be released or reused on the next call.
struct StreamFence {
  cudaStream_t stream;
  ~StreamFence() { cudaStreamSynchronize(stream); }
};

int cvType(nvinfer1::DataType type) {
  if (type == nvinfer1::DataType::kFLOAT) return CV_32F;
  if (type == nvinfer1::DataType::kHALF) return CV_16F;
  throw std::invalid_argument("IGEV supports FP32/FP16 tensor I/O only");
}

size_t tensorBytes(const nvinfer1::Dims& dimensions, int type) {
  if (dimensions.nbDims < 1 || dimensions.nbDims > nvinfer1::Dims::MAX_DIMS)
    throw std::invalid_argument("IGEV encountered an unsupported tensor rank");
  size_t bytes = CV_ELEM_SIZE(type);
  for (int i = 0; i < dimensions.nbDims; ++i) {
    if (dimensions.d[i] <= 0 || dimensions.d[i] > std::numeric_limits<int>::max() ||
        bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimensions.d[i]))
      throw std::invalid_argument("IGEV output shape is unresolved or too large");
    bytes *= dimensions.d[i];
  }
  return bytes;
}
}  // namespace

class IgevMatcher::Impl {
 public:
  explicit Impl(const IgevParameters& parameters) : parameters_(parameters) {
    if (parameters.engine_path.empty() || parameters.device_id < 0 ||
        parameters.input_width < 0 || parameters.input_height < 0 ||
        parameters.padding_multiple <= 0 || parameters.left_input_name.empty() ||
        parameters.right_input_name.empty() || parameters.output_name.empty() ||
        parameters.left_input_name == parameters.right_input_name ||
        (parameters.padding_mode != "right_bottom" && parameters.padding_mode != "sintel"))
      throw std::invalid_argument("Invalid IGEV engine path, tensor names, dimensions, device or padding mode");
    std::ifstream file(parameters.engine_path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0)
      throw std::runtime_error("IGEV engine is missing or empty (conversion may still be running): " +
                               parameters.engine_path);
    const auto length = static_cast<size_t>(file.tellg());
    std::vector<char> serialized(length);
    file.seekg(0);
    if (!file.read(serialized.data(), static_cast<std::streamsize>(length)))
      throw std::runtime_error("Cannot read IGEV engine: " + parameters.engine_path);
    cudaCheck(cudaSetDevice(parameters.device_id), "select CUDA device");
    if (!initLibNvInferPlugins(&logger_, ""))
      throw std::runtime_error(logger_.error("failed to initialize TensorRT plugins"));
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error(logger_.error("failed to create runtime"));
    engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!engine_) throw std::runtime_error(logger_.error("failed to deserialize engine"));
    int inputs = 0;
    bool found_left = false, found_right = false, found_output = false;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const std::string name = engine_->getIOTensorName(i);
      const bool is_input = engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
      cvType(engine_->getTensorDataType(name.c_str()));
      if (engine_->getTensorFormat(name.c_str()) != nvinfer1::TensorFormat::kLINEAR ||
          engine_->getTensorLocation(name.c_str()) != nvinfer1::TensorLocation::kDEVICE)
        throw std::invalid_argument("IGEV requires linear device tensor I/O: " + name);
      if (is_input) {
        ++inputs;
        const auto shape = engine_->getTensorShape(name.c_str());
        if (shape.nbDims != 4 || (shape.d[0] != 1 && shape.d[0] != -1) ||
            (shape.d[1] != 3 && shape.d[1] != -1))
          throw std::invalid_argument("IGEV inputs must be NCHW [1,3,H,W]: " + name);
        for (int axis : {2, 3})
          if (shape.d[axis] != -1 && (shape.d[axis] <= 0 || shape.d[axis] > std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid IGEV input spatial dimensions: " + name);
        found_left |= name == parameters.left_input_name;
        found_right |= name == parameters.right_input_name;
      } else {
        found_output |= name == parameters.output_name;
      }
    }
    if (inputs != 2 || !found_left || !found_right || !found_output)
      throw std::invalid_argument("IGEV engine tensor names differ from the configured left/right/disparity names");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error(logger_.error("failed to create execution context"));
    cudaCheck(cudaStreamCreate(&stream_.value), "create CUDA stream");
    std::clog << "[IGEV] Loaded " << parameters.engine_path << " on CUDA device " << parameters.device_id
              << "; inputs=" << parameters.left_input_name << ',' << parameters.right_input_name
              << "; output=" << parameters.output_name << "; padding=" << parameters.padding_mode << '\n';
  }

  ~Impl() {
    // ROS constructs this object on the main thread but inference runs on its
    // worker thread; the CUDA device must be selected on both threads.
    cudaSetDevice(parameters_.device_id);
    if (stream_.value) cudaStreamSynchronize(stream_.value);
  }

  cv::Mat compute(const cv::Mat& left, const cv::Mat& right) {
    // One context/stream per matcher; serialize direct callers as well as ROS.
    std::lock_guard<std::mutex> lock(inference_mutex_);
    cudaCheck(cudaSetDevice(parameters_.device_id), "select inference CUDA device");
    const auto size = inputSize(left.size());
    const auto prepared = prepareIgevInput(left, right, size, parameters_.padding_mode);
    const nvinfer1::Dims4 dimensions(1, 3, size.height, size.width);
    for (const auto& name : {parameters_.left_input_name, parameters_.right_input_name}) {
      if (!context_->setInputShape(name.c_str(), dimensions))
        throw std::invalid_argument("IGEV input dimensions are outside the engine shape/profile: " + name);
    }
    if (context_->inferShapes(0, nullptr) != 0)
      throw std::runtime_error(logger_.error("could not resolve all engine tensor shapes"));
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const std::string name = engine_->getIOTensorName(i);
      auto& tensor = tensors_[name];
      const auto shape = context_->getTensorShape(name.c_str());
      const int type = cvType(engine_->getTensorDataType(name.c_str()));
      tensor.bytes = tensorBytes(shape, type);
      // TensorRT 10 dimensions are int64; OpenCV uses int dimensions.
      std::vector<int> cv_shape;
      for (int axis = 0; axis < shape.nbDims; ++axis)
        cv_shape.push_back(static_cast<int>(shape.d[axis]));
      tensor.host.create(shape.nbDims, cv_shape.data(), type);
      tensor.device.ensure(tensor.bytes);
      if (!context_->setTensorAddress(name.c_str(), tensor.device.data()))
        throw std::runtime_error(logger_.error("could not bind tensor " + name));
    }
    // Validate final output before enqueue; never guess a low-resolution
    // output's disparity scale or select an arbitrary output tensor.
    auto& output = tensors_.at(parameters_.output_name);
    const auto output_shape = context_->getTensorShape(parameters_.output_name.c_str());
    if (output_shape.nbDims < 2 || output_shape.nbDims > 4 ||
        output_shape.d[output_shape.nbDims - 2] != size.height ||
        output_shape.d[output_shape.nbDims - 1] != size.width)
      throw std::invalid_argument("IGEV engine must export final disparity at its input resolution");
    for (int i = 0; i < output_shape.nbDims - 2; ++i)
      if (output_shape.d[i] != 1)
        throw std::invalid_argument("IGEV engine must output one disparity image");

    auto& left_tensor = tensors_.at(parameters_.left_input_name);
    auto& right_tensor = tensors_.at(parameters_.right_input_name);
    prepared.left_blob.convertTo(left_tensor.host, left_tensor.host.type());
    prepared.right_blob.convertTo(right_tensor.host, right_tensor.host.type());
    StreamFence fence{stream_.value};
    for (auto* input : {&left_tensor, &right_tensor})
      cudaCheck(cudaMemcpyAsync(input->device.data(), input->host.data, input->bytes,
                                cudaMemcpyHostToDevice, stream_.value), "upload input");
    if (!context_->enqueueV3(stream_.value))
      throw std::runtime_error(logger_.error("enqueueV3 failed"));
    cudaCheck(cudaMemcpyAsync(output.host.data, output.device.data(), output.bytes,
                              cudaMemcpyDeviceToHost, stream_.value), "download disparity");
    cudaCheck(cudaStreamSynchronize(stream_.value), "synchronize inference");
    return restoreIgevDisparity(output.host, prepared);
  }

 private:
  cv::Size inputSize(cv::Size image) const {
    const auto left = engine_->getTensorShape(parameters_.left_input_name.c_str());
    const auto right = engine_->getTensorShape(parameters_.right_input_name.c_str());
    const auto resolve = [&](int original, int requested, int left_dim, int right_dim) {
      int target = requested;
      if (target == 0 && left_dim > 0) target = left_dim;
      if (target == 0 && right_dim > 0) target = right_dim;
      if (target == 0) {
        const int64_t rounded = ((int64_t(original) + parameters_.padding_multiple - 1) /
                                  parameters_.padding_multiple) * parameters_.padding_multiple;
        if (rounded <= 0 || rounded > std::numeric_limits<int>::max())
          throw std::invalid_argument("IGEV padded image dimension overflow");
        target = static_cast<int>(rounded);
      }
      if ((left_dim > 0 && target != left_dim) || (right_dim > 0 && target != right_dim))
        throw std::invalid_argument("Configured IGEV dimensions differ from the static engine input shapes");
      return target;
    };
    return cv::Size(resolve(image.width, parameters_.input_width, left.d[3], right.d[3]),
                    resolve(image.height, parameters_.input_height, left.d[2], right.d[2]));
  }

  struct Tensor { DeviceBuffer device; cv::Mat host; size_t bytes = 0; };
  Logger logger_;  // Must outlive all TensorRT objects.
  IgevParameters parameters_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  Stream stream_;
  std::map<std::string, Tensor> tensors_;
  std::mutex inference_mutex_;
};

IgevMatcher::IgevMatcher(const IgevParameters& parameters) : impl_(std::make_unique<Impl>(parameters)) {}
IgevMatcher::~IgevMatcher() = default;
cv::Mat IgevMatcher::compute(const cv::Mat& left, const cv::Mat& right) {
  return impl_->compute(left, right);
}
}  // namespace msckf_vio::stereo

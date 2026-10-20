# Only the dense stereo target depends on TensorRT; do not add the vendor
# template's examples, global flags or -ffast-math to the VIO project.
option(MSCKF_ENABLE_TENSORRT "Build the IGEV/S2M2 TensorRT 10.x stereo backends" OFF)

if(MSCKF_ENABLE_TENSORRT)
  set(TensorRT_ROOT "" CACHE PATH "TensorRT 10.x installation root")
  set(CUDA_ROOT "" CACHE PATH "CUDA Toolkit installation root")
  find_path(MSCKF_TENSORRT_INCLUDE NvInfer.h
    HINTS "${TensorRT_ROOT}/include" "$ENV{TensorRT_ROOT}/include"
          "$ENV{TENSORRT_ROOT}/include" /usr/include/x86_64-linux-gnu)
  find_library(MSCKF_NVINFER nvinfer
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64"
          "$ENV{TENSORRT_ROOT}/lib")
  find_library(MSCKF_NVINFER_PLUGIN nvinfer_plugin
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64"
          "$ENV{TENSORRT_ROOT}/lib")
  find_path(MSCKF_CUDA_INCLUDE cuda_runtime_api.h
    HINTS "${CUDA_ROOT}/include" "$ENV{CUDA_ROOT}/include"
          "$ENV{CUDA_PATH}/include" /usr/local/cuda/include)
  find_library(MSCKF_CUDART cudart
    HINTS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib"
          "${CUDA_ROOT}/targets/x86_64-linux/lib"
          "$ENV{CUDA_ROOT}/lib64" /usr/local/cuda/lib64)
  if(NOT MSCKF_TENSORRT_INCLUDE OR NOT MSCKF_NVINFER OR NOT MSCKF_NVINFER_PLUGIN
      OR NOT MSCKF_CUDA_INCLUDE OR NOT MSCKF_CUDART)
    message(FATAL_ERROR "TensorRT stereo needs TensorRT 10.x headers, nvinfer/nvinfer_plugin and CUDA cudart. "
      "Set -DTensorRT_ROOT=<installation> -DCUDA_ROOT=<toolkit>, or disable MSCKF_ENABLE_TENSORRT.")
  endif()
  target_sources(stereo_reconstruction_lib PRIVATE src/stereo/tensorrt_stereo_matcher.cpp)
  target_include_directories(stereo_reconstruction_lib PRIVATE
    ${MSCKF_TENSORRT_INCLUDE} ${MSCKF_CUDA_INCLUDE})
  target_link_libraries(stereo_reconstruction_lib PRIVATE
    ${MSCKF_NVINFER} ${MSCKF_NVINFER_PLUGIN} ${MSCKF_CUDART} Threads::Threads)
  get_filename_component(msckf_trt_library_dir "${MSCKF_NVINFER}" DIRECTORY)
  get_filename_component(msckf_cuda_library_dir "${MSCKF_CUDART}" DIRECTORY)
  # Preserve runtime discovery for SDKs installed outside the system loader paths.
  set_property(TARGET stereo_reconstruction_lib stereo_reconstruction_node APPEND PROPERTY
    INSTALL_RPATH "${msckf_trt_library_dir}" "${msckf_cuda_library_dir}")
  message(STATUS "TensorRT stereo backends enabled: ${MSCKF_NVINFER}")
else()
  target_sources(stereo_reconstruction_lib PRIVATE src/stereo/tensorrt_stereo_stub.cpp)
endif()

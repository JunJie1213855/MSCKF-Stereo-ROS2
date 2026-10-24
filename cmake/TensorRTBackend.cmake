# Only the dense stereo target depends on TensorRT; do not add the vendor
# template's examples, global flags or -ffast-math to the VIO project.
option(STEREO_RECON_ENABLE_TENSORRT "Build the IGEV/S2M2 TensorRT 10.x/11.x stereo backends" OFF)

if(STEREO_RECON_ENABLE_TENSORRT)
  set(TensorRT_ROOT "" CACHE PATH "TensorRT 10.x/11.x installation root")
  set(CUDA_ROOT "" CACHE PATH "CUDA Toolkit installation root")
  find_path(STEREO_RECON_TENSORRT_INCLUDE NvInfer.h
    HINTS "${TensorRT_ROOT}/include" "$ENV{TensorRT_ROOT}/include"
          "$ENV{TENSORRT_ROOT}/include" /usr/include/x86_64-linux-gnu)
  find_library(STEREO_RECON_NVINFER nvinfer
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64"
          "$ENV{TensorRT_ROOT}/lib" "$ENV{TensorRT_ROOT}/lib64"
          "$ENV{TENSORRT_ROOT}/lib" "$ENV{TENSORRT_ROOT}/lib64")
  find_library(STEREO_RECON_NVINFER_PLUGIN nvinfer_plugin
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64"
          "$ENV{TensorRT_ROOT}/lib" "$ENV{TensorRT_ROOT}/lib64"
          "$ENV{TENSORRT_ROOT}/lib" "$ENV{TENSORRT_ROOT}/lib64")
  find_path(STEREO_RECON_CUDA_INCLUDE cuda_runtime_api.h
    HINTS "${CUDA_ROOT}/include" "$ENV{CUDA_ROOT}/include"
          "$ENV{CUDA_PATH}/include" /usr/local/cuda/include)
  find_library(STEREO_RECON_CUDART cudart
    HINTS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib"
          "${CUDA_ROOT}/targets/x86_64-linux/lib"
          "$ENV{CUDA_ROOT}/lib64" /usr/local/cuda/lib64)
  if(NOT STEREO_RECON_TENSORRT_INCLUDE OR NOT STEREO_RECON_NVINFER OR NOT STEREO_RECON_NVINFER_PLUGIN
      OR NOT STEREO_RECON_CUDA_INCLUDE OR NOT STEREO_RECON_CUDART)
    message(FATAL_ERROR "TensorRT stereo needs TensorRT 10.x/11.x headers, nvinfer/nvinfer_plugin and CUDA cudart. "
      "Set -DTensorRT_ROOT=<installation> -DCUDA_ROOT=<toolkit>, or disable STEREO_RECON_ENABLE_TENSORRT.")
  endif()
  target_sources(stereo_recon_lib PRIVATE src/stereo/tensorrt_stereo_matcher.cpp)
  target_include_directories(stereo_recon_lib PRIVATE
    ${STEREO_RECON_TENSORRT_INCLUDE} ${STEREO_RECON_CUDA_INCLUDE})
  target_link_libraries(stereo_recon_lib PRIVATE
    ${STEREO_RECON_NVINFER} ${STEREO_RECON_NVINFER_PLUGIN} ${STEREO_RECON_CUDART} Threads::Threads)
  get_filename_component(stereo_recon_trt_library_dir "${STEREO_RECON_NVINFER}" DIRECTORY)
  get_filename_component(stereo_recon_cuda_library_dir "${STEREO_RECON_CUDART}" DIRECTORY)
  # Preserve runtime discovery for SDKs installed outside the system loader paths.
  set_property(TARGET stereo_recon_lib stereo_reconstruction_node APPEND PROPERTY
    INSTALL_RPATH "${stereo_recon_trt_library_dir}" "${stereo_recon_cuda_library_dir}")
  message(STATUS "TensorRT stereo backends enabled: ${STEREO_RECON_NVINFER}")
else()
  target_sources(stereo_recon_lib PRIVATE src/stereo/tensorrt_stereo_stub.cpp)
endif()

#!/usr/bin/env bash
# Source this file before configuring/building or running TensorRT stereo.
# Override these roots before sourcing when using another local installation.
export TensorRT_ROOT="${TensorRT_ROOT:-/home/ros/lib/TensorRT-11.3.0.99}"
export TENSORRT_ROOT="$TensorRT_ROOT"
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.1}"

if [ ! -f "$TensorRT_ROOT/include/NvInfer.h" ] ||
   [ ! -f "$CUDA_ROOT/include/cuda_runtime_api.h" ]; then
  printf '%s\n' 'TensorRT/CUDA headers missing; set TensorRT_ROOT and CUDA_ROOT.' >&2
  return 1 2>/dev/null || exit 1
fi

case ":${PATH:-}:" in
  *":$TensorRT_ROOT/bin:"*) ;;
  *) export PATH="$TensorRT_ROOT/bin:${PATH:-}" ;;
esac
case ":${PATH:-}:" in
  *":$CUDA_ROOT/bin:"*) ;;
  *) export PATH="$CUDA_ROOT/bin:${PATH:-}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$TensorRT_ROOT/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$TensorRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$CUDA_ROOT/lib64:"*) ;;
  *) export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

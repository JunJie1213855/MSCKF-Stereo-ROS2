# Local WSL TensorRT environment

The local SDK is `/home/ros/lib/TensorRT-11.3.0.99`; CUDA Toolkit is installed at
`/usr/local/cuda-12.1`. The project supports the named-tensor runtime API in
TensorRT 10.x and 11.x. It discovers headers/libraries from explicit CMake roots
and preserves the runtime library paths in installed reconstruction binaries.

```bash
source /opt/ros/humble/setup.bash
source /home/ros/rosws/msckf_stereo_ws/src/stereo_recon/scripts/setup_wsl_env.sh
cd /home/ros/rosws/msckf_stereo_ws
MAKEFLAGS=-j2 colcon build --packages-select stereo_recon --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DSTEREO_RECON_ENABLE_TENSORRT=ON -DBUILD_TESTING=ON \
  -DTensorRT_ROOT="$TensorRT_ROOT" -DCUDA_ROOT="$CUDA_ROOT"
source install/setup.bash
```

The environment script adds the SDK/CUDA binaries and shared libraries to
`PATH` and `LD_LIBRARY_PATH` without duplicating existing entries. The local
`/home/ros/.bashrc` sources `setup_wsl_env.sh` for new interactive shells. Override
`TensorRT_ROOT` and `CUDA_ROOT` before sourcing to use another installation.

A Python environment is available at `/home/ros/.venvs/msckf-trt11`, with the
SDK's Python 3.10 wheel installed locally. It exposes system packages; use `-s`
to avoid incompatible user-installed NumPy 2.x with ROS/OpenCV.

## Reproduce the GPU interface check

```bash
source src/stereo_recon/scripts/setup_wsl_env.sh
/home/ros/.venvs/msckf-trt11/bin/python -s \
  src/stereo_recon/test/build_tensorrt_smoke_engines.py \
  /tmp/msckf_validation/trt
STEREO_RECON_TRT_SMOKE_DIR=/tmp/msckf_validation/trt \
  colcon test --packages-select stereo_recon --return-code-on-test-failure
colcon test-result --verbose
```

The generated engines have deterministic arithmetic, with IGEV-style FP32
inputs and S2M2-style UINT8 inputs/quality outputs. C++ checks execute the real
project matchers: GPU transfers, `enqueueV3`, repeated inference, preprocessing,
padding/cropping and confidence filtering. Without `STEREO_RECON_TRT_SMOKE_DIR`, the
two GPU cases are skipped, allowing CPU-only test runs.

These are interface tests without trained weights. Real IGEV/S2M2 accuracy and
model-operator compatibility require their ONNX/engine files. Old serialized
engines may need rebuilding for this TensorRT version and GPU.

The installed CUDA Toolkit version and `nvidia-smi`'s CUDA driver capability
are different quantities. Record the actual numeric GPU checks when evaluating
this local SDK/toolkit combination; passing these checks does not establish
compatibility of every external model or CUDA operator.

## Local ROS transport

The local Fast DDS talker/listener check discovered endpoints without delivering
messages. Cyclone DDS delivered the small demo messages; large stereo images on
loopback required smaller UDP packets. `setup_wsl_env.sh` selects
`rmw_cyclonedds_cpp` and `config/cyclonedds_local.xml`: loopback-only unicast
peers, 1200-byte maximum UDP payload, 1024-byte DDS fragments and a 4–8 MiB receive
buffer. No system network settings or drivers were changed. ROS_LOCALHOST_ONLY
is set to zero because the XML explicitly restricts the interface to loopback.
This profile is for processes inside the same WSL instance; use another DDS
profile when connecting to cameras or ROS nodes on other machines.

Source this environment in every terminal running the bag, VIO, reconstruction
or RViz. Keep `ROS_DOMAIN_ID` consistent. Transport settings follow
[Cyclone DDS configuration](https://cyclonedds.io/docs/cyclonedds/latest/config/config_file_reference.html).

## Local verification results

- GPU: NVIDIA GeForce RTX 4060 Ti, 16 GiB; driver reported by nvidia-smi: 581.80.
- TensorRT Python import and GPU builder creation: 11.3.0.99 passed.
- Synthetic FP32 and UINT8 engine builds and trtexec inference: passed.
- Real project IGEV/S2M2 C++ matchers: both numerical GPU tests passed.
- Package tests: colcon reported 42 tests, zero failures/errors/skips.
- Frontend ROS smoke: all four detector/distribution combinations passed with
  the explicit local DDS configuration.

Logs and generated test engines are under `/tmp/msckf_validation`. These results
cover CUDA 12.1 runtime API calls used here with the installed SDK and GPU driver;
they do not claim vendor support for every SDK/toolkit combination. No trained
IGEV/S2M2 model was available during this check.

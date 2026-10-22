"""Build tiny deterministic IGEV/S2M2-shaped engines, without model weights.

These exercise the project's runtime contracts; they do not estimate depth.
Run with TensorRT's Python bindings and the SDK lib directory on the loader path.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt


def build_engine(destination, s2m2):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    shape = (1, 3, 64 if s2m2 else 32, 64)
    dtype = trt.uint8 if s2m2 else trt.float32
    left = network.add_input("input_left" if s2m2 else "left", dtype, shape)
    right = network.add_input("input_right" if s2m2 else "right", dtype, shape)
    if s2m2:
        left = network.add_cast(left, trt.float32).get_output(0)
        right = network.add_cast(right, trt.float32).get_output(0)
    difference = network.add_elementwise(
        left, right, trt.ElementWiseOperation.SUB).get_output(0)
    mean = network.add_reduce(
        difference, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    offset = np.full((1, 1, 1, 1), 8.0, dtype=np.float32)
    constant = network.add_constant(offset.shape, offset).get_output(0)
    disparity = network.add_elementwise(
        mean, constant, trt.ElementWiseOperation.SUM).get_output(0)
    disparity.name = "output_disp" if s2m2 else "disparity"
    network.mark_output(disparity)
    # Retain NumPy arrays until serialization finishes.
    quality_arrays = []
    if s2m2:
        for name, value in (("output_conf", 0.9), ("output_occ", 1.0)):
            array = np.full((1, 1, 64, 64), value, dtype=np.float32)
            if name == "output_conf":
                array[0, 0, 18, 7] = 0.0
            quality_arrays.append(array)
            output = network.add_constant(array.shape, array).get_output(0)
            output.name = name
            network.mark_output(output)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build {destination}")
    destination.write_bytes(bytes(serialized))
    print(f"Built {destination} ({destination.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"TensorRT {trt.__version__}")
    build_engine(args.output / "igev_smoke.engine", False)
    build_engine(args.output / "s2m2_smoke.engine", True)


if __name__ == "__main__":
    main()

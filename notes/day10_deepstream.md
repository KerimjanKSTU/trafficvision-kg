# Day 10 — NVIDIA DeepStream / GStreamer

## Environment

- Hardware: NVIDIA Jetson Orin Nano
- L4T: 36.5.0
- Ubuntu: 22.04
- DeepStream: 7.1
- CUDA Runtime: 12.6
- TensorRT: 10.3
- GStreamer: 1.20.3
- Container: nvcr.io/nvidia/deepstream:7.1-samples-multiarch

## Compatibility issue

DeepStream 7.1 container initially failed to use nvv4l2decoder on
L4T 36.5 with:

S_EXT_CTRLS for CUDA_GPU_ID failed

Host NVDEC was tested separately and worked correctly.

Container workaround:

AARCH64_IGPU=1

After this, hardware H.264 decoding inside the DeepStream
container successfully reached EOS.

## Reference pipeline

4 × H.264 1080p sources
→ NVDEC
→ nvstreammux
→ PGIE
→ nvtracker / NvDCF
→ SGIE VehicleType
→ SGIE VehicleMake
→ FakeSink

## Streammux

batch-size=4
live-source=0
width=1920
height=1080
batched-push-timeout=40000

## Primary inference

TrafficCamNet
TensorRT INT8
batch-size=4

4 sources → nvstreammux batch=4 → PGIE batch=4

## Tracker

nvtracker plugin

Low-level tracker:
NvDCF

Config:
config_tracker_NvDCF_perf.yml

tracker-width=960
tracker-height=544

## Secondary inference

VehicleTypes:
batch-size=16

VehicleMake:
batch-size=16

PGIE batch represents frames.
SGIE batch represents detected object crops.

## Performance

Headless configuration:
- tiled-display=0
- OSD=0
- FakeSink
- sync=0

Observed throughput:

Stream 0 ≈ 62.22 FPS
Stream 1 ≈ 62.22 FPS
Stream 2 ≈ 62.22 FPS
Stream 3 ≈ 62.22 FPS

Approximate aggregate source-frame throughput:
≈249 FPS

This benchmark must NOT be directly compared to the TrafficVision
4.81 FPS result because the models and pipeline stages are different.

## Main engineering conclusions

1. DeepStream is a GStreamer-based video analytics framework,
   not a replacement for TensorRT.

2. nvstreammux enables real multi-source batching.

3. TensorRT batch=4 now has practical meaning:
   four video streams are combined into one inference batch.

4. nvtracker is a tracking plugin; NvDCF, NvSORT,
   NvDeepSORT and IOU are possible tracker implementations/configs.

5. PGIE batches frames, while SGIE batches detected objects.

6. DeepStream propagates inference and tracking results as metadata.

7. Host-vs-container A/B testing was useful for isolating
   the NVDEC compatibility problem.

# HopperKV

### A Hopper-Native, Memory- and Compute-Optimized LLM Inference Engine

**HopperKV** is an experimental C++17/CUDA LLM inference engine focused on GPU memory management, attention, KV-cache optimization, and architecture-aware execution on NVIDIA GPUs, with NVIDIA Hopper as its primary target.

The project builds on **gemm-foundry**, a separate GPU matrix multiplication library supporting architecture-optimized GEMM implementations for NVIDIA Turing, Ampere, and Hopper. Rather than duplicating that work, HopperKV is designed to integrate the existing library and focus on the systems problems surrounding LLM inference.

The long-term goal is to build a low-level, measurable inference system that connects optimized matrix computation, attention, KV-cache management, GPU data movement, and autoregressive generation.

> **Project status:** Active development. Foundational CUDA abstractions and prototype kernels are being developed. The GEMM backend exists as a separate project; its integration into HopperKV, the attention and KV-cache subsystems, end-to-end inference, and Hopper-specific optimizations remain future milestones.

---

## Table of Contents

- [1. Vision and Motivation](#1-vision-and-motivation)
- [2. Project Scope](#2-project-scope)
- [3. System Architecture](#3-system-architecture)
- [4. Integration with gemm-foundry](#4-integration-with-gemm-foundry)
- [5. Technical Focus Areas](#5-technical-focus-areas)
- [6. Hopper-Specific Optimization](#6-hopper-specific-optimization)
- [7. KV-Cache and Paged Memory](#7-kv-cache-and-paged-memory)
- [8. Correctness and Numerical Accuracy](#8-correctness-and-numerical-accuracy)
- [9. Benchmarking and Evaluation](#9-benchmarking-and-evaluation)
- [10. Development Roadmap](#10-development-roadmap)
- [11. Planned Repository Structure](#11-planned-repository-structure)
- [12. Technology Stack](#12-technology-stack)
- [13. Build and Development Environment](#13-build-and-development-environment)
- [14. Current Implementation Status](#14-current-implementation-status)
- [15. Engineering Principles](#15-engineering-principles)
- [16. References](#16-references)
- [17. License](#17-license)

---

## 1. Vision and Motivation

High-performance LLM inference is not simply a matter of making matrix multiplication faster.

Inference performance depends on how computation, memory allocation, data movement, attention, and previously computed sequence state work together. Optimizing one kernel in isolation does not necessarily improve the performance of the complete system.

Important factors include:

- **Compute throughput:** How efficiently matrix operations use CUDA cores and Tensor Cores.
- **Memory bandwidth:** How model weights, activations, and attention state move through the GPU memory hierarchy.
- **Data movement:** How effectively memory transfers overlap with computation.
- **KV-cache management:** How keys and values are allocated, stored, addressed, and reused during token generation.
- **Attention execution:** How score computation, softmax, and weighted value accumulation are implemented without unnecessary memory traffic.
- **Numerical precision:** How precision choices affect speed, memory consumption, and output accuracy.
- **Runtime behavior:** How synchronization, kernel-launch overhead, workload shape, and resource utilization affect end-to-end latency.

These factors are tightly coupled. For example, an attention kernel may be efficient when its input is contiguous but perform differently when it must access cache blocks distributed across memory. Likewise, a fast GEMM implementation does not by itself solve KV-cache allocation or the scheduling of inference operations.

HopperKV treats these as interconnected systems problems.

### Central engineering question

**How can a CUDA inference engine combine an existing optimized GEMM backend with efficient attention, KV-cache management, and Hopper-aware data movement to improve real LLM inference workloads?**

The project will approach this question incrementally, using clear baselines, modular implementations, correctness tests, and reproducible measurements.

---

## 2. Project Scope

HopperKV is intended to become a single-GPU inference engine, initially emphasizing NVIDIA Hopper hardware such as the H100.

Its scope is divided into four areas.

### 2.1 Reuse optimized computation

HopperKV will use **gemm-foundry** for supported matrix multiplication operations rather than creating a competing production GEMM implementation.

This separates the concerns of optimized matrix computation from inference-specific execution and state management.

### 2.2 Develop inference-specific kernels

The project will focus its custom CUDA work on operations and data paths surrounding GEMM, including attention, softmax, KV-cache reads and writes, and efficient data movement.

These operations will be developed with an emphasis on memory behavior, synchronization, and integration with the inference pipeline.

### 2.3 Investigate Hopper-native execution

The project will explore TMA, WGMMA, asynchronous execution, and reduced-precision computation where those capabilities address a relevant bottleneck.

Hopper-specific mechanisms will be evaluated against simpler implementations. Their presence in the roadmap does not imply that they are already implemented.

### 2.4 Measure the complete system

The eventual objective is not merely to report fast individual kernels. It is to integrate the components into a working inference path and evaluate correctness, latency, throughput, memory usage, and scaling behavior.

---

## 3. System Architecture

HopperKV is intended to evolve into a layered inference system.

```text
                         HopperKV
                            |
             +--------------+--------------+
             |                             |
       Inference Runtime             Memory System
             |                             |
       Execution Pipeline           KV-Cache Manager
       Kernel Dispatch              Cache Allocation
       Synchronization              Layout and Indexing
             |                             |
             +--------------+--------------+
                            |
                    Inference Operations
                            |
              +-------------+-------------+
              |             |             |
           Attention      Softmax      Cache Ops
              |                           |
              +-------------+-------------+
                            |
                    Matrix Operations
                            |
                      gemm-foundry
                            |
              Architecture-Optimized GEMM
                 Turing / Ampere / Hopper
                            |
                       NVIDIA GPU
                            |
                Hopper-Specific Fast Paths
                   TMA / WGMMA / FP8
```

*Conceptual target architecture. The diagram describes the intended system, not a claim that every component is implemented.*

### Layer 1 — CUDA and tensor foundations

Provides reusable infrastructure for device memory ownership, tensor shape and stride metadata, memory transfers, error handling, and kernel interfaces.

These components establish explicit ownership and layout rules for data passed between operations.

### Layer 2 — Inference runtime

Will coordinate execution dependencies, kernel dispatch, synchronization, and the movement of data between inference operations.

The runtime should make it possible to select supported implementations and measure their effects under controlled conditions.

### Layer 3 — Inference-specific operations

Will contain custom attention, softmax, cache operations, and other necessary kernels.

The emphasis is on understanding the memory and execution costs of these operations rather than reimplementing functionality already provided by the GEMM backend.

### Layer 4 — GEMM backend

The matrix multiplication layer will be provided by the separate `gemm-foundry` project.

HopperKV will define the integration boundary between its inference operations and the GEMM library, including input and output layouts, supported data types, dimensions, and execution requirements.

### Layer 5 — Hopper-specific optimization

Selected operations may use Hopper-native data movement and compute mechanisms where appropriate.

The design must account for synchronization, resource requirements, supported layouts, and the relationship between individual kernel latency and the complete inference workload.

---

## 4. Integration with gemm-foundry

[https://github.com/sanjog1000/gemm-foundry](https://github.com/sanjog1000/gemm-foundry) is the separate GEMM library associated with this project.

It provides architecture-optimized matrix multiplication implementations targeting NVIDIA Turing, Ampere, and Hopper GPUs.

HopperKV is intended to **reuse this implementation instead of rebuilding the same production GEMM functionality**.

### Why keep the projects separate?

The two projects have different responsibilities.

| Responsibility                               | Project        |
| -------------------------------------------- | -------------- |
| Architecture-optimized GEMM implementations  | `gemm-foundry` |
| Matrix-operation dispatch within the library | `gemm-foundry` |
| LLM inference execution flow                 | HopperKV       |
| Attention and KV-cache integration           | HopperKV       |
| Cache allocation, indexing, and updates      | HopperKV       |
| Inference-specific memory management         | HopperKV       |
| End-to-end inference evaluation              | HopperKV       |
| Integration and system-level profiling       | HopperKV       |

This separation reduces duplicated implementation work and allows each repository to evolve independently.

It also makes the system easier to inspect: matrix computation belongs to one clearly identified component, while inference-specific behavior remains in HopperKV.

### Planned integration approach

The integration should use a clearly defined interface and a pinned version, tag, or commit of `gemm-foundry` so that experimental results can be reproduced against a known backend revision.

The exact build integration will depend on the library's exported CMake targets and public API.

The integration work will need to establish:

- Supported matrix dimensions, layouts, and data types.
- Input and output memory ownership.
- Device and stream compatibility.
- Error handling and failure behavior.
- Kernel selection and architecture support.
- Correctness tests for the integrated operations.
- Benchmarks comparing isolated GEMM performance with its contribution to end-to-end inference.

Small reference implementations may be retained for testing or comparison. They will not be developed as a second production GEMM backend.

**The key boundary is deliberate: `gemm-foundry` handles optimized GEMM; HopperKV investigates how to build an efficient inference system around it.**

---

## 5. Technical Focus Areas

### 5.1 GPU memory hierarchy

The memory hierarchy is central to the project.

| Memory level        | Role in HopperKV                                               |
| ------------------- | -------------------------------------------------------------- |
| Global memory / HBM | Stores model weights, activations, and KV-cache data           |
| L2 cache            | Can reduce repeated accesses to global memory                  |
| Shared memory       | Holds explicitly staged data for cooperating threads           |
| Registers           | Hold thread-local values, accumulators, and intermediate state |

The project will investigate coalesced accesses, memory layout, data reuse, synchronization, register pressure, and the cost of moving intermediate results between memory levels.

### 5.2 Attention computation

Attention combines several operations whose execution and memory costs depend on sequence length, head configuration, and workload shape.

Planned areas of investigation include:

- Query-key score computation.
- Numerically stable softmax.
- Attention-weighted value accumulation.
- Tiled attention algorithms.
- Reduced intermediate-memory traffic.
- Causal attention where required.
- Integration with the KV-cache during autoregressive decoding.

The objective is to understand the entire attention data path and select suitable algorithms for the workload.

### 5.3 Memory-efficient execution

The project will investigate avoiding unnecessary materialization of intermediate tensors and improving the reuse of values while they are available in fast on-chip storage.

The particular strategy will depend on the algorithm, precision, tensor shape, and hardware constraints.

### 5.4 Asynchronous data movement

High-throughput GPU workloads can benefit when memory transfers overlap with computation.

HopperKV will explore asynchronous copies, shared-memory staging, double buffering, and pipeline synchronization where they are useful for inference-specific kernels.

The objective is not to add asynchronous mechanisms indiscriminately, but to determine when their benefits justify their complexity.

---

## 6. Hopper-Specific Optimization

Hopper is the primary architectural target. The NVIDIA H100 is the main reference platform for evaluating Hopper-specific execution paths.

### Tensor Memory Accelerator (TMA)

TMA provides hardware-assisted multidimensional data movement, including asynchronous transfers between global memory and shared memory.

HopperKV will investigate how it may help stage attention operands and cache data, reduce per-thread data-movement overhead, and overlap transfers with useful computation.

A particular area of interest is whether TMA can be used effectively when the logical attention data does not reside in a single contiguous memory region.

### Warp-Group Matrix Multiply-Accumulate (WGMMA)

WGMMA provides Hopper-specific matrix operations using Tensor Cores and warp-group execution.

The project will investigate its requirements around operand layout, shared memory, accumulator handling, asynchronous completion, and synchronization where it is relevant to custom inference kernels.

GEMM operations already handled by `gemm-foundry` should remain within that library's responsibility unless a specific integration requirement justifies a different design.

### Pipeline optimization

The project will examine producer-consumer patterns, data staging, and execution overlap for inference-specific kernels.

Potential gains must be established experimentally; asynchronous execution can introduce synchronization costs and resource constraints.

### Reduced precision

FP16, BF16, and FP8 may offer different trade-offs between throughput, memory consumption, and numerical accuracy.

Planned experiments will consider supported formats, scaling, accumulation precision, and error against a reference implementation.

FP8 is not assumed to be suitable for every operation or model configuration.

**TMA, WGMMA, and FP8 remain optimization areas to investigate. Listing them here does not imply that HopperKV already implements or validates those paths.**

---

## 7. KV-Cache and Paged Memory

The KV-cache stores keys and values generated by previous tokens. Reusing this state avoids recomputing the full historical key and value projections at every autoregressive decoding step.

However, cache memory consumption grows with context length and the number of active sequences.

A useful estimate for dense KV-cache storage is:

2 L B S H\_{\mathrm{KV}} D_h b
]

where:

- (L): number of Transformer layers.
- (B): number of active sequences.
- (S): cached sequence length.
- (H\_{\mathrm{KV}}): number of key/value heads.
- (D_h): dimension per head.
- (b): bytes per stored element.
- The factor of two accounts for keys and values.

This estimate assumes uniform dimensions and sequence lengths. Actual memory usage can differ due to allocation overhead, padding, metadata, and the cache format.

### Planned KV-cache subsystem

HopperKV intends to implement a cache-management layer responsible for:

- Cache allocation and capacity tracking.
- Key and value layouts.
- Cache indexing and position mapping.
- Cache writes during token generation.
- Efficient reads during attention.
- Memory usage and correctness validation.

### Paged KV-cache investigation

Paged allocation divides cache storage into blocks instead of requiring every sequence to occupy one contiguous region.

A logical token position can be mapped through a block table to a physical block and an offset within that block.

This approach can improve allocation flexibility for variable-length sequences, but it introduces block-table accesses and potentially non-contiguous memory operations.

The project will investigate:

- The trade-off between block size and allocation overhead.
- Cache fragmentation and memory utilization.
- Address translation overhead.
- How attention kernels access paged storage.
- Whether Hopper-native data movement can improve access to non-contiguous cache data.
- The effect on throughput, latency, and memory consumption.

This is a planned research direction, not a claim of novelty. Existing inference systems already use optimized attention and paged cache techniques. Any future research contribution must be established through literature review, meaningful baselines, and measured results.

---

## 8. Correctness and Numerical Accuracy

An optimization is useful only if it preserves the intended computation.

HopperKV will use multiple levels of validation.

### Operation-level tests

Custom kernels should be compared against simpler reference implementations across representative dimensions, edge cases, and irregular shapes.

### Integration tests

The GEMM backend, attention operations, and cache subsystem must be tested together. Correct individual components do not guarantee a correct inference pipeline.

### Memory-safety checks

Invalid memory accesses, out-of-bounds indexing, synchronization errors, and resource-lifetime problems must be addressed before trusting performance measurements.

### Numerical validation

Floating-point calculations can differ due to rounding and changes in operation order. Tests should use justified tolerances and, where useful, report absolute error, relative error, and maximum error.

### Reproducible configurations

Tests should record their inputs, dimensions, precision, backend revision, and relevant environment information. Reference implementations should be kept available for comparison.

---

## 9. Benchmarking and Evaluation

HopperKV will distinguish between isolated kernel benchmarks and complete inference measurements.

### Component-level benchmarks

Depending on implementation status, these may measure:

- GEMM backend dispatch and integration overhead.
- Attention kernels.
- KV-cache reads and writes.
- Cache indexing and block-table operations.
- Memory movement and synchronization.
- Kernel-launch overhead.

### End-to-end benchmarks

As the inference runtime develops, evaluation will include:

- Time to first token (TTFT).
- Inter-token latency (ITL).
- Time per generated token.
- Throughput in tokens per second.
- Behavior across context lengths and batch sizes.
- KV-cache memory consumption.
- Peak GPU memory usage.

### Experimental controls

Reports should specify:

- GPU and architecture.
- CUDA Toolkit, driver, and compiler versions.
- Model and workload configuration.
- Batch size, context length, and generation length.
- Precision and cache configuration.
- Backend revision.
- Warm-up and measurement procedure.
- Correctness criteria and synchronization method.

Where possible, measurements should separate kernel execution time from end-to-end latency. Comparisons should use equivalent workloads and precision requirements.

**No speedup or performance advantage will be claimed without measurement and a documented baseline.**

---

## 10. Development Roadmap

### Phase 1 — CUDA foundations

- [x] Establish the initial repository structure.
- [x] Begin device-memory and tensor abstractions.
- [x] Begin tensor metadata and descriptor abstractions.
- [x] Develop prototype memory-access and matrix-operation kernels.
- [ ] Validate the complete foundation on suitable CUDA hardware.
- [ ] Establish repeatable correctness tests.

### Phase 2 — Integrate gemm-foundry

- [ ] Define the public interface between HopperKV and the GEMM backend.
- [ ] Establish a reproducible dependency revision.
- [ ] Integrate the backend into the CMake build.
- [ ] Validate supported layouts, types, dimensions, and device pointers.
- [ ] Add integration tests for GEMM operations used by the inference pipeline.
- [ ] Measure integration overhead and preserve suitable reference tests.

### Phase 3 — Attention and KV-cache

- [ ] Implement reference attention operations.
- [ ] Develop and validate custom attention kernels.
- [ ] Implement KV-cache allocation, indexing, reads, and writes.
- [ ] Validate autoregressive cache behavior.
- [ ] Measure cache and attention performance across relevant workload sizes.

### Phase 4 — Hopper-specific optimization

- [ ] Investigate TMA-based data movement for selected kernels.
- [ ] Evaluate WGMMA where it is appropriate and not already covered by the GEMM backend.
- [ ] Explore asynchronous pipelines and shared-memory staging.
- [ ] Evaluate reduced-precision paths and numerical error.
- [ ] Compare specialized implementations against verified baselines.

### Phase 5 — End-to-end inference

- [ ] Integrate model operations into a working single-GPU inference path.
- [ ] Validate generated outputs against a reference implementation.
- [ ] Measure TTFT, inter-token latency, throughput, and memory usage.
- [ ] Profile the complete pipeline.
- [ ] Document supported model configurations and known limitations.

### Phase 6 — Advanced systems and research

- [ ] Investigate paged KV-cache designs.
- [ ] Explore page-aware attention and Hopper-native data movement.
- [ ] Evaluate cache-management and workload-aware scheduling policies.
- [ ] Establish research questions and baselines through a literature review.
- [ ] Publish reproducible experimental results where justified.

The roadmap distinguishes planned work from validated results. Checkboxes should be updated only when each milestone has actually been completed.

---

## 11. Planned Repository Structure

The repository is expected to evolve toward the following organization. This is a target structure, not a claim that all files and directories currently exist.

```text
HopperKV/
├── CMakeLists.txt
├── README.md
├── include/
│   └── hopperkv/
│       ├── cuda_check.cuh
│       ├── cuda_buffer.cuh
│       ├── tensor.hpp
│       └── tensor_desc.cuh
├── src/
│   ├── main.cu
│   ├── runtime/
│   ├── memory/
│   ├── kernels/
│   │   ├── attention/
│   │   ├── kv_cache/
│   │   └── utilities/
│   └── benchmarks/
├── tests/
│   ├── unit/
│   ├── numerical/
│   └── integration/
├── benchmarks/
│   ├── microbenchmarks/
│   └── inference/
├── docs/
│   ├── architecture.md
│   ├── design-decisions.md
│   └── benchmarking.md
└── scripts/
    └── benchmark/
```

`gemm-foundry` remains a separate repository and is intended to be consumed as a pinned dependency rather than copied into HopperKV as a duplicate implementation.

The final build configuration will depend on the dependency's actual public API and exported CMake targets.

---

## 12. Technology Stack

| Component                      | Technology                                                          |
| ------------------------------ | ------------------------------------------------------------------- |
| Primary language               | C++17                                                               |
| GPU programming                | CUDA C++                                                            |
| Build system                   | CMake                                                               |
| Existing GEMM backend          | `gemm-foundry`                                                      |
| Target GPU architectures       | NVIDIA Turing, Ampere, and Hopper where supported                   |
| Primary optimization target    | NVIDIA Hopper / H100                                                |
| Attention and cache operations | Custom CUDA implementations planned                                 |
| Hopper data movement           | TMA investigation                                                   |
| Hopper matrix operations       | WGMMA investigation where appropriate                               |
| Numerical formats              | FP32 reference paths, FP16/BF16, experimental FP8 where appropriate |
| Validation                     | Reference implementations, numerical tests, integration tests       |
| Performance analysis           | CUDA events and NVIDIA profiling tools                              |
| Development workflow           | Git, GitHub, reproducible experiments                               |

Hopper-specific functionality will be isolated where practical so that generic CUDA components and compatible backend paths can be tested on supported NVIDIA architectures.

---

## 13. Build and Development Environment

### Prerequisites

A CUDA development environment generally requires:

- A supported NVIDIA GPU and compatible driver.
- NVIDIA CUDA Toolkit and `nvcc`.
- A supported C++17 toolchain.
- CMake.
- The appropriate version of `gemm-foundry` once the backend integration is configured.

Hopper-specific experiments require compatible hardware and software support. Testing on an earlier architecture does not establish Hopper-specific correctness or performance.

### Build configuration

Once the dependency and targets are wired into the project, the conventional CMake workflow is:

```bash
cmake -S . -B build
cmake --build build --config Release
```

These commands describe the intended build workflow; they do not imply that the current repository is already configured for a complete build.

The exact dependency setup will be documented after the library's public CMake interface and integration method have been finalized.

### Development without a local NVIDIA GPU

Host-side abstractions, interfaces, reference calculations, test design, and code review can progress without local GPU access.

However, device-side correctness, race detection, architecture-specific behavior, compiler validation, and performance claims must be verified on suitable NVIDIA hardware.

---

## 14. Current Implementation Status

HopperKV is being developed incrementally. Its current foundation consists of initial abstractions and prototype kernels, not a complete inference runtime.

| Component                                | Status                                                                                         |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- |
| HopperKV repository and build scaffold   | Initial                                                                                        |
| CUDA error handling                      | Initial implementation                                                                         |
| Device-memory RAII abstraction           | Initial implementation                                                                         |
| Tensor shape and stride metadata         | Initial implementation                                                                         |
| Tensor device descriptor                 | Initial implementation                                                                         |
| Memory-access experiments                | Prototype                                                                                      |
| Transpose experiments                    | Prototype                                                                                      |
| Reference GEMM experiments               | Prototype; testing/comparison only                                                             |
| `gemm-foundry` library                   | Existing separate project; supports architecture-optimized GEMM for Turing, Ampere, and Hopper |
| `gemm-foundry` integration into HopperKV | Planned                                                                                        |
| Custom attention subsystem               | Planned                                                                                        |
| KV-cache subsystem                       | Planned                                                                                        |
| Paged KV-cache                           | Planned                                                                                        |
| TMA/WGMMA inference-specific paths       | Planned investigation                                                                          |
| End-to-end LLM inference                 | Planned                                                                                        |
| Validated end-to-end benchmark suite     | Planned                                                                                        |

Prototype code has not yet been fully validated on GPU hardware. The existence of a separate GEMM library does not imply that it has already been integrated into HopperKV.

**Implemented, integrated, compiled, tested, benchmarked, and optimized are distinct milestones.** The project will document these milestones separately.

---

## 15. Engineering Principles

### Reuse instead of duplicating

Use `gemm-foundry` for its intended matrix-computation responsibility. Focus HopperKV's implementation effort on the inference-specific work around it.

### Correctness before optimization

Every optimization must preserve the intended computation and pass relevant tests.

### Measure before claiming

Theoretical throughput and architectural knowledge help form hypotheses. Controlled experiments establish actual performance.

### Keep reference paths

Simple reference operations remain useful for correctness checks and comparisons even when production computation is delegated to optimized backends.

### Make memory behavior explicit

Layouts, strides, allocation, cache indexing, transfers, and synchronization are core design concerns.

### Optimize the complete system

An isolated kernel speedup is only one part of the picture. The ultimate objective is better behavior across realistic inference workloads.

### Document limitations honestly

Unimplemented features, untested paths, hardware requirements, numerical trade-offs, and measurement limitations should remain visible.

---

## 16. References

1. **NVIDIA Hopper Architecture**
   [NVIDIA Hopper overview](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) — GPU architecture and platform capabilities.

2. **NVIDIA Hopper Architecture In-Depth**
   [NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — Hopper execution features and data movement.

3. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**
   [Research paper](https://arxiv.org/abs/2407.08608) — attention optimization, asynchronous execution, and low-precision computation.

4. **FlashAttention**
   [Official repository](https://github.com/Dao-AILab/flash-attention) — reference implementations and documentation for memory-efficient attention.

5. **NVIDIA CUTLASS**
   [Official repository](https://github.com/NVIDIA/cutlass) — GPU matrix-operation abstractions and architecture-specific optimization techniques.

These resources provide reference points for learning, comparison, and identifying meaningful engineering problems. HopperKV does not claim to outperform existing systems without supporting evidence.

---

## 17. License

The license should be specified in a `LICENSE` file. Until that file is added, the licensing status of HopperKV should be treated as **not yet specified**.

---

**HopperKV is a work in progress.** Its objective is to build a coherent, reproducible LLM inference system by combining an existing architecture-optimized GEMM library with inference-specific CUDA kernels, KV-cache management, memory-aware execution, and Hopper-oriented optimization.

# Performance Evaluation of Matrix Multiplication

## Project Overview
This repository contains the implementation and analysis of various matrix multiplication algorithms to study the effect of memory hierarchy and multi-core execution on processor performance. The project is divided into two primary parts: single-core performance evaluation (comparing C++ and Rust) and multi-core performance evaluation (using OpenMP in C++).

The computational complexity for all implemented matrix multiplication algorithms is $2n^3$ FLOPs for matrices of size $n\times n$.

## Single-Core Performance

This section analyzes the impact of memory access patterns and cache utilization on a single CPU core.

### Implemented Algorithms

1.  **Basic Algorithm (BMMA):** The naive approach. Iterates in the mathematical order (rows $\times$ columns), resulting in poor spatial locality for the second matrix.
2.  **Line-by-Line Algorithm (LLA):** A cache-friendly reformulation (row-by-row) that dramatically reduces cache misses by accessing memory sequentially.
3.  **Block-Oriented Algorithm (BOA):** A cache-aware algorithm that divides matrices into sub-blocks to maximize data reuse within L1/L2 cache levels.

### Metrics and Tooling

* **Execution Time:** Measured for matrices from $1024\times 1024$ to $10240\times 10240$.
* **Hardware Counters:** Linux `perf` was used to track `mem_load_retired.l1_miss` and `mem_load_retired.l2_miss`.
* **Normalized Misses:** To compare across scales, we used the formula: $Normalised\ Misses = \frac{L1\ or\ L2\ Misses}{n^3}$.

---

## Part 2: Multi-Core Performance

We scaled the analysis to a multi-core architecture using the FEUP university machines (Intel® Core™ i7-14700T).

### Parallel Implementations

1.  **Parallel Outer Loop:** Distributes the $i$-loop. While it reduces work per thread, it does not fix the underlying memory bottleneck of the basic algorithm.
2.  **Parallel Inner Loop:** **Warning:** This implementation is incorrect as it introduces race conditions on the results matrix and significant synchronization overhead.
3.  **Parallel Line:** Parallelizes the $i$-loop of the cache-friendly `ikj` algorithm. This was our most successful strategy.

### Advanced Directives
For the $8192\times 8192$ matrix, we explored:
* `#pragma omp simd`: Adds AVX2 vectorization to thread parallelism, providing a ~15-18% boost.
* `#pragma omp parallel for collapse(2)`: Merges loops to expose more parallelism, though it can occasionally break spatial locality.

---

## Results and Analysis

### 1. The Power of Locality
The jump from the Basic Algorithm to the Line-by-Line Algorithm was the most significant optimization, yielding a **~17.5x speedup** in C++ at $n=1024$. This is directly attributed to reducing the normalized L1 cache miss rate from near **1.00** to roughly **0.06**.

### 2. Language Comparison (C++ vs. Rust)
* **C++** generally outperformed Rust in optimized variants (LLA/BOA) due to GCC’s aggressive auto-vectorization with `-O3 -march=native`.
* **Rust Safe** was limited by runtime bounds checks, which prevented the compiler from using SIMD instructions.
* **Rust Unsafe** (using `get_unchecked`) recovered some performance but did not consistently beat C++.

### 3. Multi-Core Scaling and the "Topology Ceiling"
Performance did not scale linearly with thread count. We observed a plateau beyond **12–16 threads**. This is caused by:
* **Memory Bandwidth Saturation:** The $n=8192$ matrix (~1.5 GB) far exceeds cache, bottlenecking at the RAM.
* **Heterogeneous Architecture:** The test CPU uses a mix of P-cores (Performance) and E-cores (Efficiency). Once the workload spills onto E-cores, per-thread throughput drops.

---

## Benchmarking & Automation

To ensure consistency across all experimental runs, we provided automation scripts that handle compilation, execution, and data collection via `perf`.

### Single-Core Batch Testing
This script executes the entire single-core suite (Basic and Line-by-Line) for both C++ and Rust. It iterates through the required matrix sizes ($1024$ to $3072$) and logs hardware counters for L1 and L2 misses.

```bash
# Run from the project root (assign1)
# One must ensure that the perf line for the intened CPU (Intel or AMD) is uncommented and the undesired one is commented
./testing_unsafe_rs.sh 
```
*Note: This script automatically detects CPU architecture features (e.g., AVX2) to optimize the C++ and rust variants accordingly*

### Multi-Core Scalability Testing
This script automates the multi-threaded benchmarks for the Line-by-Line algorithm. It tests the matrix size $8192 \times 8192$ across the full range of thread counts ($4, 8, 12, 16, 20, 24$).

```bash
# Run from the project root (assign1)
./multi_thread.sh 
```

### Data Outputs
After running these scripts, the raw data will be populated in the `results/` directory as log files. These files serve as the primary data source for the analysis, tables and graphs provided in the final report.

---


## Compilation and Execution

### C++ Implementations
```bash
# Basic compilation
g++ -O2 -fopenmp matrixproduct.cpp -o matrixproduct

# Optimized Line/Block variants
g++ -O3 -fno-plt -flto -march=native -fopenmp matrix_line.cpp -o matrix_line
```

### Rust Implementations
```bash
cargo build --release
```

### Profiling with Perf
To collect the cache miss data used in our analysis:
```bash
perf stat -e cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.l2_miss/ ./bin/matrix_executable
```

---

## Key Conclusions
Performance optimization is fundamentally hierarchical:
1.  **Memory Access Patterns First:** No amount of threads can fix a cache-unfriendly algorithm.
2.  **Compiler/Language Second:** Once locality is fixed, SIMD and language overheads (like bounds checks) become the next bottleneck.
3.  **Parallelism Last:** Multi-core execution is only effective once the previous two layers are optimized, and even then, it is constrained by hardware topology and memory bandwidth.

---




This updated version of your README reintegrates the **Prerequisites** section, using technical details from your hardware setup and the tools specified in your report. I’ve also included a "Troubleshooting" subsection for `perf` permissions to ensure others can replicate your results without hits.

---

## Prerequisites

To compile and run the programs and replicate the benchmarks, ensure your environment meets the following requirements:


### Software Requirements
* **C++ Compiler:** `g++` (GCC) with support for OpenMP and C++11 or higher.
* **Rust Toolchain:** `rustc` and `cargo` (Edition 2024 was used for this study).
* **Profiling Tool:** Linux `perf` utilities.

### Troubleshooting `perf` Permissions
Accessing hardware performance counters requires specific kernel permissions. If `perf stat` fails, run the following command to allow profiling:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```
*Note: This setting will reset after a reboot unless added to `/etc/sysctl.conf`.*

---

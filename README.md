# Performance Evaluation of Matrix Multiplication

A comparative performance and scalability study of single-core and multi-core matrix multiplication ($n \times n$), analyzing the impact of CPU cache hierarchies, compiler auto-vectorization, programming language runtime overhead (C++ vs. Safe/Unsafe Rust), and OpenMP parallelization strategies.

---

## Overview

Matrix multiplication ($C = A \times B$) has a deterministic computational complexity of $2n^3$ floating-point operations. Because arithmetic complexity is identical across standard algorithmic formulations, all throughput variations derive directly from memory access patterns, hardware cache locality, compiler vectorization, and thread orchestration.

This project investigates:

1. **Memory Locality & Cache Hierarchy:** Analyzing cache misses (L1, L2) and data reuse across naive, row-major line traversal, and cache-blocked implementations.
2. **Language & Compiler Efficiency:** Benchmarking C++ (GCC with AVX2 SIMD) against Safe Rust (with bounds checking) and Unsafe Rust (`get_unchecked`).
3. **Multi-Core Scalability:** Evaluating thread distribution strategies, OpenMP worksharing directives, SIMD hints, loop collapsing, and scaling limits on heterogeneous architectures (Intel Raptor Lake P-core / E-core).

---

## Key Performance Results

| Optimization Level | Strategy | Representative Throughput | Primary Bottleneck / Speedup Driver |
| --- | --- | --- | --- |
| **Baseline (BMMA)** | Naive $ijk$ loop order | $\sim 0.3 - 0.9$ GFlop/s | Column-stride memory traversal on matrix $B$; L1 miss rates $> 85\%$. |
| **Spatial Locality (LLA)** | Loop reordering to $ikj$ | $\sim 6.6 - 16.6$ GFlop/s | **$\sim 17.5\times$ speedup**; consecutive row access enables hardware prefetching and AVX2 SIMD. |
| **Cache Blocking (BOA)** | Block tiling ($b = 128, 256, 512$) | $\sim 12.0 - 17.2$ GFlop/s | **$\sim 2\times$ speedup** over LLA at large matrices ($n \ge 8192$); working sets fit within 2 MB L2 cache. |
| **Language Dynamics** | C++ vs. Rust Safe vs. Rust Unsafe | C++ ($16.6$) > Rust Unsafe ($6.5$) > Rust Safe ($3.4$) | Rust safe runtime bounds checks inhibit SIMD vectorization; unsafe recovers partial vectorization. |
| **Multi-Core (OpenMP)** | Parallel Line + SIMD (24 threads) | $\sim 23.3$ GFlop/s ($n = 8192$) | Sub-linear scaling plateauing beyond 12–16 threads due to memory bandwidth limits and P/E-core topology. |

---

## Repository Structure

```text
.
├── doc/                        # Technical report (PDF and LaTeX source)
├── graphImgs/                  # Performance, cache-miss, and speedup comparison plots
├── results/
│   ├── machineSpecs.txt        # Benchmark machine hardware configuration
│   ├── pythonGraphs/           # Matplotlib scripts for data visualization
│   ├── tests_logs/             # Execution and Linux perf hardware counter raw logs
│   ├── single_core_test_script.sh
│   └── multi_core_test_script.sh
├── src/
│   ├── singleCore.cpp          # C++ single-core implementations (Basic, Line, Block)
│   ├── multiCore.cpp           # C++ multi-core OpenMP implementations
│   └── matrix_mult/            # Rust implementations
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs         # Safe Rust implementations
│           └── main_unsafe.rs  # Unsafe Rust implementations
├── multi_thread.sh             # Benchmark automation for multi-core scaling
├── testing_unsafe_rs.sh        # Benchmark automation for single-core comparisons
└── README.md

```

---

## Experimental Setup

* **CPU:** Intel® Core™ i7-14700T (20 physical cores: 8 P-cores up to 5.2 GHz + 12 E-cores; 28 total threads)
* **Caches:**
* L1 Data: 48 KB per P-core / 32 KB per E-core
* L2: 2 MB per P-core / 2 MB per 4 E-cores
* L3: 33 MB shared


* **RAM:** 32 GiB DDR5
* **OS:** Ubuntu 24.04 LTS (Linux kernel 6.17)
* **Compilers:** GCC 13+ (`-O3 -march=native -flto -fopenmp`), Rust 2024 Edition (`--release`)
* **Profiling:** Linux `perf` (`mem_load_retired.l1_miss`, `mem_load_retired.l2_miss`)

---

## Prerequisites

Ensure all required compilers and profilers are installed:

```bash
# C++ toolchain and OpenMP
sudo apt update
sudo apt install build-essential linux-tools-common linux-tools-generic

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python dependencies for graphing
pip install matplotlib numpy pandas

```

### Linux `perf` Event Permissions

Hardware performance counter collection requires elevated privileges:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1

```

*To persist across reboots, add `kernel.perf_event_paranoid = -1` to `/etc/sysctl.conf`.*

---

## Build & Manual Execution

### 1. C++ Implementations

* **Single-Core (Line and Block variants):**
```bash
g++ -O3 -fno-plt -flto -march=native src/singleCore.cpp -o bin_single_core
./bin_single_core

```


* **Multi-Core (OpenMP):**
```bash
g++ -O3 -fno-plt -flto -march=native -fopenmp src/multiCore.cpp -o bin_multi_core
./bin_multi_core

```



### 2. Rust Implementations

From the Rust workspace root:

```bash
cd src/matrix_mult

# Safe implementation
cargo build --release --bin matrix_mult
./target/release/matrix_mult

# Unsafe implementation (eliding bounds checks via get_unchecked)
cargo build --release --bin main_unsafe
./target/release/main_unsafe

```

### 3. Profiling Hardware Counters with `perf`

```bash
perf stat -e cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.l2_miss/ ./bin_single_core

```

---

## Automated Benchmarking & Visualization

The repository includes automation scripts that orchestrate compilation, loop over matrix dimensions ($n = 1024$ through $10240$), capture execution times, query hardware performance counters, and export formatted logs into `results/tests_logs/`:

* **Single-Core Test Suite:**
```bash
chmod +x testing_unsafe_rs.sh ./results/single_core_test_script.sh
./testing_unsafe_rs.sh

```


* **Multi-Core Scalability Suite ($n = 8192$, 4 to 24 threads):**
```bash
chmod +x multi_thread.sh ./results/multi_core_test_script.sh
./multi_thread.sh

```


* **Regenerating Report Graphs:**
```bash
cd results/pythonGraphs
python3 graphGen.py
python3 graphGenOverlap.py

```


Generated plots will be saved directly into `graphImgs/`.

---

## Architecture Insights

1. **Memory Access Order Dominates Complexity:** Moving from naive $ijk$ to $ikj$ traversal drops normalized L1 miss rates from $> 0.85$ to under $0.08$. Algorithmic memory alignment provides an order-of-magnitude greater speedup than upgrading compiler flags or adding thread parallelism to unoptimized access patterns.
2. **Compiler Vectorization vs. Language Safety:** GCC auto-vectorizes continuous inner loops using AVX2 registers. Safe Rust enforces runtime slice bounds checks that inhibit full vector loop vectorization. Bypassing bounds checks via `get_unchecked` restores partial vectorization, doubling throughput.
3. **The Topology Ceiling in Multi-Threading:** Beyond 12–16 threads on hybrid multi-core CPUs, scaling degrades due to:
* **Memory bandwidth saturation:** Large matrices ($n = 8192$, $\sim 1.5$ GB) continuously evict cache and saturate shared memory bus channels.
* **Core asymmetry:** Work dispatched to Efficiency cores (E-cores) exhibits lower IPC, causing thread synchronization delays across the thread barrier.

#!/bin/bash

mkdir -p bin
mkdir -p results

#PERF_COUNTERS="L1-dcache-load-misses,l2_cache_misses_from_dc_misses" #AMD
PERF_COUNTERS="mem_load_retired.l1_miss,mem_load_retired.l2_miss" #INTEL
CPP_SRC="src/LabCode.cpp"
RUST_SRC="src/matrix_mult/src/main.rs"
RUST_UNSAFE_SRC="src/matrix_mult/src/main_unsafe.rs"

echo "================================================="
echo " Compiling Source Files"
echo "================================================="
#g++ -O2 "$CPP_SRC" -o bin/cpp_basic
g++ -O3 -fno-plt -flto -march=native "$CPP_SRC" -o bin/cpp_opt

#rustc -C opt-level=2 "$RUST_SRC" -o bin/rust_basic
rustc -C opt-level=3 -C lto=fat -C target-cpu=native "$RUST_SRC" -o bin/rust_opt

#rustc -C opt-level=2 "$RUST_UNSAFE_SRC" -o bin/rust_unsafe_basic
rustc -C opt-level=3 -C lto=fat -C target-cpu=native "$RUST_UNSAFE_SRC" -o bin/rust_unsafe_opt
echo -e "\nCompilation complete.\n"

run_test() {
    local lang=$1
    local algo=$2
    local bin_path=$3
    local size=$4
    local block_size=$5
    local menu_opt=$6
    local log_file=$7

    echo "-------------------------------------------------" | tee -a "$log_file"
    if [ -z "$block_size" ]; then
        echo "Testing: $lang | $algo | Size: $size x $size" | tee -a "$log_file"
        INPUT_SEQ="$menu_opt\n$size\n0\n"
    else
        echo "Testing: $lang | $algo | Size: $size x $size | Block: $block_size" | tee -a "$log_file"
        INPUT_SEQ="$menu_opt\n$size\n$block_size\n0\n"
    fi
    echo "-------------------------------------------------" | tee -a "$log_file"

    echo -e "$INPUT_SEQ" | perf stat -e "$PERF_COUNTERS" "$bin_path" >> "$log_file" 2>&1
    echo -e "\n" >> "$log_file"
}

LOG_BASIC="results/basic_log.txt"
> "$LOG_BASIC"
echo "Starting Basic Algorithm Tests... (Saving to $LOG_BASIC)"
for size in $(seq 1024 512 3072); do
    run_test "C++" "Basic" "./bin/cpp_basic" "$size" "" "1" "$LOG_BASIC"
    run_test "Rust" "Basic" "./bin/rust_basic" "$size" "" "1" "$LOG_BASIC"
    run_test "Rust-Unsafe" "Basic" "./bin/rust_unsafe_basic" "$size" "" "1" "$LOG_BASIC"
done

LOG_LINE="results/line_log.txt"
> "$LOG_LINE"
echo "Starting Line-by-Line Algorithm Tests... (Saving to $LOG_LINE)"
for size in $(seq 1024 512 3072); do
    run_test "C++" "Line-by-Line" "./bin/cpp_opt" "$size" "" "2" "$LOG_LINE"
    run_test "Rust" "Line-by-Line" "./bin/rust_opt" "$size" "" "2" "$LOG_LINE"
    run_test "Rust-Unsafe" "Line-by-Line" "./bin/rust_unsafe_opt" "$size" "" "2" "$LOG_LINE"
done
for size in $(seq 4096 2048 10240); do
    run_test "C++" "Line-by-Line" "./bin/cpp_opt" "$size" "" "2" "$LOG_LINE"
    run_test "Rust" "Line-by-Line" "./bin/rust_opt" "$size" "" "2" "$LOG_LINE"
    run_test "Rust-Unsafe" "Line-by-Line" "./bin/rust_unsafe_opt" "$size" "" "2" "$LOG_LINE"
done

LOG_BLOCK="results/block_log.txt"
> "$LOG_BLOCK"
echo "Starting Block-Oriented Algorithm Tests... (Saving to $LOG_BLOCK)"
for size in $(seq 4096 2048 10240); do
    for block in 128 256 512; do
        run_test "C++" "Block-Oriented" "./bin/cpp_opt" "$size" "$block" "3" "$LOG_BLOCK"
        run_test "Rust" "Block-Oriented" "./bin/rust_opt" "$size" "$block" "3" "$LOG_BLOCK"
        run_test "Rust-Unsafe" "Block-Oriented" "./bin/rust_unsafe_opt" "$size" "$block" "3" "$LOG_BLOCK"
    done
done

echo "================================================="
echo "All tests complete! Check the 'results' folder."

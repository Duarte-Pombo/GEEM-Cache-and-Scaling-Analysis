#!/bin/bash

CPP_SRC="src/multiCore.cpp"
BIN_PATH="bin/multiCore"

echo "================================================="
echo " Compiling Multi-Core Source File"
echo "================================================="

# Compiling with OpenMP and O2 optimization as requested in the source code comments
g++ -fopenmp -O2 "$CPP_SRC" -o "$BIN_PATH"
echo -e "\nCompilation complete.\n"

# Function to run a specific test case
run_test() {
    local desc=$1
    local menu_opt=$2
    local size=$3
    local threads=$4
    local log_file=$5

    echo "-------------------------------------------------" | tee -a "$log_file"
    echo "Testing: $desc | Size: ${size}x${size} | Threads: $threads" | tee -a "$log_file"
    echo "-------------------------------------------------" | tee -a "$log_file"

    # Set the number of OpenMP threads
    export OMP_NUM_THREADS=$threads
    
    # Construct the input sequence for the C++ program: Option -> Size -> Exit (0)
    INPUT_SEQ="$menu_opt\n$size\n0\n"

    # Pipe the input sequence directly into the executable
    echo -e "$INPUT_SEQ" | "$BIN_PATH" >> "$log_file" 2>&1
    
    echo -e "\n" >> "$log_file"
}

# -------------------------------------------------------------------------
# Part 2, Step 1: Thread count = 4, Sizes = 1024 to 3072 (Increments of 512)
# -------------------------------------------------------------------------
LOG_P1="results/part2_step1_log.txt"
> "$LOG_P1"
echo "Starting Part 2, Step 1 Tests (Threads: 4, Sizes: 1024-3072)... (Saving to $LOG_P1)"

for size in $(seq 1024 512 3072); do    
    # Parallel runs (4 threads)
    run_test "Parallel Outer Loop (Menu 2)" 2 "$size" 4 "$LOG_P1"
    run_test "Parallel Inner Loop (Menu 3)" 3 "$size" 4 "$LOG_P1"
    run_test "Parallel Line (Menu 5)" 5 "$size" 4 "$LOG_P1"
done

# -------------------------------------------------------------------------
# Part 2, Step 2: Thread counts = 4 to 24, Size = 8192
# Exploring different OpenMP directives (SIMD, Collapse)
# -------------------------------------------------------------------------
LOG_P2="results/part2_step2_log.txt"
> "$LOG_P2"
echo "Starting Part 2, Step 2 Tests (Size: 8192, Threads: 4-24)... (Saving to $LOG_P2)"

# Sequential baseline for the 8192x8192 matrix to calculate speedup
for threads in 4 8 12 16 20 24; do
    run_test "Parallel Line (Menu 5)" 5 8192 "$threads" "$LOG_P2"
    run_test "Parallel + SIMD Line (Menu 6)" 6 8192 "$threads" "$LOG_P2"
    run_test "Parallel + Collapse(2) Line (Menu 7)" 7 8192 "$threads" "$LOG_P2"
done

echo "================================================="
echo "All multi-core tests complete! Check the 'results' folder."

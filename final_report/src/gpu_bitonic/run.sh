#!/bin/bash
# Quick run script for GPU bitonic sort benchmark
# Usage: bash run.sh [size] [runs]

echo "================================"
echo " GPU Bitonic Sort Benchmark"
echo "================================"
echo ""

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Make sure CUDA is installed."
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Build
echo "Building..."
make clean 2>/dev/null
make
echo ""

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
echo ""

SIZE=${1:-""}
RUNS=${2:-5}

if [ -z "$SIZE" ]; then
    # Full test
    echo "Running full benchmark (all sizes)..."
    ./bitonic_sort_cuda
else
    echo "Running test with n=$SIZE, $RUNS runs..."
    ./bitonic_sort_cuda "$SIZE" "$RUNS"
fi

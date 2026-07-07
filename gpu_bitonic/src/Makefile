# Makefile for GPU Bitonic Sort
# Requires: NVIDIA CUDA Toolkit (nvcc)
#
# Usage:
#   make              - Build GPU bitonic sort
#   make test         - Build and run small test
#   make run          - Build and run full benchmark
#   make clean        - Remove build artifacts

# CUDA Compiler and flags
NVCC       = nvcc
NVCC_FLAGS = -O3 -arch=sm_61 -use_fast_math

# GCC for CPU-only version
GCC        = g++
GCC_FLAGS  = -O3 -march=native -std=c++11

# Targets
GPU_TARGET = bitonic_sort_cuda
CPU_TARGET = bitonic_sort_cpu

# Build all
all: $(GPU_TARGET) $(CPU_TARGET)

# GPU binary
$(GPU_TARGET): bitonic_sort_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# CPU-only benchmark (for faster CPU comparison)
$(CPU_TARGET): bitonic_sort_cpu.cpp
	$(GCC) $(GCC_FLAGS) -o $@ $< -fopenmp

# Quick test (small sizes)
test: $(GPU_TARGET)
	./$(GPU_TARGET) 4096 3

# Full benchmark
run: $(GPU_TARGET)
	./$(GPU_TARGET)

# Clean
clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET)

.PHONY: all test run clean

NVCC ?= /usr/local/cuda-12.8/bin/nvcc
NVCC_FLAGS ?= -O3 -std=c++17 -arch=sm_61 -use_fast_math

TARGET = ann_gpu_cuda

all: $(TARGET)

$(TARGET): ann_gpu_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

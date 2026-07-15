CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -march=native
INCLUDES ?= -I../.. -I../../hnswlib

TARGET = ann_advanced_bench

all: $(TARGET)

$(TARGET): ann_advanced_bench.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

run: $(TARGET)
	./$(TARGET) /anndata 200 100000

clean:
	rm -f $(TARGET)

.PHONY: all run clean

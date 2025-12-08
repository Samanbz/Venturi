#pragma once
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

// Base state container for simulation benchmarks
struct BenchmarkState {
    bool initialized = false;
    size_t current_capacity = 0;

    virtual ~BenchmarkState() { cleanup(); }

    virtual void allocate(size_t n) = 0;
    virtual void free_memory() = 0;

    void ensureCapacity(size_t required_n) {
        if (initialized && current_capacity >= required_n) {
            return;  // Already have enough capacity
        }
        // Free existing memory if any
        if (initialized) {
            free_memory();
        }
        // Allocate new memory
        allocate(required_n);
        current_capacity = required_n;
        initialized = true;

        cudaDeviceSynchronize();
    }

    void cleanup() {
        if (initialized) {
            free_memory();
            initialized = false;
            current_capacity = 0;
        }
    }
};

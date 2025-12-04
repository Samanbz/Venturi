#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "types.h"

// Global state for benchmarks
// This struct maintains GPU memory allocations across benchmark iterations
// to avoid expensive allocation/deallocation in the benchmark loop
struct BenchmarkState {
    float* d_inventory;               // Device memory for inventory data
    float* d_risk_aversion;           // Device memory for risk aversion data
    curandState* d_rngStates;         // Device memory for RNG states (pre-initialized)
    MarketParams params;              // Simulation parameters
    bool initialized = false;         // Track if GPU memory is allocated
    unsigned long long current_seed;  // Track current RNG seed

    // Initialize GPU memory and parameters for a given number of agents
    void initialize(int num_agents, unsigned long long seed = 12345ULL);

    // Re-initialize RNG states with a new seed (if needed)
    void reinitializeRNG(unsigned long long seed);

    // Free GPU memory
    void cleanup();
};

// Global instance shared across all benchmarks
extern BenchmarkState g_state;

// Fixture class for automatic setup/teardown
// Inheriting from benchmark::Fixture provides SetUp/TearDown hooks
// that run before/after each benchmark
class SimulationFixture : public benchmark::Fixture {
   public:
    void SetUp(const ::benchmark::State& state) override;
    void TearDown(const ::benchmark::State& state) override;
};

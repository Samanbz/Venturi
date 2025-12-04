#include "simulation_benchmark_fixture.h"

#include "simulation.h"

// Define the global state instance
BenchmarkState g_state;

void BenchmarkState::initialize(int num_agents, unsigned long long seed) {
    // Clean up previous allocation if it exists
    if (initialized) {
        cleanup();
    }

    // Set parameters
    params.num_agents = num_agents;
    params.decay_rate = 0.5f;
    params.risk_mean = 0.0f;
    params.risk_stddev = 1.0f;
    current_seed = seed;

    // Allocate GPU memory
    cudaMalloc(&d_inventory, num_agents * sizeof(float));
    cudaMalloc(&d_risk_aversion, num_agents * sizeof(float));
    cudaMalloc(&d_rngStates, num_agents * sizeof(curandState));

    // Copy parameters to device constant memory
    copyParamsToDevice(params);

    // Initialize RNG states ONCE (this is expensive, so do it outside benchmark loop)
    setupRNG(d_rngStates, num_agents, seed);

    initialized = true;
}

void BenchmarkState::reinitializeRNG(unsigned long long seed) {
    if (!initialized)
        return;

    current_seed = seed;
    setupRNG(d_rngStates, params.num_agents, seed);
}

void BenchmarkState::cleanup() {
    if (initialized) {
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_rngStates);
        initialized = false;
    }
}

void SimulationFixture::SetUp(const ::benchmark::State& state) {
    // Extract the number of agents from benchmark parameters
    // state.range(0) gets the first argument passed via ->Args({...})
    int num_agents = state.range(0);
    g_state.initialize(num_agents);
}

void SimulationFixture::TearDown(const ::benchmark::State& state) {
    // Keep state alive for multiple iterations
    // We don't cleanup here to avoid re-allocating GPU memory between iterations
    // Cleanup happens in the next SetUp() or at program end
}

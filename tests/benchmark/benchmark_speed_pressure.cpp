#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "../common/benchmark_common.h"

struct SpeedPressureState : public BenchmarkState {
    float* d_inventory = nullptr;
    float* d_risk_aversion = nullptr;
    float* d_local_density = nullptr;
    float* d_speed_term_1 = nullptr;
    float* d_speed_term_2 = nullptr;
    float* d_speed = nullptr;
    float* d_execution_cost = nullptr;
    int* d_agent_indices = nullptr;
    curandState* d_rngStates = nullptr;
    MarketParams params;

    ~SpeedPressureState() override { cleanup(); }

    void allocate(size_t n) override {
        cudaMalloc(&d_inventory, n * sizeof(float));
        cudaMalloc(&d_risk_aversion, n * sizeof(float));
        cudaMalloc(&d_local_density, n * sizeof(float));
        cudaMalloc(&d_speed_term_1, n * sizeof(float));
        cudaMalloc(&d_speed_term_2, n * sizeof(float));
        cudaMalloc(&d_speed, n * sizeof(float));
        cudaMalloc(&d_execution_cost, n * sizeof(float));
        cudaMalloc(&d_agent_indices, n * sizeof(int));
        cudaMalloc(&d_rngStates, n * sizeof(curandState));

        // Initialize identity mapping for indices
        std::vector<int> h_indices(n);
        std::iota(h_indices.begin(), h_indices.end(), 0);
        cudaMemcpy(d_agent_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        launchSetupRNG(d_rngStates, n, 12345ULL);
    }

    void free_memory() override {
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_local_density);
        cudaFree(d_speed_term_1);
        cudaFree(d_speed_term_2);
        cudaFree(d_speed);
        cudaFree(d_execution_cost);
        cudaFree(d_agent_indices);
        cudaFree(d_rngStates);
    }
} g_speed_pressure_state;

class SpeedPressureFixture : public benchmark::Fixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        g_speed_pressure_state.params.num_agents = state.range(0);
        g_speed_pressure_state.params.num_steps = 100;
        g_speed_pressure_state.params.permanent_impact = 0.1f;
        g_speed_pressure_state.params.temporary_impact = 0.05f;

        g_speed_pressure_state.ensureCapacity(g_speed_pressure_state.params.num_agents);
        copyParamsToDevice(g_speed_pressure_state.params);

        // Initialize data with random values
        launchInitializeUniform(g_speed_pressure_state.d_inventory, 0.0f, 100.0f,
                                g_speed_pressure_state.d_rngStates,
                                g_speed_pressure_state.params.num_agents);
        launchInitializeUniform(g_speed_pressure_state.d_risk_aversion, 0.1f, 1.0f,
                                g_speed_pressure_state.d_rngStates,
                                g_speed_pressure_state.params.num_agents);
        launchInitializeUniform(g_speed_pressure_state.d_local_density, 0.1f, 5.0f,
                                g_speed_pressure_state.d_rngStates,
                                g_speed_pressure_state.params.num_agents);

        cudaDeviceSynchronize();
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive
    }
};

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputeSpeedTerms)(benchmark::State& state) {
    int dt = 0;
    for (auto _ : state) {
        launchComputeSpeedTerms(
            g_speed_pressure_state.d_risk_aversion, g_speed_pressure_state.d_local_density,
            g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_speed_term_1,
            g_speed_pressure_state.d_speed_term_2, dt, g_speed_pressure_state.params.num_agents);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputePressure)(benchmark::State& state) {
    float pressure = 0.0f;
    // Pre-compute terms once
    launchComputeSpeedTerms(
        g_speed_pressure_state.d_risk_aversion, g_speed_pressure_state.d_local_density,
        g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_speed_term_1,
        g_speed_pressure_state.d_speed_term_2, 0, g_speed_pressure_state.params.num_agents);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        launchComputePressure(g_speed_pressure_state.d_speed_term_1,
                              g_speed_pressure_state.d_speed_term_2, &pressure,
                              g_speed_pressure_state.params.num_agents);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputeSpeed)(benchmark::State& state) {
    float pressure = 10.0f;
    // Pre-compute terms once
    launchComputeSpeedTerms(
        g_speed_pressure_state.d_risk_aversion, g_speed_pressure_state.d_local_density,
        g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_speed_term_1,
        g_speed_pressure_state.d_speed_term_2, 0, g_speed_pressure_state.params.num_agents);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        launchUpdateAgentState(
            g_speed_pressure_state.d_speed_term_1, g_speed_pressure_state.d_speed_term_2,
            g_speed_pressure_state.d_local_density, g_speed_pressure_state.d_agent_indices,
            pressure, g_speed_pressure_state.d_speed, g_speed_pressure_state.d_inventory,
            g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_execution_cost,
            g_speed_pressure_state.d_execution_cost, g_speed_pressure_state.params.num_agents);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, FullSpeedPressureStep)(benchmark::State& state) {
    int dt = 0;
    float pressure = 0.0f;
    for (auto _ : state) {
        launchComputeSpeedTerms(
            g_speed_pressure_state.d_risk_aversion, g_speed_pressure_state.d_local_density,
            g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_speed_term_1,
            g_speed_pressure_state.d_speed_term_2, dt, g_speed_pressure_state.params.num_agents);

        launchComputePressure(g_speed_pressure_state.d_speed_term_1,
                              g_speed_pressure_state.d_speed_term_2, &pressure,
                              g_speed_pressure_state.params.num_agents);

        launchUpdateAgentState(
            g_speed_pressure_state.d_speed_term_1, g_speed_pressure_state.d_speed_term_2,
            g_speed_pressure_state.d_local_density, g_speed_pressure_state.d_agent_indices,
            pressure, g_speed_pressure_state.d_speed, g_speed_pressure_state.d_inventory,
            g_speed_pressure_state.d_inventory, g_speed_pressure_state.d_execution_cost,
            g_speed_pressure_state.d_execution_cost, g_speed_pressure_state.params.num_agents);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputeSpeedTerms)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputePressure)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputeSpeed)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, FullSpeedPressureStep)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMicrosecond);

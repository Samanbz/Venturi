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
    float* d_target_inventory = nullptr;
    float* d_speed_term_1 = nullptr;
    float* d_speed_term_2 = nullptr;
    float* d_speed = nullptr;
    float* d_execution_cost = nullptr;
    float* d_cash = nullptr;
    int* d_agent_indices = nullptr;
    float* d_pressure_buffer = nullptr;
    float* d_greed = nullptr;
    float* d_belief = nullptr;
    curandState* d_rngStates = nullptr;
    MarketParams params{};

    ~SpeedPressureState() override { cleanup(); }

    void allocate(size_t n) override {
        cudaMalloc(&d_inventory, n * sizeof(float));
        cudaMalloc(&d_risk_aversion, n * sizeof(float));
        cudaMalloc(&d_local_density, n * sizeof(float));
        cudaMalloc(&d_target_inventory, n * sizeof(float));
        cudaMalloc(&d_speed_term_1, n * sizeof(float));
        cudaMalloc(&d_speed_term_2, n * sizeof(float));
        cudaMalloc(&d_speed, n * sizeof(float));

        cudaMemset(d_target_inventory, 0, n * sizeof(float));
        cudaMalloc(&d_execution_cost, n * sizeof(float));
        cudaMalloc(&d_cash, n * sizeof(float));
        cudaMalloc(&d_agent_indices, n * sizeof(int));
        cudaMalloc(&d_pressure_buffer, 2 * sizeof(float));
        cudaMalloc(&d_rngStates, n * sizeof(curandState));

        cudaMalloc(&d_greed, n * sizeof(float));
        cudaMemset(d_greed, 0, n * sizeof(float));
        cudaMalloc(&d_belief, n * sizeof(float));
        cudaMemset(d_belief, 0, n * sizeof(float));

        // Initialize identity mapping for indices
        std::vector<int> h_indices(n);
        std::iota(h_indices.begin(), h_indices.end(), 0);
        cudaMemcpy(d_agent_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        launchSetupRNG(d_rngStates, n, 12345ULL);
    }

    void free_memory() override {
        cudaFree(d_target_inventory);
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_local_density);
        cudaFree(d_speed_term_1);
        cudaFree(d_speed_term_2);
        cudaFree(d_speed);
        if (d_greed)
            cudaFree(d_greed);
        if (d_belief)
            cudaFree(d_belief);
        cudaFree(d_execution_cost);
        cudaFree(d_cash);
        cudaFree(d_agent_indices);
        cudaFree(d_pressure_buffer);
        cudaFree(d_rngStates);
    }
} g_state;

class SpeedPressureFixture : public benchmark::Fixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        g_state.params.num_agents = state.range(0);
        g_state.params.num_steps = 100;
        g_state.params.permanent_impact = 0.1f;
        g_state.params.temporary_impact = 0.05f;

        g_state.ensureCapacity(g_state.params.num_agents);
        copyParamsToDevice(g_state.params);

        // Initialize data with random values
        launchInitializeUniform(g_state.d_inventory, 0.0f, 100.0f, g_state.d_rngStates,
                                g_state.params.num_agents);
        launchInitializeUniform(g_state.d_risk_aversion, 0.1f, 1.0f, g_state.d_rngStates,
                                g_state.params.num_agents);
        launchInitializeUniform(g_state.d_local_density, 0.1f, 5.0f, g_state.d_rngStates,
                                g_state.params.num_agents);

        cudaDeviceSynchronize();
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive
    }
};

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputeSpeedTerms)(benchmark::State& state) {
    int dt = 0;
    for (auto _ : state) {
        launchComputeSpeedTerms(g_state.d_risk_aversion, g_state.d_local_density,
                                g_state.d_inventory, g_state.d_target_inventory,
                                g_state.d_speed_term_1, g_state.d_speed_term_2, dt, g_state.params);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputePressure)(benchmark::State& state) {
    float pressure = 0.0f;
    // Pre-compute terms once
    launchComputeSpeedTerms(g_state.d_risk_aversion, g_state.d_local_density, g_state.d_inventory,
                            g_state.d_target_inventory, g_state.d_speed_term_1,
                            g_state.d_speed_term_2, 0, g_state.params);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        launchComputePressure(g_state.d_speed_term_1, g_state.d_speed_term_2,
                              g_state.d_pressure_buffer, &pressure, g_state.params.num_agents);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, ComputeSpeed)(benchmark::State& state) {
    float pressure = 10.0f;
    // Pre-compute terms once
    launchComputeSpeedTerms(g_state.d_risk_aversion, g_state.d_local_density, g_state.d_inventory,
                            g_state.d_target_inventory, g_state.d_speed_term_1,
                            g_state.d_speed_term_2, 0, g_state.params);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        launchUpdateAgentState(g_state.d_speed_term_1, g_state.d_speed_term_2,
                               g_state.d_local_density, g_state.d_agent_indices, pressure,
                               g_state.d_greed, g_state.d_belief, 0.0f, g_state.d_speed,
                               g_state.d_inventory, g_state.d_target_inventory,
                               g_state.d_execution_cost, g_state.d_cash, 100.0f, g_state.params);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_DEFINE_F(SpeedPressureFixture, FullSpeedPressureStep)(benchmark::State& state) {
    int dt = 0;
    float pressure = 0.0f;
    for (auto _ : state) {
        launchComputeSpeedTerms(g_state.d_risk_aversion, g_state.d_local_density,
                                g_state.d_inventory, g_state.d_target_inventory,
                                g_state.d_speed_term_1, g_state.d_speed_term_2, dt, g_state.params);

        launchComputePressure(g_state.d_speed_term_1, g_state.d_speed_term_2,
                              g_state.d_pressure_buffer, &pressure, g_state.params.num_agents);

        launchUpdateAgentState(g_state.d_speed_term_1, g_state.d_speed_term_2,
                               g_state.d_local_density, g_state.d_agent_indices, pressure,
                               g_state.d_greed, g_state.d_belief, 0.0f, g_state.d_speed,
                               g_state.d_inventory, g_state.d_target_inventory,
                               g_state.d_execution_cost, g_state.d_cash, 100.0f, g_state.params);
        cudaDeviceSynchronize();
    }
}

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputeSpeedTerms)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputePressure)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, ComputeSpeed)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpeedPressureFixture, FullSpeedPressureStep)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

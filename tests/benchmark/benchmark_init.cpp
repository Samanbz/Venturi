#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../common/benchmark_common.h"
#include "simulation.h"
#include "types.h"

struct InitState : public BenchmarkState {
    float* d_inventory = nullptr;
    float* d_risk_aversion = nullptr;
    curandState* d_rngStates = nullptr;
    MarketParams params;

    void allocate(size_t n) override {
        cudaMalloc(&d_inventory, n * sizeof(float));
        cudaMalloc(&d_risk_aversion, n * sizeof(float));
        cudaMalloc(&d_rngStates, n * sizeof(curandState));

        setupRNG(d_rngStates, n, 12345ULL);
    }

    void free_memory() override {
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_rngStates);
    }
} g_init_state;

class InitFixture : public benchmark::Fixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        g_init_state.params.num_agents = state.range(0);
        g_init_state.ensureCapacity(g_init_state.params.num_agents);

        copyParamsToDevice(g_init_state.params);
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive for multiple iterations
    }
};

/*
 * Scalability Benchmarks - Test kernel performance across different agent counts
 */

BENCHMARK_DEFINE_F(InitFixture, InitializeInventories)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeExponential(g_init_state.d_inventory, g_init_state.params.decay_rate,
                                    g_init_state.d_rngStates, g_init_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_init_state.params.num_agents * sizeof(float));
}

BENCHMARK_DEFINE_F(InitFixture, InitializeRiskAversions)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeNormal(g_init_state.d_risk_aversion, g_init_state.params.risk_mean,
                               g_init_state.params.risk_stddev, g_init_state.d_rngStates,
                               g_init_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_init_state.params.num_agents * sizeof(float));
}

BENCHMARK_REGISTER_F(InitFixture, InitializeInventories)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(InitFixture, InitializeRiskAversions)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

/*
 * Parameter Variation Benchmarks - Test impact of different parameter values
 */

BENCHMARK_DEFINE_F(InitFixture, InitInventories_DecayRateVariation)(benchmark::State& state) {
    float decay_rate = static_cast<float>(state.range(1)) / 10.0f;
    g_init_state.params.decay_rate = decay_rate;
    copyParamsToDevice(g_init_state.params);

    for (auto _ : state) {
        launchInitializeExponential(g_init_state.d_inventory, g_init_state.params.decay_rate,
                                    g_init_state.d_rngStates, g_init_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents);
    state.SetLabel("decay_rate=" + std::to_string(decay_rate));
}

BENCHMARK_REGISTER_F(InitFixture, InitInventories_DecayRateVariation)
    ->Args({1000000, 1})    // 1M agents, decay_rate=0.1
    ->Args({1000000, 5})    // 1M agents, decay_rate=0.5
    ->Args({1000000, 10})   // 1M agents, decay_rate=1.0
    ->Args({1000000, 20})   // 1M agents, decay_rate=2.0
    ->Args({1000000, 100})  // 1M agents, decay_rate=10.0
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(InitFixture, InitRiskAversions_StdDevVariation)(benchmark::State& state) {
    // Note: Fixture already initialized GPU memory in SetUp() with 1M agents (first ->Args value)
    float risk_stddev = static_cast<float>(state.range(1)) / 10.0f;
    g_init_state.params.risk_stddev = risk_stddev;
    copyParamsToDevice(g_init_state.params);

    for (auto _ : state) {
        launchInitializeNormal(g_init_state.d_risk_aversion, g_init_state.params.risk_mean,
                               g_init_state.params.risk_stddev, g_init_state.d_rngStates,
                               g_init_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents);
    state.SetLabel("stddev=" + std::to_string(risk_stddev));
}

BENCHMARK_REGISTER_F(InitFixture, InitRiskAversions_StdDevVariation)
    ->Args({1000000, 5})   // 1M agents, stddev=0.5
    ->Args({1000000, 10})  // 1M agents, stddev=1.0
    ->Args({1000000, 20})  // 1M agents, stddev=2.0
    ->Args({1000000, 50})  // 1M agents, stddev=5.0
    ->Unit(benchmark::kMillisecond);

/*
 * Combined Kernel Benchmark - Test both kernels running sequentially
 */

BENCHMARK_DEFINE_F(InitFixture, CombinedInitialization)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeExponential(g_init_state.d_inventory, g_init_state.params.decay_rate,
                                    g_init_state.d_rngStates, g_init_state.params.num_agents);
        launchInitializeNormal(g_init_state.d_risk_aversion, g_init_state.params.risk_mean,
                               g_init_state.params.risk_stddev, g_init_state.d_rngStates,
                               g_init_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents *
                            2);  // Both kernels
}

BENCHMARK_REGISTER_F(InitFixture, CombinedInitialization)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

/*
 * RNG Setup Benchmark - Measure the cost of initializing RNG states
 * This is the expensive operation that we do ONCE, not per iteration
 */

BENCHMARK_DEFINE_F(InitFixture, SetupRNG)(benchmark::State& state) {
    unsigned long long seed = 99999ULL;

    for (auto _ : state) {
        setupRNG(g_init_state.d_rngStates, g_init_state.params.num_agents, seed);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_init_state.params.num_agents);
    state.SetLabel("RNG_init_cost");
}

BENCHMARK_REGISTER_F(InitFixture, SetupRNG)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

/*
 * Cleanup - Runs at the end to free GPU memory
 */

static void BM_Cleanup(benchmark::State& state) {
    for (auto _ : state) {
        g_init_state.cleanup();
    }
}

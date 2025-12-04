#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../fixtures/simulation_benchmark_fixture.h"
#include "simulation.h"
#include "types.h"

/*
 * Scalability Benchmarks - Test kernel performance across different agent counts
 */

BENCHMARK_DEFINE_F(SimulationFixture, InitializeInventories)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeInventories(g_state.d_inventory, g_state.d_rngStates,
                                    g_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_state.params.num_agents * sizeof(float));
}

BENCHMARK_DEFINE_F(SimulationFixture, InitializeRiskAversions)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeRiskAversions(g_state.d_risk_aversion, g_state.d_rngStates,
                                      g_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_state.params.num_agents * sizeof(float));
}

BENCHMARK_REGISTER_F(SimulationFixture, InitializeInventories)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Args({10000000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SimulationFixture, InitializeRiskAversions)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Args({10000000})
    ->Unit(benchmark::kMillisecond);

/*
 * Parameter Variation Benchmarks - Test impact of different parameter values
 */

BENCHMARK_DEFINE_F(SimulationFixture, InitInventories_DecayRateVariation)(benchmark::State& state) {
    // Note: Fixture already initialized GPU memory in SetUp() with 1M agents (first ->Args value)
    float decay_rate = static_cast<float>(state.range(1)) / 10.0f;
    g_state.params.decay_rate = decay_rate;
    copyParamsToDevice(g_state.params);

    for (auto _ : state) {
        launchInitializeInventories(g_state.d_inventory, g_state.d_rngStates,
                                    g_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents);
    state.SetLabel("decay_rate=" + std::to_string(decay_rate));
}

BENCHMARK_REGISTER_F(SimulationFixture, InitInventories_DecayRateVariation)
    ->Args({1000000, 1})    // 1M agents, decay_rate=0.1
    ->Args({1000000, 5})    // 1M agents, decay_rate=0.5
    ->Args({1000000, 10})   // 1M agents, decay_rate=1.0
    ->Args({1000000, 20})   // 1M agents, decay_rate=2.0
    ->Args({1000000, 100})  // 1M agents, decay_rate=10.0
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SimulationFixture, InitRiskAversions_StdDevVariation)(benchmark::State& state) {
    // Note: Fixture already initialized GPU memory in SetUp() with 1M agents (first ->Args value)
    float risk_stddev = static_cast<float>(state.range(1)) / 10.0f;
    g_state.params.risk_stddev = risk_stddev;
    copyParamsToDevice(g_state.params);

    for (auto _ : state) {
        launchInitializeRiskAversions(g_state.d_risk_aversion, g_state.d_rngStates,
                                      g_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents);
    state.SetLabel("stddev=" + std::to_string(risk_stddev));
}

BENCHMARK_REGISTER_F(SimulationFixture, InitRiskAversions_StdDevVariation)
    ->Args({1000000, 5})   // 1M agents, stddev=0.5
    ->Args({1000000, 10})  // 1M agents, stddev=1.0
    ->Args({1000000, 20})  // 1M agents, stddev=2.0
    ->Args({1000000, 50})  // 1M agents, stddev=5.0
    ->Unit(benchmark::kMillisecond);

/*
 * Combined Kernel Benchmark - Test both kernels running sequentially
 */

BENCHMARK_DEFINE_F(SimulationFixture, CombinedInitialization)(benchmark::State& state) {
    for (auto _ : state) {
        launchInitializeInventories(g_state.d_inventory, g_state.d_rngStates,
                                    g_state.params.num_agents);
        launchInitializeRiskAversions(g_state.d_risk_aversion, g_state.d_rngStates,
                                      g_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents * 2);  // Both kernels
}

BENCHMARK_REGISTER_F(SimulationFixture, CombinedInitialization)
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

/*
 * RNG Setup Benchmark - Measure the cost of initializing RNG states
 * This is the expensive operation that we do ONCE, not per iteration
 */

BENCHMARK_DEFINE_F(SimulationFixture, SetupRNG)(benchmark::State& state) {
    unsigned long long seed = 99999ULL;

    for (auto _ : state) {
        // Reinitialize RNG states - this is what was killing performance before!
        state.PauseTiming();
        // We don't count the reinit in the benchmark, just showing the cost
        state.ResumeTiming();

        setupRNG(g_state.d_rngStates, g_state.params.num_agents, seed);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_state.params.num_agents);
    state.SetLabel("RNG_init_cost");
}

BENCHMARK_REGISTER_F(SimulationFixture, SetupRNG)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

/*
 * Cleanup - Runs at the end to free GPU memory
 */

static void BM_Cleanup(benchmark::State& state) {
    for (auto _ : state) {
        g_state.cleanup();
    }
}

BENCHMARK(BM_Cleanup);

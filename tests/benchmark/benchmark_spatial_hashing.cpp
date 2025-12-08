#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "../common/benchmark_common.h"

struct SpatialHashingState : public SpatialState {
    int* d_agent_hash = nullptr;
    int* d_agent_indices = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;

    void allocate(size_t n) override {
        allocateCommon(n);
        cudaMalloc(&d_agent_hash, n * sizeof(int));
        cudaMalloc(&d_agent_indices, n * sizeof(int));
        cudaMalloc(&d_cell_start, n * sizeof(int));  // Over-allocate for simplicity
        cudaMalloc(&d_cell_end, n * sizeof(int));    // Over-allocate for simplicity
    };

    void free_memory() override {
        cudaFree(d_agent_hash);
        cudaFree(d_agent_indices);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
    };
} g_spatial_hashing_state;

class SpatialHashingFixture : public SimulationFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        setupSimulation(g_spatial_hashing_state, state, false);
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive for multiple iterations
    }
};

class SpatialHashingFixtureCustom : public SimulationFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        setupSimulation(g_spatial_hashing_state, state, true);
    }

    void TearDown(const ::benchmark::State& state) override {}
};

BENCHMARK_DEFINE_F(SpatialHashingFixture, CalculateSpatialHash)(benchmark::State& state) {
    for (auto _ : state) {
        launchCalculateSpatialHash(
            g_spatial_hashing_state.d_inventory, g_spatial_hashing_state.d_execution_cost,
            g_spatial_hashing_state.d_agent_hash, g_spatial_hashing_state.d_agent_indices,
            g_spatial_hashing_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents *
                            (sizeof(float) * 2 + sizeof(int) * 2));
}

BENCHMARK_DEFINE_F(SpatialHashingFixture, SortByKey)(benchmark::State& state) {
    for (auto _ : state) {
        // Note: Sorting already sorted data (after first iteration) might be faster
        // than random data, but this measures the throughput of the sort call.
        launchSortByKey(g_spatial_hashing_state.d_agent_hash,
                        g_spatial_hashing_state.d_agent_indices,
                        g_spatial_hashing_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents *
                            sizeof(int) * 2);
}

BENCHMARK_DEFINE_F(SpatialHashingFixture, FindCellBounds)(benchmark::State& state) {
    for (auto _ : state) {
        launchFindCellBounds(
            g_spatial_hashing_state.d_agent_hash, g_spatial_hashing_state.d_cell_start,
            g_spatial_hashing_state.d_cell_end, g_spatial_hashing_state.params.num_agents);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
}

BENCHMARK_DEFINE_F(SpatialHashingFixture, FullSpatialHashingPipeline)(benchmark::State& state) {
    for (auto _ : state) {
        launchCalculateSpatialHash(
            g_spatial_hashing_state.d_inventory, g_spatial_hashing_state.d_execution_cost,
            g_spatial_hashing_state.d_agent_hash, g_spatial_hashing_state.d_agent_indices,
            g_spatial_hashing_state.params.num_agents);

        launchSortByKey(g_spatial_hashing_state.d_agent_hash,
                        g_spatial_hashing_state.d_agent_indices,
                        g_spatial_hashing_state.params.num_agents);

        launchFindCellBounds(
            g_spatial_hashing_state.d_agent_hash, g_spatial_hashing_state.d_cell_start,
            g_spatial_hashing_state.d_cell_end, g_spatial_hashing_state.params.num_agents);

        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
}

BENCHMARK_REGISTER_F(SpatialHashingFixture, CalculateSpatialHash)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, SortByKey)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, FindCellBounds)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, FullSpatialHashingPipeline)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

// Custom Parameter Benchmarks
BENCHMARK_DEFINE_F(SpatialHashingFixtureCustom, CalculateSpatialHash_HighDensity)
(benchmark::State& state) {
    for (auto _ : state) {
        launchCalculateSpatialHash(
            g_spatial_hashing_state.d_inventory, g_spatial_hashing_state.d_execution_cost,
            g_spatial_hashing_state.d_agent_hash, g_spatial_hashing_state.d_agent_indices,
            g_spatial_hashing_state.params.num_agents);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
}

BENCHMARK_REGISTER_F(SpatialHashingFixtureCustom, CalculateSpatialHash_HighDensity)
    // 1M agents, Gaussian, Mean 0.5 (50/100), StdDev 0.01 (1/100) -> Very dense cluster
    ->Args({1000000, 1, 50, 1})
    // 1M agents, Gaussian, Mean 0.5 (50/100), StdDev 0.2 (20/100) -> Spread out
    ->Args({1000000, 1, 50, 10})
    ->Args({1000000, 1, 50, 100})
    ->Unit(benchmark::kMillisecond);

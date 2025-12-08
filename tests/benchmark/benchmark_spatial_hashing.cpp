#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "../common/benchmark_common.h"

struct SpatialHashingState : public BenchmarkState {
    float* d_inventory = nullptr;
    float* d_execution_cost = nullptr;
    int* d_agent_hash = nullptr;
    int* d_agent_indices = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;
    curandState* d_rngStates = nullptr;

    MarketParams params;

    void allocate(size_t n) override {
        cudaMalloc(&d_inventory, n * sizeof(float));
        cudaMalloc(&d_execution_cost, n * sizeof(float));
        cudaMalloc(&d_agent_hash, n * sizeof(int));
        cudaMalloc(&d_agent_indices, n * sizeof(int));
        cudaMalloc(&d_cell_start, n * sizeof(int));  // Over-allocate for simplicity
        cudaMalloc(&d_cell_end, n * sizeof(int));    // Over-allocate for simplicity
        cudaMalloc(&d_rngStates, n * sizeof(curandState));
    };

    void free_memory() override {
        cudaFree(d_inventory);
        cudaFree(d_execution_cost);
        cudaFree(d_agent_hash);
        cudaFree(d_agent_indices);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        cudaFree(d_rngStates);
    };
} g_spatial_hashing_state;

class SpatialHashingFixture : public benchmark::Fixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        g_spatial_hashing_state.params.num_agents = state.range(0);
        g_spatial_hashing_state.ensureCapacity(g_spatial_hashing_state.params.num_agents);

        copyParamsToDevice(g_spatial_hashing_state.params);

        // Initialize RNG and data to ensure non-trivial spatial hashing
        setupRNG(g_spatial_hashing_state.d_rngStates, g_spatial_hashing_state.params.num_agents,
                 1234ULL);

        launchInitializeInventories(g_spatial_hashing_state.d_inventory,
                                    g_spatial_hashing_state.d_rngStates,
                                    g_spatial_hashing_state.params.num_agents);

        // Reuse inventory init to populate execution costs with random floats
        launchInitializeInventories(g_spatial_hashing_state.d_execution_cost,
                                    g_spatial_hashing_state.d_rngStates,
                                    g_spatial_hashing_state.params.num_agents);

        cudaDeviceSynchronize();
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive for multiple iterations
    }
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
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, SortByKey)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, FindCellBounds)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixture, FullSpatialHashingPipeline)
    ->Args({1000})
    ->Args({10000})
    ->Args({100000})
    ->Args({1000000})
    ->Unit(benchmark::kMillisecond);

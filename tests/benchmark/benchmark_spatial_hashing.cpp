#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "../common/benchmark_common.h"

struct SpatialHashingState : public SpatialState {
    int* d_cell_head = nullptr;
    int* d_agent_next = nullptr;

    ~SpatialHashingState() override { cleanup(); }

    void allocate(size_t n) override {
        allocateCommon(n);
        size_t table_size = std::max((size_t) params.hash_table_size, n);
        cudaMalloc(&d_cell_head, table_size * sizeof(int));
        cudaMalloc(&d_agent_next, n * sizeof(int));
    };

    void free_memory() override {
        freeCommon();
        cudaFree(d_cell_head);
        cudaFree(d_agent_next);
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

BENCHMARK_DEFINE_F(SpatialHashingFixture, BuildSpatialHash)(benchmark::State& state) {
    for (auto _ : state) {
        // Reset heads before each run to simulate fresh frame
        cudaMemset(g_spatial_hashing_state.d_cell_head, -1,
                   g_spatial_hashing_state.params.hash_table_size * sizeof(int));

        launchBuildSpatialHash(
            g_spatial_hashing_state.d_inventory, g_spatial_hashing_state.d_execution_cost,
            g_spatial_hashing_state.d_cell_head, g_spatial_hashing_state.d_agent_next,
            g_spatial_hashing_state.params);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
    state.SetBytesProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents *
                            (sizeof(float) * 2 + sizeof(int) * 2));
}

BENCHMARK_DEFINE_F(SpatialHashingFixtureCustom, BuildSpatialHash_HighDensity)
(benchmark::State& state) {
    for (auto _ : state) {
        cudaMemset(g_spatial_hashing_state.d_cell_head, -1,
                   g_spatial_hashing_state.params.hash_table_size * sizeof(int));

        launchBuildSpatialHash(
            g_spatial_hashing_state.d_inventory, g_spatial_hashing_state.d_execution_cost,
            g_spatial_hashing_state.d_cell_head, g_spatial_hashing_state.d_agent_next,
            g_spatial_hashing_state.params);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * g_spatial_hashing_state.params.num_agents);
}

BENCHMARK_REGISTER_F(SpatialHashingFixture, BuildSpatialHash)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SpatialHashingFixtureCustom, BuildSpatialHash_HighDensity)
    ->Args({1024, 0, 50, 10})
    ->Args({4096, 0, 50, 10})
    ->Args({16384, 0, 50, 10})
    ->Unit(benchmark::kMillisecond);

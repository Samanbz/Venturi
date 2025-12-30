#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "../common/benchmark_common.h"

struct LocalDensityState : public SpatialState {
    // Raw agent data (inventory, execution_cost, rngStates in base)
    float* d_cash = nullptr;
    float* d_speed = nullptr;
    float* d_risk_aversion = nullptr;

    // Sorted agent data
    float* d_inventory_sorted = nullptr;
    float* d_execution_cost_sorted = nullptr;
    float* d_cash_sorted = nullptr;
    float* d_speed_sorted = nullptr;
    float* d_risk_aversion_sorted = nullptr;

    // Spatial hashing data
    int* d_agent_hash = nullptr;
    int* d_agent_indices = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;

    // Output
    float* d_local_density = nullptr;

    ~LocalDensityState() override { cleanup(); }

    void allocate(size_t n) override {
        allocateCommon(n);
        cudaMalloc(&d_cash, n * sizeof(float));
        cudaMalloc(&d_speed, n * sizeof(float));
        cudaMalloc(&d_risk_aversion, n * sizeof(float));

        cudaMalloc(&d_inventory_sorted, n * sizeof(float));
        cudaMalloc(&d_execution_cost_sorted, n * sizeof(float));
        cudaMalloc(&d_cash_sorted, n * sizeof(float));
        cudaMalloc(&d_speed_sorted, n * sizeof(float));
        cudaMalloc(&d_risk_aversion_sorted, n * sizeof(float));

        cudaMalloc(&d_agent_hash, n * sizeof(int));
        cudaMalloc(&d_agent_indices, n * sizeof(int));

        // Ensure we have enough space for the hash table
        size_t table_size = std::max((size_t) params.hash_table_size, n);
        cudaMalloc(&d_cell_start, table_size * sizeof(int));
        cudaMalloc(&d_cell_end, table_size * sizeof(int));

        cudaMalloc(&d_local_density, n * sizeof(float));
    };

    void free_memory() override {
        freeCommon();
        cudaFree(d_cash);
        cudaFree(d_speed);
        cudaFree(d_risk_aversion);

        cudaFree(d_inventory_sorted);
        cudaFree(d_execution_cost_sorted);
        cudaFree(d_cash_sorted);
        cudaFree(d_speed_sorted);
        cudaFree(d_risk_aversion_sorted);

        cudaFree(d_agent_hash);
        cudaFree(d_agent_indices);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);

        cudaFree(d_local_density);
    };
} g_local_density_state;

class LocalDensityFixture : public SimulationFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        setupSimulation(g_local_density_state, state, false);
        runPipelineSetup(g_local_density_state.params.num_agents);
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive for multiple iterations
    }

   protected:
    void runPipelineSetup(int num_agents) {
        cudaMemset(g_local_density_state.d_cash, 0, num_agents * sizeof(float));
        cudaMemset(g_local_density_state.d_speed, 0, num_agents * sizeof(float));

        // Prepare Spatial Hash structures (Hash -> Sort -> Bounds)
        cudaMemset(g_local_density_state.d_cell_start, -1,
                   g_local_density_state.params.hash_table_size * sizeof(int));
        cudaMemset(g_local_density_state.d_cell_end, -1,
                   g_local_density_state.params.hash_table_size * sizeof(int));

        launchCalculateSpatialHash(
            g_local_density_state.d_inventory, g_local_density_state.d_execution_cost,
            g_local_density_state.d_agent_hash, g_local_density_state.d_agent_indices, num_agents);

        launchSortByKey(g_local_density_state.d_agent_hash, g_local_density_state.d_agent_indices,
                        num_agents);

        launchFindCellBounds(g_local_density_state.d_agent_hash, g_local_density_state.d_cell_start,
                             g_local_density_state.d_cell_end, num_agents);

        // Also run ReorderData once so sorted arrays are populated for Density benchmark
        launchReorderData(
            g_local_density_state.d_agent_indices, num_agents, g_local_density_state.d_inventory,
            g_local_density_state.d_inventory_sorted, g_local_density_state.d_execution_cost,
            g_local_density_state.d_execution_cost_sorted);

        cudaDeviceSynchronize();
    }
};

class LocalDensityFixtureCustom : public LocalDensityFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        setupSimulation(g_local_density_state, state, true);
        runPipelineSetup(g_local_density_state.params.num_agents);
    }
};

BENCHMARK_DEFINE_F(LocalDensityFixture, ReorderData)(benchmark::State& state) {
    for (auto _ : state) {
        launchReorderData(
            g_local_density_state.d_agent_indices, g_local_density_state.params.num_agents,
            g_local_density_state.d_inventory, g_local_density_state.d_inventory_sorted,
            g_local_density_state.d_execution_cost, g_local_density_state.d_execution_cost_sorted);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * g_local_density_state.params.num_agents);
    // Bytes: Read 4 floats + 1 int per agent. Write 4 floats per agent.
    // Total: 8 floats + 1 int = 36 bytes per agent.
    state.SetBytesProcessed(state.iterations() * g_local_density_state.params.num_agents *
                            (8 * sizeof(float) + sizeof(int)));
}

BENCHMARK_DEFINE_F(LocalDensityFixture, ComputeLocalDensities)(benchmark::State& state) {
    for (auto _ : state) {
        launchComputeLocalDensities(
            g_local_density_state.d_inventory_sorted, g_local_density_state.d_execution_cost_sorted,
            g_local_density_state.d_cell_start, g_local_density_state.d_cell_end,
            g_local_density_state.d_agent_indices, g_local_density_state.d_local_density,
            g_local_density_state.params.num_agents);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * g_local_density_state.params.num_agents);
}

BENCHMARK_REGISTER_F(LocalDensityFixture, ReorderData)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(LocalDensityFixture, ComputeLocalDensities)
    ->Args({1000, 0})
    ->Args({10000, 0})
    ->Args({100000, 0})
    ->Args({1000000, 0})
    ->Args({1000000, 1})  // Gaussian
    ->Unit(benchmark::kMillisecond);

// Custom Parameter Benchmarks
BENCHMARK_DEFINE_F(LocalDensityFixtureCustom, ComputeLocalDensities_HighDensity)
(benchmark::State& state) {
    for (auto _ : state) {
        launchComputeLocalDensities(
            g_local_density_state.d_inventory_sorted, g_local_density_state.d_execution_cost_sorted,
            g_local_density_state.d_cell_start, g_local_density_state.d_cell_end,
            g_local_density_state.d_agent_indices, g_local_density_state.d_local_density,
            g_local_density_state.params.num_agents);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * g_local_density_state.params.num_agents);
}

BENCHMARK_REGISTER_F(LocalDensityFixtureCustom, ComputeLocalDensities_HighDensity)
    // 1M agents, Gaussian, Mean 0.5 (50/100), StdDev 0.01 (1/100) -> Very dense cluster
    ->Args({100000, 0, 0, 1})
    ->Args({100000, 0, 0, 10})
    ->Args({100000, 1, 50, 1})
    ->Args({100000, 1, 50, 10})
    ->Args({100000, 1, 50, 100})
    ->Unit(benchmark::kMillisecond);

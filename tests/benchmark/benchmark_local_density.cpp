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

    // Spatial hashing data
    int* d_cell_head = nullptr;
    int* d_agent_next = nullptr;

    // Output
    float* d_local_density = nullptr;

    ~LocalDensityState() override { cleanup(); }

    void allocate(size_t n) override {
        allocateCommon(n);
        cudaMalloc(&d_cash, n * sizeof(float));
        cudaMalloc(&d_speed, n * sizeof(float));
        cudaMalloc(&d_risk_aversion, n * sizeof(float));

        // Ensure we have enough space for the hash table
        size_t table_size = std::max((size_t) params.hash_table_size, n);
        cudaMalloc(&d_cell_head, table_size * sizeof(int));
        cudaMalloc(&d_agent_next, n * sizeof(int));

        cudaMalloc(&d_local_density, n * sizeof(float));
    };

    void free_memory() override {
        freeCommon();
        cudaFree(d_cash);
        cudaFree(d_speed);
        cudaFree(d_risk_aversion);

        cudaFree(d_cell_head);
        cudaFree(d_agent_next);

        cudaFree(d_local_density);
    };
} g_local_density_state;

class LocalDensityFixture : public SimulationFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        // Use custom params to allow scaling domain size
        setupSimulation(g_local_density_state, state, true);
        runPipelineSetup(g_local_density_state.params.num_agents);
    }

    void TearDown(const ::benchmark::State& state) override {
        // Keep state alive for multiple iterations
    }

   protected:
    void runPipelineSetup(int num_agents) {
        cudaMemset(g_local_density_state.d_cash, 0, num_agents * sizeof(float));
        cudaMemset(g_local_density_state.d_speed, 0, num_agents * sizeof(float));

        // Prepare Spatial Hash structures
        cudaMemset(g_local_density_state.d_cell_head, -1,
                   g_local_density_state.params.hash_table_size * sizeof(int));

        launchBuildSpatialHash(g_local_density_state.d_inventory,
                               g_local_density_state.d_execution_cost,
                               g_local_density_state.d_cell_head,
                               g_local_density_state.d_agent_next, g_local_density_state.params);

        cudaDeviceSynchronize();
    }
};

BENCHMARK_DEFINE_F(LocalDensityFixture, ComputeLocalDensities)(benchmark::State& state) {
    for (auto _ : state) {
        launchComputeLocalDensities(
            g_local_density_state.d_inventory, g_local_density_state.d_execution_cost,
            g_local_density_state.d_cell_head, g_local_density_state.d_agent_next,
            g_local_density_state.d_local_density, g_local_density_state.params);
        cudaDeviceSynchronize();
    }
}

// Custom fixture for high density scenario
class LocalDensityFixtureCustom : public LocalDensityFixture {
   public:
    void SetUp(const ::benchmark::State& state) override {
        setupSimulation(g_local_density_state, state, true);  // true for high density
        runPipelineSetup(g_local_density_state.params.num_agents);
    }
};

BENCHMARK_DEFINE_F(LocalDensityFixtureCustom, ComputeLocalDensities_HighDensity)
(benchmark::State& state) {
    for (auto _ : state) {
        launchComputeLocalDensities(
            g_local_density_state.d_inventory, g_local_density_state.d_execution_cost,
            g_local_density_state.d_cell_head, g_local_density_state.d_agent_next,
            g_local_density_state.d_local_density, g_local_density_state.params);
        cudaDeviceSynchronize();
    }
}

static void GenerateDensityArgs(benchmark::internal::Benchmark* b) {
    for (int n = 1024; n <= 1048576; n *= 4) {
        // Scale domain size to keep density constant
        // Base density: 1024 agents in [0, 1]x[0, 1]
        // L = sqrt(N / 1024)
        float L = sqrtf((float) n / 1024.0f);
        int range_max_scaled = (int) (L * 100.0f);

        // Args: {num_agents, dist_type(0=uniform), min_scaled(0), max_scaled}
        b->Args({n, 0, 0, range_max_scaled});
    }
}

BENCHMARK_REGISTER_F(LocalDensityFixture, ComputeLocalDensities)
    ->Apply(GenerateDensityArgs)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(LocalDensityFixtureCustom, ComputeLocalDensities_HighDensity)
    ->Args({1024, 0, 50, 10})
    ->Args({4096, 0, 50, 10})
    ->Args({16384, 0, 50, 10})
    ->Unit(benchmark::kMillisecond);

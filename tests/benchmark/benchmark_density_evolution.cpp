#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "../../src/simulation.h"
#include "../../src/types.h"

// A custom fixture that runs the simulation and measures specific kernels
class DensityEvolutionFixture : public benchmark::Fixture {
   public:
    Simulation* sim = nullptr;
    MarketParams params;

    void SetUp(const ::benchmark::State& state) override {
        params.num_agents = state.range(0);
        params.num_steps = 1000;
        params.time_delta = 1.0f / 60.0f;
        params.price_init = 100.0f;
        params.price_randomness_stddev = 0.1f;
        params.permanent_impact = 1e-5f;
        params.temporary_impact = 0.01f;
        params.sph_smoothing_radius = 1.0f;  // Larger radius for this test
        params.congestion_sensitivity = 0.05f;

        // Use a larger hash table to avoid collisions
        int power = 1;
        while ((1 << power) < params.num_agents) {
            power++;
        }
        params.hash_table_size = (1 << (power + 1));

        params.decay_rate = 0.0001f;
        params.mass_alpha = 0.4f;
        params.mass_beta = 0.1f;
        params.risk_mean = 0.01f;
        params.risk_stddev = 0.1f;

        sim = new Simulation(params);
    }

    void TearDown(const ::benchmark::State& state) override { delete sim; }
};

BENCHMARK_DEFINE_F(DensityEvolutionFixture, DensityAtStep)(benchmark::State& state) {
    int target_step = state.range(1);

    // Advance simulation to target step
    for (int i = 0; i < target_step; ++i) {
        sim->step(false, false);
    }
    cudaDeviceSynchronize();

    // Measure computeLocalDensities performance at this state
    for (auto _ : state) {
        sim->computeLocalDensities();
        cudaDeviceSynchronize();
    }
}

// Register benchmarks for different agent counts and steps
// We want to see how performance changes at step 0, 100, 500, 1000
// for a fixed number of agents (e.g., 65536 and 262144)

BENCHMARK_REGISTER_F(DensityEvolutionFixture, DensityAtStep)
    ->RangeMultiplier(10)
    ->Ranges({{10000, 10000}, {0, 10000}})
    ->Iterations(5)  // Limit iterations to avoid hanging on slow steps
    ->Unit(benchmark::kMillisecond);

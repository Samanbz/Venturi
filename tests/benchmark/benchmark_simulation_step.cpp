#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../../src/simulation.h"
#include "../../src/types.h"

class SimulationStepFixture : public benchmark::Fixture {
   public:
    Simulation* sim = nullptr;
    MarketParams params{};

    void SetUp(const ::benchmark::State& state) override {
        params.num_agents = state.range(0);
        params.num_steps = 1000;  // Not used for single step benchmark
        params.time_delta = 1.0f / 60.0f;
        params.price_init = 100.0f;
        params.price_randomness_stddev = 0.1f;
        params.permanent_impact = 1e-5f;
        params.temporary_impact = 0.01f;
        params.sph_smoothing_radius = 1.0f;
        params.congestion_sensitivity = 0.05f;
        params.decay_rate = 0.0001f;
        params.mass_alpha = 0.4f;
        params.mass_beta = 0.1f;
        params.risk_mean = 0.01f;
        params.risk_stddev = 0.1f;

        // Adjust hash table size
        int power = 1;
        while ((1 << power) < params.num_agents) {
            power++;
        }
        params.hash_table_size = (1 << (power + 1));

        sim = new Simulation(params);

        // Warmup
        sim->step(false, false);
        cudaDeviceSynchronize();
    }

    void TearDown(const ::benchmark::State& state) override { delete sim; }
};

BENCHMARK_DEFINE_F(SimulationStepFixture, FullStep)(benchmark::State& state) {
    for (auto _ : state) {
        sim->step(false, false);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * params.num_agents);
}

BENCHMARK_REGISTER_F(SimulationStepFixture, FullStep)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

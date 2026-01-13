#pragma once
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "kernels/common.cuh"
#include "kernels/launchers.h"

/**
 * @brief Base state container for managing benchmark resources.
 *
 * Provides a standard interface for allocating and freeing GPU memory
 * needed for different benchmarking scenarios.
 */
struct BenchmarkState {
    bool initialized = false;
    size_t current_capacity = 0;

    virtual ~BenchmarkState() {}

    /**
     * @brief Pure virtual method to allocate GPU memory for `n` elements.
     */
    virtual void allocate(size_t n) = 0;

    /**
     * @brief Pure virtual method to free allocated GPU memory.
     */
    virtual void free_memory() = 0;

    /**
     * @brief Re-allocates memory if the requested capacity `required_n` exceeds current.
     *
     * @param required_n Number of elements to reserve space for.
     */
    void ensureCapacity(size_t required_n) {
        if (initialized && current_capacity >= required_n) {
            return;  // Already have enough capacity
        }
        // Free existing memory if any
        if (initialized) {
            free_memory();
        }
        // Allocate new memory
        allocate(required_n);
        current_capacity = required_n;
        initialized = true;

        cudaDeviceSynchronize();
    }

    /**
     * @brief Frees resources and resets state.
     */
    virtual void cleanup() {
        if (initialized) {
            free_memory();
            initialized = false;
            current_capacity = 0;
        }
    }
};

/**
 * @brief State container specifically for Spatial Hashing and Local Density benchmarks.
 *
 * Holds device pointers for inventory, execution cost, RNG states, and
 * the spatial hash grid structures (cell heads and linked lists).
 */
struct SpatialState : public BenchmarkState {
    float* d_inventory = nullptr;
    float* d_execution_cost = nullptr;
    curandState* d_rngStates = nullptr;
    MarketParams params;

    void allocateCommon(size_t n) {
        cudaMalloc(&d_inventory, n * sizeof(float));
        cudaMalloc(&d_execution_cost, n * sizeof(float));
        cudaMalloc(&d_rngStates, n * sizeof(curandState));
    }

    void freeCommon() {
        cudaFree(d_inventory);
        cudaFree(d_execution_cost);
        cudaFree(d_rngStates);
        d_inventory = nullptr;
        d_execution_cost = nullptr;
        d_rngStates = nullptr;
    }

    void cleanup() override {
        freeCommon();
        BenchmarkState::cleanup();
    }
};

class SimulationFixture : public benchmark::Fixture {
   protected:
    void setupSimulation(SpatialState& state,
                         const benchmark::State& bench_state,
                         bool use_custom_params = false) {
        state.params.num_agents = bench_state.range(0);
        int dist_type = 0;

        // Set defaults
        state.params.hash_table_size = state.params.num_agents;
        state.params.sph_smoothing_radius = 0.1f;
        state.params.decay_rate = 1.0f;
        state.params.risk_mean = 0.5f;
        state.params.risk_stddev = 0.1f;
        state.params.mass_alpha = 1.0f;
        state.params.mass_beta = 0.1f;

        state.ensureCapacity(state.params.num_agents);
        copyParamsToDevice(state.params);

        launchSetupRNG(state.d_rngStates, state.params.num_agents, 1234ULL);

        float p1 = 0.0f;  // min or mean
        float p2 = 0.0f;  // max or stddev

        if (dist_type == 0) {  // Uniform
            p1 = 0.0f;
            p2 = 1.0f;
        } else {  // Gaussian
            p1 = 0.5f;
            p2 = 0.1f;
        }

        if (use_custom_params) {
            // Convention: Params passed as int scaled by 100
            p1 = static_cast<float>(bench_state.range(2)) / 100.0f;
            p2 = static_cast<float>(bench_state.range(3)) / 100.0f;
        }

        if (dist_type == 0) {
            launchInitializeUniform(state.d_inventory, p1, p2, state.d_rngStates,
                                    state.params.num_agents);
            launchInitializeUniform(state.d_execution_cost, p1, p2, state.d_rngStates,
                                    state.params.num_agents);
        } else {
            launchInitializeNormal(state.d_inventory, p1, p2, state.d_rngStates,
                                   state.params.num_agents);
            launchInitializeNormal(state.d_execution_cost, p1, p2, state.d_rngStates,
                                   state.params.num_agents);
        }
    }
};

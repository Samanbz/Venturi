#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "../../src/simulation.h"
#include "../../src/types.h"
#include "kernels/common.cuh"
#include "kernels/launchers.h"

// Common base class for Venturi tests
class BaseTestFixture : public ::testing::Test {
   protected:
    MarketParams params;

    BaseTestFixture() {
        // Default parameters common to most tests
        params.num_agents = 10000;
        params.num_steps = 100;
        params.time_delta = 1.0f;
        params.price_init = 100.0f;
        params.permanent_impact = 0.1f;
        params.temporary_impact = 0.05f;
        params.congestion_sensitivity = 0.01f;
        params.price_randomness_stddev = 0.1f;
        params.decay_rate = 1.0f;
        params.risk_mean = 0.5f;
        params.risk_stddev = 0.1f;

        // Spatial parameters
        params.sph_smoothing_radius = 1.0f;
        params.hash_table_size = 2048;
        params.mass_alpha = 1.0f;
        params.mass_beta = 0.1f;
    }

    void SetUp() override {
        // Base setup if needed
    }

    void TearDown() override {
        // Base teardown if needed
    }
};

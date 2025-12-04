#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>

#include <vector>

#include "types.h"

// Helper class to manage CUDA memory in tests
class SimulationFixture : public ::testing::Test {
   protected:
    void SetUp() override;
    void TearDown() override;

    void copyInventoryToHost();
    void copyRiskAversionToHost();

    MarketParams params;
    float* d_inventory = nullptr;
    float* d_risk_aversion = nullptr;
    curandState* d_rngStates = nullptr;
    std::vector<float> h_inventory;
    std::vector<float> h_risk_aversion;
};

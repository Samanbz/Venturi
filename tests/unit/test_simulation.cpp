#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "../common/test_common.h"

class SimulationFixture : public BaseTestFixture {
   protected:
    void SetUp() override {
        BaseTestFixture::SetUp();

        // Copy params to device constant memory
        copyParamsToDevice(params);

        // Allocate device memory
        size_t size = params.num_agents * sizeof(float);
        cudaMalloc(&d_inventory, size);
        cudaMalloc(&d_risk_aversion, size);
        cudaMalloc(&d_rngStates, params.num_agents * sizeof(curandState));

        // Initialize RNG states once
        setupRNG(d_rngStates, params.num_agents, 42ULL);

        // Allocate host memory for results
        h_inventory.resize(params.num_agents);
        h_risk_aversion.resize(params.num_agents);
    }

    void TearDown() override {
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_rngStates);
    }

    void copyInventoryToHost() {
        cudaMemcpy(h_inventory.data(), d_inventory, params.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    void copyRiskAversionToHost() {
        cudaMemcpy(h_risk_aversion.data(), d_risk_aversion, params.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    float* d_inventory = nullptr;
    float* d_risk_aversion = nullptr;
    curandState* d_rngStates = nullptr;
    std::vector<float> h_inventory;
    std::vector<float> h_risk_aversion;
};

// Tests for initializeInventories kernel (Exponential Distribution)

TEST_F(SimulationFixture, InventoriesAreAllPositive) {
    // Exponential distribution should produce only positive values
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_GT(h_inventory[i], 0.0f) << "Inventory at index " << i << " should be positive";
    }
}

TEST_F(SimulationFixture, InventoriesMeanApproximatesExpectedValue) {
    // For exponential distribution: E[X] = 1/lambda
    // With decay_rate = 1.0, expected mean ≈ 1.0
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();

    float sum = std::accumulate(h_inventory.begin(), h_inventory.end(), 0.0f);
    float mean = sum / params.num_agents;
    float expected_mean = 1.0f / params.decay_rate;

    // Allow 5% tolerance for statistical variation
    EXPECT_NEAR(mean, expected_mean, expected_mean * 0.05f)
        << "Mean should be close to 1/decay_rate";
}

TEST_F(SimulationFixture, InventoriesWithDifferentDecayRate) {
    // Test with a different decay rate
    params.decay_rate = 2.0f;
    copyParamsToDevice(params);

    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();

    float sum = std::accumulate(h_inventory.begin(), h_inventory.end(), 0.0f);
    float mean = sum / params.num_agents;
    float expected_mean = 1.0f / params.decay_rate;  // 0.5

    EXPECT_NEAR(mean, expected_mean, expected_mean * 0.05f);
}

TEST_F(SimulationFixture, InventoriesAreReproducibleWithSameSeed) {
    // Same seed should produce same results
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();
    std::vector<float> first_run = h_inventory;

    // Re-initialize RNG with same seed
    setupRNG(d_rngStates, params.num_agents, 42ULL);
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_FLOAT_EQ(first_run[i], h_inventory[i])
            << "Same seed should produce identical results at index " << i;
    }
}

TEST_F(SimulationFixture, InventoriesDifferWithDifferentSeeds) {
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();
    std::vector<float> first_run = h_inventory;

    // Re-initialize RNG with different seed
    setupRNG(d_rngStates, params.num_agents, 43ULL);
    launchInitializeInventories(d_inventory, d_rngStates, params.num_agents);
    copyInventoryToHost();

    // At least some values should be different
    int differences = 0;
    for (int i = 0; i < params.num_agents; ++i) {
        if (first_run[i] != h_inventory[i])
            differences++;
    }
    EXPECT_GT(differences, params.num_agents / 2)
        << "Different seeds should produce mostly different results";
}

// Tests for initializeRiskAversions kernel (Normal Distribution)

TEST_F(SimulationFixture, RiskAversionsMeanApproximatesExpectedValue) {
    launchInitializeRiskAversions(d_risk_aversion, d_rngStates, params.num_agents);
    copyRiskAversionToHost();

    float sum = std::accumulate(h_risk_aversion.begin(), h_risk_aversion.end(), 0.0f);
    float mean = sum / params.num_agents;

    // Allow 5% tolerance
    EXPECT_NEAR(mean, params.risk_mean, params.risk_mean * 0.05f)
        << "Mean should be close to risk_mean parameter";
}

TEST_F(SimulationFixture, RiskAversionsStdDevApproximatesExpectedValue) {
    launchInitializeRiskAversions(d_risk_aversion, d_rngStates, params.num_agents);
    copyRiskAversionToHost();

    // Calculate sample mean
    float sum = std::accumulate(h_risk_aversion.begin(), h_risk_aversion.end(), 0.0f);
    float mean = sum / params.num_agents;

    // Calculate sample standard deviation
    float sq_sum = 0.0f;
    for (float val : h_risk_aversion) {
        sq_sum += (val - mean) * (val - mean);
    }
    float stddev = std::sqrt(sq_sum / params.num_agents);

    // Allow 10% tolerance for stddev (more variable than mean)
    EXPECT_NEAR(stddev, params.risk_stddev, params.risk_stddev * 0.10f)
        << "Standard deviation should be close to risk_stddev parameter";
}

TEST_F(SimulationFixture, RiskAversionsAreReproducibleWithSameSeed) {
    launchInitializeRiskAversions(d_risk_aversion, d_rngStates, params.num_agents);
    copyRiskAversionToHost();
    std::vector<float> first_run = h_risk_aversion;

    // Re-initialize RNG with same seed
    setupRNG(d_rngStates, params.num_agents, 42ULL);
    launchInitializeRiskAversions(d_risk_aversion, d_rngStates, params.num_agents);
    copyRiskAversionToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_FLOAT_EQ(first_run[i], h_risk_aversion[i])
            << "Same seed should produce identical results at index " << i;
    }
}

TEST_F(SimulationFixture, RiskAversionsWithDifferentParameters) {
    params.risk_mean = 2.0f;
    params.risk_stddev = 0.5f;
    copyParamsToDevice(params);

    launchInitializeRiskAversions(d_risk_aversion, d_rngStates, params.num_agents);
    copyRiskAversionToHost();

    float sum = std::accumulate(h_risk_aversion.begin(), h_risk_aversion.end(), 0.0f);
    float mean = sum / params.num_agents;

    EXPECT_NEAR(mean, params.risk_mean, params.risk_mean * 0.05f);
}

// Edge case tests

TEST_F(SimulationFixture, SmallNumberOfAgents) {
    // Test with small number of agents (edge case for thread blocks)
    params.num_agents = 10;
    copyParamsToDevice(params);

    h_inventory.resize(params.num_agents);
    h_risk_aversion.resize(params.num_agents);

    float* d_small_inv;
    float* d_small_risk;
    curandState* d_small_rng;
    cudaMalloc(&d_small_inv, params.num_agents * sizeof(float));
    cudaMalloc(&d_small_risk, params.num_agents * sizeof(float));
    cudaMalloc(&d_small_rng, params.num_agents * sizeof(curandState));

    setupRNG(d_small_rng, params.num_agents, 12345ULL);
    launchInitializeInventories(d_small_inv, d_small_rng, params.num_agents);
    launchInitializeRiskAversions(d_small_risk, d_small_rng, params.num_agents);

    cudaMemcpy(h_inventory.data(), d_small_inv, params.num_agents * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_risk_aversion.data(), d_small_risk, params.num_agents * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Just verify no crashes and all values are valid (not NaN/Inf)
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_FALSE(std::isnan(h_inventory[i]));
        EXPECT_FALSE(std::isinf(h_inventory[i]));
        EXPECT_FALSE(std::isnan(h_risk_aversion[i]));
        EXPECT_FALSE(std::isinf(h_risk_aversion[i]));
    }

    cudaFree(d_small_inv);
    cudaFree(d_small_risk);
    cudaFree(d_small_rng);
}

TEST_F(SimulationFixture, LargeNumberOfAgents) {
    // Test with large number of agents (stress test)
    params.num_agents = 1000000;
    copyParamsToDevice(params);

    float* d_large_inv;
    curandState* d_large_rng;
    cudaMalloc(&d_large_inv, params.num_agents * sizeof(float));
    cudaMalloc(&d_large_rng, params.num_agents * sizeof(curandState));

    setupRNG(d_large_rng, params.num_agents, 12345ULL);
    launchInitializeInventories(d_large_inv, d_large_rng, params.num_agents);

    // Spot check a few values
    std::vector<float> sample(100);
    cudaMemcpy(sample.data(), d_large_inv, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    for (float val : sample) {
        EXPECT_GT(val, 0.0f);
        EXPECT_FALSE(std::isnan(val));
    }

    cudaFree(d_large_inv);
    cudaFree(d_large_rng);
}

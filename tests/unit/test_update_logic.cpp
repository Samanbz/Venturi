#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../../src/simulation.h"
#include "../common/test_common.h"

class UpdateLogicFixture : public BaseTestFixture {
   protected:
    void SetUp() override {
        BaseTestFixture::SetUp();
        params.time_delta = 1.0f;
        params.price_randomness_stddev = 0.1f;

        allocateDeviceMemory();
    }

    void TearDown() override { freeDeviceMemory(); }

    void allocateDeviceMemory() {
        size_t size = params.num_agents * sizeof(float);
        cudaMalloc(&d_inventory, size);
        cudaMalloc(&d_risk_aversion, size);
        cudaMalloc(&d_local_density, size);
        cudaMalloc(&d_speed_term_1, size);
        cudaMalloc(&d_speed_term_2, size);
        cudaMalloc(&d_speed, size);
        cudaMalloc(&d_execution_cost, size);
        cudaMalloc(&d_cash, size);
        cudaMalloc(&d_agent_indices, params.num_agents * sizeof(int));

        // Initialize identity mapping for indices
        std::vector<int> h_indices(params.num_agents);
        std::iota(h_indices.begin(), h_indices.end(), 0);
        cudaMemcpy(d_agent_indices, h_indices.data(), params.num_agents * sizeof(int),
                   cudaMemcpyHostToDevice);

        h_inventory.resize(params.num_agents);
        h_risk_aversion.resize(params.num_agents);
        h_local_density.resize(params.num_agents);
        h_speed_term_1.resize(params.num_agents);
        h_speed_term_2.resize(params.num_agents);
        h_speed.resize(params.num_agents);
        h_execution_cost.resize(params.num_agents);
        h_cash.resize(params.num_agents);
    }

    void freeDeviceMemory() {
        if (d_inventory)
            cudaFree(d_inventory);
        if (d_risk_aversion)
            cudaFree(d_risk_aversion);
        if (d_local_density)
            cudaFree(d_local_density);
        if (d_speed_term_1)
            cudaFree(d_speed_term_1);
        if (d_speed_term_2)
            cudaFree(d_speed_term_2);
        if (d_speed)
            cudaFree(d_speed);
        if (d_execution_cost)
            cudaFree(d_execution_cost);
        if (d_cash)
            cudaFree(d_cash);
        if (d_agent_indices)
            cudaFree(d_agent_indices);
    }

    void copyToDevice() {
        size_t size = params.num_agents * sizeof(float);
        cudaMemcpy(d_inventory, h_inventory.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_risk_aversion, h_risk_aversion.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_local_density, h_local_density.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_speed_term_1, h_speed_term_1.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_speed_term_2, h_speed_term_2.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cash, h_cash.data(), size, cudaMemcpyHostToDevice);
    }

    void copyFromDevice() {
        size_t size = params.num_agents * sizeof(float);
        cudaMemcpy(h_speed.data(), d_speed, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_inventory.data(), d_inventory, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_execution_cost.data(), d_execution_cost, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cash.data(), d_cash, size, cudaMemcpyDeviceToHost);
    }

    float* d_inventory = nullptr;
    float* d_risk_aversion = nullptr;
    float* d_local_density = nullptr;
    float* d_speed_term_1 = nullptr;
    float* d_speed_term_2 = nullptr;
    float* d_speed = nullptr;
    float* d_execution_cost = nullptr;
    float* d_cash = nullptr;
    int* d_agent_indices = nullptr;

    std::vector<float> h_inventory;
    std::vector<float> h_risk_aversion;
    std::vector<float> h_local_density;
    std::vector<float> h_speed_term_1;
    std::vector<float> h_speed_term_2;
    std::vector<float> h_speed;
    std::vector<float> h_execution_cost;
    std::vector<float> h_cash;

    // Helpers to access Simulation private members
    void setPressure(Simulation& sim, float p) { sim.state_.pressure = p; }

    float getPrice(Simulation& sim) { return sim.state_.price; }

    void runUpdatePrice(Simulation& sim) { sim.updatePrice(); }
    void getInventory(Simulation& sim, std::vector<float>& out_inventory) {
        cudaMemcpy(out_inventory.data(), sim.state_.d_inventory, params.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    void setInventory(Simulation& sim, const std::vector<float>& in_inventory) {
        cudaMemcpy(sim.state_.d_inventory, in_inventory.data(), params.num_agents * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    void runStep(Simulation& sim) { sim.step(); }
};

// Inventory Directionality and Liquidation Logic
TEST_F(UpdateLogicFixture, InventoryDirectionalityAndIntegration) {
    params.num_agents = 1;
    params.time_delta = 0.1f;
    params.temporary_impact = 0.05f;
    params.permanent_impact = 0.1f;
    params.num_steps = 10;
    copyParamsToDevice(params);

    // Setup agent with positive inventory
    h_inventory[0] = 100.0f;
    h_risk_aversion[0] = 0.5f;
    h_local_density[0] = 1.0f;

    copyToDevice();

    // Calculate speed terms
    int dt = 0;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    // Calculate pressure
    float pressure = 0.0f;
    launchComputePressure(d_speed_term_1, d_speed_term_2, &pressure, params.num_agents);

    // Update
    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    // Backup old inventory
    float old_inventory = h_inventory[0];

    copyFromDevice();

    // Direction Check: For Q>0, assert speed < 0
    EXPECT_LT(h_speed[0], 0.0f)
        << "Agent with positive inventory should be selling (negative speed)";

    // Integration Check: Assert that inventory_new equals inventory_old + (speed * dt)
    float expected_inventory = old_inventory + h_speed[0] * params.time_delta;
    EXPECT_NEAR(h_inventory[0], expected_inventory, 1e-5f) << "Inventory integration mismatch";
}

// Ticket 5: Verify Deterministic Impact vs. Stochastic Noise in Price Evolution
TEST_F(UpdateLogicFixture, PriceDeterministicImpact) {
    params.price_randomness_stddev = 0.0f;
    params.permanent_impact = 0.1f;
    params.time_delta = 1.0f;
    params.price_init = 100.0f;

    Simulation sim(params);

    // Force pressure
    setPressure(sim, 10.0f);
    float old_price = getPrice(sim);
    runUpdatePrice(sim);
    float new_price = getPrice(sim);

    float expected_change = params.permanent_impact * 10.0f * params.time_delta;
    EXPECT_NEAR(new_price - old_price, expected_change, 1e-5f);
}

TEST_F(UpdateLogicFixture, PriceStochasticDistribution) {
    params.price_randomness_stddev = 1.0f;
    params.permanent_impact = 0.0f;
    params.time_delta = 1.0f;
    params.price_init = 100.0f;

    Simulation sim(params);
    setPressure(sim, 10.0f);  // Should have no effect

    std::vector<float> changes;
    int N = 10000;
    for (int i = 0; i < N; ++i) {
        float p1 = getPrice(sim);
        runUpdatePrice(sim);
        float p2 = getPrice(sim);
        changes.push_back(p2 - p1);
    }

    // Calculate mean and stddev
    double sum = std::accumulate(changes.begin(), changes.end(), 0.0);
    double mean = sum / N;

    double sq_sum = std::inner_product(changes.begin(), changes.end(), changes.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / N - mean * mean);

    EXPECT_NEAR(mean, 0.0, 0.05);  // Allow some noise
    EXPECT_NEAR(stdev, params.price_randomness_stddev * sqrt(params.time_delta), 0.05);
}

// Execution Cost Scaling and Local Density Dependency
TEST_F(UpdateLogicFixture, ExecutionCostScaling) {
    params.num_agents = 2;
    params.temporary_impact = 0.1f;
    params.congestion_sensitivity = 1.0f;  // beta
    copyParamsToDevice(params);

    // Agent 0: Low density
    h_local_density[0] = 0.0f;

    // Agent 1: High density
    h_local_density[1] = 10.0f;

    // We want identical speeds.
    // speed = pressure * term1 - term2.
    // If we set pressure = 0, speed = -term2.
    // term2 = (-2 * sqrt(kappa * phi) * Q) / (2 * kappa * (1 + beta * rho))
    // To get same speed, we need term2 to be same.
    // So Q / (1 + beta * rho) must be same.
    // Q0 / 1 = Q1 / (1 + 10) = Q1 / 11.
    // So Q1 = 11 * Q0.

    h_risk_aversion[0] = 0.5f;
    h_risk_aversion[1] = 0.5f;

    h_inventory[0] = 10.0f;
    h_inventory[1] = 10.0f * (1.0f + params.congestion_sensitivity * h_local_density[1]);

    copyToDevice();

    int dt = 0;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    // Force pressure to 0 to isolate term2
    float pressure = 0.0f;

    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    copyFromDevice();

    // Verify speeds are approximately equal
    EXPECT_NEAR(h_speed[0], h_speed[1], 1e-4f) << "Speeds should be identical for this test setup";

    // Congestion Penalty: Assert cost(HighDensity) > cost(LowDensity) (magnitude)
    EXPECT_GT(std::abs(h_execution_cost[1]), std::abs(h_execution_cost[0]))
        << "High density should incur higher execution cost magnitude";

    // Linearity check: cost = kappa * (1 + beta * rho) * speed
    // cost0 = kappa * 1 * speed
    // cost1 = kappa * 11 * speed
    // cost1 should be 11 * cost0
    float expected_ratio = 1.0f + params.congestion_sensitivity * h_local_density[1];
    EXPECT_NEAR(h_execution_cost[1] / h_execution_cost[0], expected_ratio, 1e-4f)
        << "Execution cost should scale with density";
}

// Integration Test: Multi-step simulation
TEST_F(UpdateLogicFixture, MultiStepSimulation) {
    params.num_agents = 10;
    params.num_steps = 50;
    params.time_delta = 0.1f;
    params.permanent_impact = 0.01f;
    params.temporary_impact = 0.01f;
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.0f;  // Deterministic for checking trends

    Simulation sim(params);

    // Initialize agents with positive inventory
    std::vector<float> initial_inventory(params.num_agents, 100.0f);
    setInventory(sim, initial_inventory);

    float initial_price = getPrice(sim);

    // Run simulation
    for (int i = 0; i < params.num_steps; ++i) {
        runStep(sim);
    }

    float final_price = getPrice(sim);

    std::vector<float> final_inventory(params.num_agents);
    getInventory(sim, final_inventory);

    // Price should fall due to selling pressure
    EXPECT_LT(final_price, initial_price) << "Price should fall as agents liquidate";

    // Inventories should shrink
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_LT(final_inventory[i], 100.0f) << "Agent " << i << " should have reduced inventory";
    }

    // Total inventory reduction
    float total_initial = 100.0f * params.num_agents;
    float total_final = std::accumulate(final_inventory.begin(), final_inventory.end(), 0.0f);
    EXPECT_LT(total_final, total_initial) << "Total inventory should decrease";
}

TEST_F(UpdateLogicFixture, CashAccumulation) {
    params.num_agents = 1;
    params.time_delta = 0.1f;
    params.temporary_impact = 0.1f;
    params.permanent_impact = 0.0f;
    copyParamsToDevice(params);

    h_inventory[0] = 100.0f;
    h_risk_aversion[0] = 0.5f;
    h_local_density[0] = 0.0f;  // No congestion for simple math
    h_cash[0] = 1000.0f;        // Initial cash

    copyToDevice();

    // We need to set up speed terms such that we get a known speed.
    // Or we can just let the kernel calculate speed and then verify cash based on that speed.

    int dt = 0;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    float pressure = 0.0f;  // Assume no pressure for simplicity, or calculate it.
    // If pressure is 0, speed = -term2.

    float price = 100.0f;

    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, price,
                           params.num_agents);

    cudaDeviceSynchronize();
    copyFromDevice();

    float speed = h_speed[0];
    float expected_cash_change =
        -speed * (price + params.temporary_impact * speed) * params.time_delta;
    float expected_cash = 1000.0f + expected_cash_change;

    EXPECT_NEAR(h_cash[0], expected_cash, 1e-3f);
}

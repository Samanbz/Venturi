#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../common/test_common.h"

class SpeedPressureFixture : public BaseTestFixture {
   protected:
    void SetUp() override {
        BaseTestFixture::SetUp();

        // Copy params to device constant memory
        copyParamsToDevice(params);

        // Allocate device memory
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

        // Allocate host memory
        h_inventory.resize(params.num_agents);
        h_risk_aversion.resize(params.num_agents);
        h_local_density.resize(params.num_agents);
        h_speed_term_1.resize(params.num_agents);
        h_speed_term_2.resize(params.num_agents);
        h_speed.resize(params.num_agents);
    }

    void TearDown() override {
        cudaFree(d_inventory);
        cudaFree(d_risk_aversion);
        cudaFree(d_local_density);
        cudaFree(d_speed_term_1);
        cudaFree(d_speed_term_2);
        cudaFree(d_speed);
        cudaFree(d_execution_cost);
        cudaFree(d_cash);
        cudaFree(d_agent_indices);
    }

    void copyToDevice() {
        size_t size = params.num_agents * sizeof(float);
        cudaMemcpy(d_inventory, h_inventory.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_risk_aversion, h_risk_aversion.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_local_density, h_local_density.data(), size, cudaMemcpyHostToDevice);
    }

    void copyFromDevice() {
        size_t size = params.num_agents * sizeof(float);
        cudaMemcpy(h_speed_term_1.data(), d_speed_term_1, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_speed_term_2.data(), d_speed_term_2, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_speed.data(), d_speed, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_inventory.data(), d_inventory, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_execution_cost.data(), d_execution_cost, size, cudaMemcpyDeviceToHost);
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
};

// Closed-Loop Invariant Testing for Pressure Dynamics
TEST_F(SpeedPressureFixture, PressureConsistency) {
    params.num_agents = 1024;
    copyParamsToDevice(params);

    // Initialize with random data
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = 10.0f + (float) i / params.num_agents;
        h_risk_aversion[i] = 0.5f + (float) i / params.num_agents * 0.1f;
        h_local_density[i] = 1.0f + (float) i / params.num_agents;
    }
    copyToDevice();

    int dt = 0;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    float pressure = 0.0f;
    launchComputePressure(d_speed_term_1, d_speed_term_2, &pressure, params.num_agents);

    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    copyFromDevice();

    // Sum speeds on host
    float sum_speed = 0.0f;
    for (float s : h_speed) {
        sum_speed += s;
    }

    // The aggregate market pressure matches the sum of individual agent speeds
    EXPECT_NEAR(pressure, sum_speed, 1e-2f) << "Pressure should equal sum of speeds";
}

// CPU Oracle for Kernel Numerical Verification
TEST_F(SpeedPressureFixture, NumericalAccuracy) {
    params.num_agents = 10;
    copyParamsToDevice(params);

    // Initialize with deterministic data
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = 10.0f + i;
        h_risk_aversion[i] = 0.5f + i * 0.05f;
        h_local_density[i] = 1.0f + i * 0.1f;
    }

    // Backup initial inventory for CPU oracle
    std::vector<float> initial_inventory = h_inventory;

    copyToDevice();

    int dt = 5;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    float gpu_pressure = 0.0f;
    launchComputePressure(d_speed_term_1, d_speed_term_2, &gpu_pressure, params.num_agents);

    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           gpu_pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    copyFromDevice();

    // CPU Oracle
    std::vector<float> cpu_term1(params.num_agents);
    std::vector<float> cpu_term2(params.num_agents);
    std::vector<float> cpu_speed(params.num_agents);
    float cpu_pressure = 0.0f;

    float sum_term1 = 0.0f;
    float sum_term2 = 0.0f;

    for (int i = 0; i < params.num_agents; ++i) {
        float personal_decay_rate = sqrtf(h_risk_aversion[i] / params.temporary_impact);
        float local_temporary_impact =
            params.temporary_impact * (1.0f + params.congestion_sensitivity * h_local_density[i]);

        // Term 1
        float num =
            params.permanent_impact * (1.0f - expf(-personal_decay_rate * (params.num_steps - dt)));
        float den = 2.0f * local_temporary_impact * personal_decay_rate;
        cpu_term1[i] = num / den;

        // Term 2
        float term2_num =
            2.0f * sqrtf(params.temporary_impact * h_risk_aversion[i]) * initial_inventory[i];
        float term2_den = 2.0f * local_temporary_impact;
        cpu_term2[i] = term2_num / term2_den;

        sum_term1 += cpu_term1[i];
        sum_term2 += cpu_term2[i];
    }

    cpu_pressure = -sum_term2 / (1.0f - sum_term1);

    for (int i = 0; i < params.num_agents; ++i) {
        cpu_speed[i] = cpu_pressure * cpu_term1[i] - cpu_term2[i];
    }

    // Verify
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_NEAR(h_speed_term_1[i], cpu_term1[i], 1e-4f) << "Term 1 mismatch at index " << i;
        EXPECT_NEAR(h_speed_term_2[i], cpu_term2[i], 1e-4f) << "Term 2 mismatch at index " << i;
        EXPECT_NEAR(h_speed[i], cpu_speed[i], 1e-4f) << "Speed mismatch at index " << i;
    }
    EXPECT_NEAR(gpu_pressure, cpu_pressure, 1e-4f) << "Pressure mismatch";
}

// Stress Test Boundary Conditions and Singularities
TEST_F(SpeedPressureFixture, BoundaryConditions_ZeroDensity) {
    params.num_agents = 10;
    copyParamsToDevice(params);

    // Set extremely low density
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = 10.0f;
        h_risk_aversion[i] = 0.5f;
        h_local_density[i] = 1e-9f;  // Very small density
    }
    copyToDevice();

    int dt = 0;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    float pressure = 0.0f;
    launchComputePressure(d_speed_term_1, d_speed_term_2, &pressure, params.num_agents);
    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    copyFromDevice();

    // Check for NaN/Inf
    EXPECT_FALSE(std::isnan(pressure));
    EXPECT_FALSE(std::isinf(pressure));
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_FALSE(std::isnan(h_speed[i]));
        EXPECT_FALSE(std::isinf(h_speed[i]));
    }
}

TEST_F(SpeedPressureFixture, BoundaryConditions_TimeMaturity) {
    params.num_agents = 10;
    copyParamsToDevice(params);

    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = 10.0f;
        h_risk_aversion[i] = 0.5f;
        h_local_density[i] = 1.0f;
    }
    copyToDevice();

    // Set dt to final step
    int dt = params.num_steps;
    launchComputeSpeedTerms(d_risk_aversion, d_local_density, d_inventory, d_speed_term_1,
                            d_speed_term_2, dt, params.num_agents);

    float pressure = 0.0f;
    launchComputePressure(d_speed_term_1, d_speed_term_2, &pressure, params.num_agents);
    launchUpdateAgentState(d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices,
                           pressure, d_speed, d_inventory, d_execution_cost, d_cash, 100.0f,
                           params.num_agents);
    cudaDeviceSynchronize();

    copyFromDevice();

    // Check for NaN/Inf
    EXPECT_FALSE(std::isnan(pressure));
    EXPECT_FALSE(std::isinf(pressure));
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_FALSE(std::isnan(h_speed[i]));
        EXPECT_FALSE(std::isinf(h_speed[i]));
    }
}

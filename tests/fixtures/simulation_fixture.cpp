#include "simulation_fixture.h"

#include "simulation.h"

void SimulationFixture::SetUp() {
    // Default test parameters
    params.num_agents = 10000;
    params.num_steps = 100;
    params.price_init = 100.0f;
    params.permanent_impact = 0.1f;
    params.temporary_impact = 0.05f;
    params.congestion_sensitivity = 0.01f;
    params.decay_rate = 1.0f;
    params.risk_mean = 0.5f;
    params.risk_stddev = 0.1f;

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

void SimulationFixture::TearDown() {
    cudaFree(d_inventory);
    cudaFree(d_risk_aversion);
    cudaFree(d_rngStates);
}

void SimulationFixture::copyInventoryToHost() {
    cudaMemcpy(h_inventory.data(), d_inventory, params.num_agents * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void SimulationFixture::copyRiskAversionToHost() {
    cudaMemcpy(h_risk_aversion.data(), d_risk_aversion, params.num_agents * sizeof(float),
               cudaMemcpyDeviceToHost);
}
#include "simulation.h"

#include <cuda_runtime.h>

#include <ctime>

Simulation::Simulation(const MarketParams& params) : params_(params) {
    // Copy MarketParams to device constant memory
    copyParamsToDevice(params_);

    // Initialize MarketState scalars (these stay on host)
    state_.dt = 0;
    state_.d_price = params.price_init;
    state_.d_pressure = 0.0f;

    // Allocate device memory for agent-specific arrays
    size_t size = params.num_agents * sizeof(float);
    cudaMalloc(&state_.d_inventory, size);
    cudaMalloc(&state_.d_cash, size);
    cudaMalloc(&state_.d_speed, size);
    cudaMalloc(&state_.d_density, size);
    cudaMalloc(&state_.d_risk_aversion, size);
    cudaMalloc(&state_.d_rngStates, params.num_agents * sizeof(curandState));

    // Initialize RNG states once with time-based seed
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    setupRNG(state_.d_rngStates, params.num_agents, seed);

    // Initialize device memory using persistent RNG states
    launchInitializeInventories(state_.d_inventory, state_.d_rngStates, params.num_agents);
    launchInitializeRiskAversions(state_.d_risk_aversion, state_.d_rngStates, params.num_agents);
    cudaMemset(state_.d_cash, 0, size);
    cudaMemset(state_.d_speed, 0, size);
    cudaMemset(state_.d_density, 0, size);
}

void Simulation::step() {
    state_.dt++;

    // computeLocalDensities();
    // computePressure();
    // computeVelocities();
    // computePrice();
    // computeQuantities();
    // computeExecutionCosts();
}

Simulation::~Simulation() {
    // Free device memory
    cudaFree(state_.d_inventory);
    cudaFree(state_.d_cash);
    cudaFree(state_.d_speed);
    cudaFree(state_.d_density);
    cudaFree(state_.d_risk_aversion);
    cudaFree(state_.d_rngStates);
}
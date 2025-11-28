#include "simulation.h"

#include <cuda_runtime.h>

Simulation::Simulation(const MarketParams& params) : params_(params) {
    // Initialize MarketState
    state_.dt = 0;
    state_.d_price = params.price_init;
    state_.d_pressure = 0.0f;

    // Allocate device memory for agent-specific arrays
    size_t size = params.num_agents * sizeof(float);
    cudaMalloc(&state_.d_inventory, size);
    cudaMalloc(&state_.d_cash, size);
    cudaMalloc(&state_.d_speed, size);
    cudaMalloc(&state_.d_density, size);

    // Initialize device memory to zero
    initializeInventories(params.inventory_mean, params.inventory_stddev);
    cudaMemset(state_.d_cash, 0, size);
    cudaMemset(state_.d_speed, 0, size);
    computeLocalDensities();
}

void Simulation::initializeInventories(const float mean, const float stddev) {
    // Kernel to initialize inventories with random values
}

void Simulation::computeLocalDensities() {
    // Kernel to compute local densities based on inventories and execution costs
}

void Simulation::computeInventories() {
    // Kernel to update inventories based on trading speeds
}

void Simulation::computeExecutionCosts() {
    // Kernel to compute execution costs
}

void Simulation::step() {
    state_.dt++;

    computeLocalDensities();
    // these two could be combined
    // computePressure();
    // computeSpeeds();
    computeInventories();
    computeExecutionCosts();
}

Simulation::~Simulation() {
    // Free device memory
    cudaFree(state_.d_inventory);
    cudaFree(state_.d_cash);
    cudaFree(state_.d_speed);
    cudaFree(state_.d_density);
}
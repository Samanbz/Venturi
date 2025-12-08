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
    cudaMalloc(&state_.d_local_density, size);
    cudaMalloc(&state_.d_risk_aversion, size);
    cudaMalloc(&state_.d_execution_cost, size);
    cudaMalloc(&state_.d_rngStates, params.num_agents * sizeof(curandState));

    cudaMalloc(&state_.d_cell_start, params.num_agents * sizeof(int));
    cudaMalloc(&state_.d_cell_end, params.num_agents * sizeof(int));
    cudaMalloc(&state_.d_agent_hash, params.num_agents * sizeof(int));
    cudaMalloc(&state_.d_agent_index, params.num_agents * sizeof(int));

    // Initialize RNG states once with time-based seed
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    setupRNG(state_.d_rngStates, params.num_agents, seed);

    // Initialize device memory using persistent RNG states
    launchInitializeExponential(state_.d_inventory, params.decay_rate, state_.d_rngStates,
                                params.num_agents);
    launchInitializeNormal(state_.d_risk_aversion, params.risk_mean, params.risk_stddev,
                           state_.d_rngStates, params.num_agents);
    cudaMemset(state_.d_cash, 0, size);
    cudaMemset(state_.d_speed, 0, size);
    cudaMemset(state_.d_local_density, 0, size);
    cudaMemset(state_.d_execution_cost, 0, size);
}

void Simulation::computeLocalDensities() {
    // Reset cell bounds for spatial hashing. Critical step, since many cells may be empty.
    cudaMemset(state_.d_cell_start, -1, params_.hash_table_size * sizeof(int));
    cudaMemset(state_.d_cell_end, -1, params_.hash_table_size * sizeof(int));
    // Compute spatial hashes for all agents based on their current inventories and execution costs
    launchCalculateSpatialHash(state_.d_inventory, state_.d_execution_cost, state_.d_agent_hash,
                               state_.d_agent_index, params_.num_agents);

    // Sort agents by spatial hash
    launchSortByKey(state_.d_agent_hash, state_.d_agent_index, params_.num_agents);

    // Identify the start and end indices of agents within each spatial grid cell
    launchFindCellBounds(state_.d_agent_hash, state_.d_cell_start, state_.d_cell_end,
                         params_.num_agents);

    launchReorderData(state_.d_agent_index, state_.d_inventory, state_.d_execution_cost,
                      state_.d_cash, state_.d_speed, state_.d_inventory_sorted,
                      state_.d_execution_cost_sorted, state_.d_cash_sorted, state_.d_speed_sorted,
                      params_.num_agents);

    // Compute local densities for each agent using SPH within their spatial cells
    launchComputeLocalDensities(state_.d_inventory_sorted, state_.d_execution_cost_sorted,
                                state_.d_cell_start, state_.d_cell_end, state_.d_local_density,
                                params_.num_agents);
}

void Simulation::step() {
    state_.dt++;

    computeLocalDensities();
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
    cudaFree(state_.d_local_density);
    cudaFree(state_.d_risk_aversion);
    cudaFree(state_.d_rngStates);
}
#include "simulation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "kernels/common.cuh"
#include "kernels/launchers.h"

Simulation::Simulation(
    const MarketParams& params, float* vk_X, float* vk_Y, PlotVar xVar, PlotVar yVar)
    : params_(params), rng(std::random_device{}()), normal_dist(0.0f, 1.0f) {
    if (vk_X == nullptr ^ vk_Y == nullptr) {
        throw std::invalid_argument("Both vk_X and vk_Y must be provided or both must be nullptr");
    }

    externalMemoryProvided = (vk_X != nullptr && vk_Y != nullptr);

    // Copy MarketParams to device constant memory
    copyParamsToDevice(params_);

    // Initialize MarketState scalars (these stay on host)
    state_.dt = 0;
    state_.price = params.price_init;
    state_.pressure = 0.0f;

    // Allocate device memory for agent-specific arrays
    size_t size = params.num_agents * sizeof(float);

    // Helper to get pointer reference by enum
    auto getPtrRef = [&](PlotVar var) -> float*& {
        switch (var) {
            case PlotVar::Inventory:
                return state_.d_inventory;
            case PlotVar::ExecutionCost:
                return state_.d_execution_cost;
            case PlotVar::Cash:
                return state_.d_cash;
            case PlotVar::Speed:
                return state_.d_speed;
            case PlotVar::RiskAversion:
                return state_.d_risk_aversion;
            case PlotVar::LocalDensity:
                return state_.d_local_density;
            default:
                throw std::runtime_error("Unknown PlotVar");
        }
    };

    // Assign external memory if applicable
    if (externalMemoryProvided) {
        getPtrRef(xVar) = vk_X;
        d_plot_x = vk_X;

        getPtrRef(yVar) = vk_Y;
        d_plot_y = vk_Y;
    }

    // Allocate the rest
    std::vector<PlotVar> allVars = {PlotVar::Inventory,    PlotVar::ExecutionCost,
                                    PlotVar::Cash,         PlotVar::Speed,
                                    PlotVar::RiskAversion, PlotVar::LocalDensity};

    for (auto var : allVars) {
        float*& ptr = getPtrRef(var);
        if (ptr == nullptr) {
            cudaMalloc(&ptr, size);
        }
    }

    // Allocate sorted arrays and intermediate buffers
    cudaMalloc(&state_.d_inventory_sorted, size);
    cudaMalloc(&state_.d_execution_cost_sorted, size);
    cudaMalloc(&state_.d_speed_term_1, size);
    cudaMalloc(&state_.d_speed_term_2, size);

    cudaMalloc(&state_.d_rngStates, params.num_agents * sizeof(curandState));

    cudaMalloc(&state_.d_cell_start, params.hash_table_size * sizeof(int));
    cudaMalloc(&state_.d_cell_end, params.hash_table_size * sizeof(int));
    cudaMalloc(&state_.d_agent_hash, params.num_agents * sizeof(int));
    cudaMalloc(&state_.d_agent_index, params.num_agents * sizeof(int));

    // Initialize RNG states once with time-based seed
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    launchSetupRNG(state_.d_rngStates, params.num_agents, seed);

    // Initialize device memory using persistent RNG states
    launchInitializeExponential(state_.d_inventory, params.decay_rate, state_.d_rngStates,
                                params.num_agents);
    // Initialize from log-normal to avoid negative values leading to NaNs later
    launchInitializeLogNormal(state_.d_risk_aversion, params.risk_mean, params.risk_stddev,
                              state_.d_rngStates, params.num_agents);

    // Initialize others to 0
    if (state_.d_cash != vk_X && state_.d_cash != vk_Y)
        cudaMemset(state_.d_cash, 0, size);
    if (state_.d_speed != vk_X && state_.d_speed != vk_Y)
        cudaMemset(state_.d_speed, 0, size);
    if (state_.d_execution_cost != vk_X && state_.d_execution_cost != vk_Y)
        cudaMemset(state_.d_execution_cost, 0, size);

    // If mapped to external memory, we should still initialize them if they are outputs/state
    // But if they are inputs initialized by kernels above (like inventory), we are good.
    // Cash starts at 0.
    if (state_.d_cash == vk_X || state_.d_cash == vk_Y)
        cudaMemset(state_.d_cash, 0, size);
    if (state_.d_execution_cost == vk_X || state_.d_execution_cost == vk_Y)
        cudaMemset(state_.d_execution_cost, 0, size);
}

BoundaryPair Simulation::getBoundaries() const {
    // TODO: Optimize by using thrust or device reduction

    float min_y, max_y;
    float min_x, max_x;

    // Copy inventories and execution costs back to host for boundary calculation
    std::vector<float> h_y(params_.num_agents);
    std::vector<float> h_x(params_.num_agents);

    if (d_plot_y) {
        cudaMemcpy(h_y.data(), d_plot_y, params_.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        // Fallback if not set (e.g. unit tests)
        cudaMemcpy(h_y.data(), state_.d_inventory, params_.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    if (d_plot_x) {
        cudaMemcpy(h_x.data(), d_plot_x, params_.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        // Fallback
        cudaMemcpy(h_x.data(), state_.d_execution_cost, params_.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    auto [y_min_it, y_max_it] = std::minmax_element(h_y.begin(), h_y.end());
    auto [x_min_it, x_max_it] = std::minmax_element(h_x.begin(), h_x.end());

    min_y = *y_min_it;
    max_y = *y_max_it;
    min_x = *x_min_it;
    max_x = *x_max_it;

    return {{min_y, max_y}, {min_x, max_x}};
}

void Simulation::importSemaphores(int fdWait, int fdSignal) {
    // Import external semaphores for CUDA synchronization
    cudaExternalSemaphoreHandleDesc semDescWait{};
    semDescWait.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    semDescWait.handle.fd = fdWait;

    cudaExternalSemaphoreHandleDesc semDescSignal{};
    semDescSignal.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    semDescSignal.handle.fd = fdSignal;

    cudaError_t err1 = cudaImportExternalSemaphore(&cudaWaitSemaphore, &semDescWait);
    cudaError_t err2 = cudaImportExternalSemaphore(&cudaSignalSemaphore, &semDescSignal);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("Failed to import external semaphores for CUDA");
    }
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

    launchReorderData(state_.d_agent_index, params_.num_agents, state_.d_inventory,
                      state_.d_inventory_sorted, state_.d_execution_cost,
                      state_.d_execution_cost_sorted);

    // Compute local densities for each agent using SPH within their spatial cells
    launchComputeLocalDensities(state_.d_inventory_sorted, state_.d_execution_cost_sorted,
                                state_.d_cell_start, state_.d_cell_end, state_.d_agent_index,
                                state_.d_local_density, params_.num_agents);
}

void Simulation::computePressure() {
    launchComputeSpeedTerms(state_.d_risk_aversion, state_.d_local_density, state_.d_inventory,
                            state_.d_speed_term_1, state_.d_speed_term_2, state_.dt,
                            params_.num_agents);

    // Compute the pressure based on the speed terms
    launchComputePressure(state_.d_speed_term_1, state_.d_speed_term_2, &state_.pressure,
                          params_.num_agents);
}

void Simulation::updateAgentState() {
    // Compute the trading speed for each agent based on their risk aversion, local density, and
    // pressure
    launchUpdateAgentState(state_.d_speed_term_1, state_.d_speed_term_2, state_.d_local_density,
                           state_.d_agent_index, state_.pressure, state_.d_speed,
                           state_.d_inventory, state_.d_execution_cost, state_.d_cash, state_.price,
                           params_.num_agents);
}

void Simulation::updatePrice() {
    state_.price +=
        params_.permanent_impact * state_.pressure * params_.time_delta +
        params_.price_randomness_stddev * this->normal_dist(rng) * sqrt(params_.time_delta);
}

void Simulation::step() {
    // Skip wait on first step to avoid deadlock
    if (state_.dt > 0 && cudaWaitSemaphore != nullptr) {
        cudaExternalSemaphoreWaitParams waitParams{};  // Defaults are fine for binary semaphores
        cudaWaitExternalSemaphoresAsync(&cudaWaitSemaphore, &waitParams, 1,
                                        0);  // 0 = Default Stream
    }

    state_.dt++;
    computeLocalDensities();
    computePressure();
    updateAgentState();
    updatePrice();

    if (cudaSignalSemaphore != nullptr) {
        cudaExternalSemaphoreSignalParams signalParams{};
        cudaSignalExternalSemaphoresAsync(&cudaSignalSemaphore, &signalParams, 1, 0);
    }
}

void Simulation::run() {
    for (int i = 0; i < params_.num_steps; ++i) {
        step();
    }
}

Simulation::~Simulation() {
    // Free device memory
    if (externalMemoryProvided) {
        // If external memory was provided, do not free those pointers
        state_.d_inventory = nullptr;
        state_.d_execution_cost = nullptr;
    } else {
        cudaFree(state_.d_inventory);
        cudaFree(state_.d_execution_cost);
    }

    cudaFree(state_.d_cash);
    cudaFree(state_.d_speed);
    cudaFree(state_.d_local_density);
    cudaFree(state_.d_risk_aversion);
    cudaFree(state_.d_rngStates);

    cudaFree(state_.d_inventory_sorted);
    cudaFree(state_.d_execution_cost_sorted);
    cudaFree(state_.d_speed_term_1);
    cudaFree(state_.d_speed_term_2);

    cudaFree(state_.d_cell_start);
    cudaFree(state_.d_cell_end);
    cudaFree(state_.d_agent_hash);
    cudaFree(state_.d_agent_index);
}

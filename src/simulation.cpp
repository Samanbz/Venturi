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

Simulation::Simulation(const MarketParams& params,
                       float* vk_X,
                       float* vk_Y,
                       float* vk_Color,
                       PlotVar xVar,
                       PlotVar yVar,
                       PlotVar colorVar)
    : params_(params), rng(std::random_device{}()), normal_dist(0.0f, 1.0f) {
    if ((vk_X != nullptr) != (vk_Y != nullptr) || (vk_X != nullptr) != (vk_Color != nullptr)) {
        throw std::invalid_argument(
            "All of vk_X, vk_Y, and vk_Color must be provided or all must be nullptr");
    }

    externalMemoryProvided = (vk_X != nullptr && vk_Y != nullptr && vk_Color != nullptr);

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

        getPtrRef(colorVar) = vk_Color;
        d_plot_color = vk_Color;
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
    cudaMalloc(&state_.d_speed_term_1, size);
    cudaMalloc(&state_.d_speed_term_2, size);

    cudaMalloc(&state_.d_rngStates, params.num_agents * sizeof(curandState));

    cudaMalloc(&state_.d_cell_head, params.hash_table_size * sizeof(int));
    cudaMalloc(&state_.d_agent_next, params.num_agents * sizeof(int));

    cudaMalloc(&state_.d_boundaries_buffer, 6 * sizeof(float));
    cudaMalloc(&state_.d_pressure_buffer, 2 * sizeof(float));

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

Boundaries Simulation::getBoundaries() const {
    const float* ptr_x = d_plot_x ? d_plot_x : state_.d_execution_cost;
    const float* ptr_y = d_plot_y ? d_plot_y : state_.d_inventory;
    const float* ptr_c = d_plot_color ? d_plot_color : state_.d_speed;

    return launchComputeBoundaries(ptr_x, ptr_y, ptr_c, state_.d_boundaries_buffer,
                                   params_.num_agents);
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
    // Initialize cell heads to -1
    cudaMemset(state_.d_cell_head, -1, params_.hash_table_size * sizeof(int));

    // Build Linked List Spatial Hash
    launchBuildSpatialHash(state_.d_inventory, state_.d_execution_cost, state_.d_cell_head,
                           state_.d_agent_next, params_);

    // Compute local densities using linked list traversal
    launchComputeLocalDensities(state_.d_inventory, state_.d_execution_cost, state_.d_cell_head,
                                state_.d_agent_next, state_.d_local_density, params_);
}

void Simulation::computePressure() {
    launchComputeSpeedTerms(state_.d_risk_aversion, state_.d_local_density, state_.d_inventory,
                            state_.d_speed_term_1, state_.d_speed_term_2, state_.dt, params_);

    // Compute the pressure based on the speed terms
    launchComputePressure(state_.d_speed_term_1, state_.d_speed_term_2, state_.d_pressure_buffer,
                          &state_.pressure, params_.num_agents);
}

void Simulation::updateAgentState() {
    // Compute the trading speed for each agent based on their risk aversion, local density, and
    // pressure
    launchUpdateAgentState(state_.d_speed_term_1, state_.d_speed_term_2, state_.d_local_density,
                           state_.d_agent_next, state_.pressure, state_.d_speed, state_.d_inventory,
                           state_.d_execution_cost, state_.d_cash, state_.price, params_);
}

void Simulation::updatePrice() {
    state_.price +=
        params_.permanent_impact * state_.pressure * params_.time_delta +
        params_.price_randomness_stddev * this->normal_dist(rng) * sqrt(params_.time_delta);
}

void Simulation::step(bool waitForRender, bool signalRender) {
    // Skip wait on first step to avoid deadlock
    if (waitForRender && state_.dt > 0 && cudaWaitSemaphore != nullptr) {
        cudaExternalSemaphoreWaitParams waitParams{};  // Defaults are fine for binary semaphores
        cudaWaitExternalSemaphoresAsync(&cudaWaitSemaphore, &waitParams, 1,
                                        0);  // 0 = Default Stream
    }

    state_.dt++;
    computeLocalDensities();
    computePressure();
    updateAgentState();
    updatePrice();

    if (signalRender && cudaSignalSemaphore != nullptr) {
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
    auto safeFree = [&](float*& ptr) {
        if (ptr != nullptr && ptr != d_plot_x && ptr != d_plot_y && ptr != d_plot_color) {
            cudaFree(ptr);
        }
        ptr = nullptr;
    };

    safeFree(state_.d_inventory);
    safeFree(state_.d_execution_cost);
    safeFree(state_.d_cash);
    safeFree(state_.d_speed);
    safeFree(state_.d_local_density);
    safeFree(state_.d_risk_aversion);

    cudaFree(state_.d_rngStates);

    cudaFree(state_.d_speed_term_1);
    cudaFree(state_.d_speed_term_2);

    cudaFree(state_.d_cell_head);
    cudaFree(state_.d_agent_next);

    cudaFree(state_.d_boundaries_buffer);
    cudaFree(state_.d_pressure_buffer);
}

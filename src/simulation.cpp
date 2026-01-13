#include "simulation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
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
    : params_(params), colorVar_(colorVar), rng(std::random_device{}()), normal_dist(0.0f, 1.0f) {
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
            case PlotVar::Greed:
                return state_.d_greed;
            case PlotVar::Belief:
                return state_.d_belief;
            default:
                throw std::runtime_error("Unknown PlotVar");
        }
    };

    // Assign external memory if applicable
    if (externalMemoryProvided) {
        if (xVar == yVar || xVar == colorVar || yVar == colorVar) {
            throw std::invalid_argument(
                "Cannot map the same PlotVar to multiple axes when using external memory.");
        }

        getPtrRef(xVar) = vk_X;
        d_plot_x = vk_X;

        getPtrRef(yVar) = vk_Y;
        d_plot_y = vk_Y;

        getPtrRef(colorVar) = vk_Color;
        d_plot_color = vk_Color;
    }

    // Allocate the rest
    std::vector<PlotVar> allVars = {
        PlotVar::Inventory,    PlotVar::ExecutionCost, PlotVar::Cash,  PlotVar::Speed,
        PlotVar::RiskAversion, PlotVar::LocalDensity,  PlotVar::Greed, PlotVar::Belief};

    for (auto var : allVars) {
        float*& ptr = getPtrRef(var);
        if (ptr == nullptr) {
            cudaMalloc(&ptr, size);
        }
    }

    // Allocate sorted arrays and intermediate buffers
    cudaMalloc(&state_.d_speed_term_1, size);
    cudaMalloc(&state_.d_speed_term_2, size);
    cudaMalloc(&state_.d_target_inventory, size);

    cudaMalloc(&state_.d_rngStates, params.num_agents * sizeof(curandState));

    cudaMalloc(&state_.d_cell_head, params.hash_table_size * sizeof(int));
    cudaMalloc(&state_.d_agent_next, params.num_agents * sizeof(int));

    cudaMalloc(&state_.d_boundaries_buffer, 6 * sizeof(float));
    cudaMalloc(&state_.d_pressure_buffer, 2 * sizeof(float));

    // Allocate host history buffers
    int history_size = std::max(params.max_latency_steps, 2);
    state_.price_history = new float[history_size];
    std::fill_n(state_.price_history, history_size, params.price_init);

    if (params.max_latency_steps > 0) {
        state_.pressure_history = new float[params.max_latency_steps];
        std::fill_n(state_.pressure_history, params.max_latency_steps, 0.0f);
    } else {
        state_.pressure_history = nullptr;
    }

    // Initialize RNG states once with time-based seed
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    launchSetupRNG(state_.d_rngStates, params.num_agents, seed);

    // Initialize device memory using persistent RNG states
    launchInitializeExponential(state_.d_inventory, params.decay_rate, state_.d_rngStates,
                                params.num_agents);

    // Initialize buyers proportion (negative inventory)
    launchFlipSigns(state_.d_inventory, params.num_agents, params.buyer_proportion);

    // Initialize from log-normal to avoid negative values leading to NaNs later
    // Convert generic mean/stddev to underlying LogNormal parameters mu/sigma
    auto toLogNormalParams = [](float m, float s) -> std::pair<float, float> {
        if (m <= 0)
            return {0.0f, 1.0f};  // Fallback
        float v = s * s;
        float sigma2 = std::log(1.0f + v / (m * m));
        float mu = std::log(m) - 0.5f * sigma2;
        return {mu, std::sqrt(sigma2)};
    };

    auto [risk_mu, risk_sigma] = toLogNormalParams(params.risk_mean, params.risk_stddev);
    launchInitializeLogNormal(state_.d_risk_aversion, risk_mu, risk_sigma, state_.d_rngStates,
                              params.num_agents);

    auto [greed_mu, greed_sigma] = toLogNormalParams(params.greed_mean, params.greed_stddev);

    launchInitializeLogNormal(state_.d_greed, greed_mu, greed_sigma, state_.d_rngStates,
                              params.num_agents);

    // Initialize Belief with some variance so visualization isn't uniform
    // Belief usually ranges [-1, 1] or follows price trends
    launchInitializeNormal(state_.d_belief, 0.0f, 0.1f, state_.d_rngStates, params.num_agents);

    // Initialize target inventory
    launchInitializeNormal(state_.d_target_inventory, params.target_inventory_mean,
                           params.target_inventory_stddev, state_.d_rngStates, params.num_agents);

    // Initialize others to 0
    if (state_.d_cash != vk_X && state_.d_cash != vk_Y)
        cudaMemset(state_.d_cash, 0, size);
    if (state_.d_speed != vk_X && state_.d_speed != vk_Y)
        cudaMemset(state_.d_speed, 0, size);
    if (state_.d_execution_cost != vk_X && state_.d_execution_cost != vk_Y)
        cudaMemset(state_.d_execution_cost, 0, size);

    // But if they are inputs initialized by kernels above (like inventory), we are good.
    // Cash starts at 0.
    if (state_.d_cash == vk_X || state_.d_cash == vk_Y)
        cudaMemset(state_.d_cash, 0, size);
    if (state_.d_execution_cost == vk_X || state_.d_execution_cost == vk_Y)
        cudaMemset(state_.d_execution_cost, 0, size);

    // Ensure all initialization kernels and memsets are complete before rendering starts
    cudaDeviceSynchronize();
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
                            state_.d_target_inventory, state_.d_speed_term_1, state_.d_speed_term_2,
                            state_.dt, params_);

    // Compute the pressure based on the speed terms
    launchComputePressure(state_.d_speed_term_1, state_.d_speed_term_2, state_.d_pressure_buffer,
                          &state_.pressure, params_.num_agents);
}

void Simulation::updateAgentState(float observed_pressure, float price_change) {
    // Compute the trading speed for each agent based on their risk aversion, local density, and
    // pressure
    launchUpdateAgentState(state_.d_speed_term_1, state_.d_speed_term_2, state_.d_local_density,
                           state_.d_agent_next, observed_pressure, state_.d_greed, state_.d_belief,
                           price_change, state_.d_speed, state_.d_inventory,
                           state_.d_target_inventory, state_.d_execution_cost, state_.d_cash,
                           state_.price, params_);
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

    float observed_pressure = state_.pressure;

    // Always update price history (size >= 2)
    int history_size = std::max(params_.max_latency_steps, 2);
    // Be careful with signed modulo if dt is somehow negative (it shouldn't be)
    state_.price_history[state_.dt % history_size] = state_.price;

    // Latency and Jitter Logic
    if (params_.max_latency_steps > 0) {
        // Store current pressure in history
        int current_idx = state_.dt % params_.max_latency_steps;
        state_.pressure_history[current_idx] = state_.pressure;

        // Calculate latency
        float latency = params_.latency_mean;
        if (params_.latency_jitter_stddev > 0) {
            latency += normal_dist(rng) * params_.latency_jitter_stddev;
        }
        if (latency < 0)
            latency = 0;

        // Lookup historical pressure
        int steps_back = static_cast<int>(latency / params_.time_delta);
        if (steps_back >= params_.max_latency_steps)
            steps_back = params_.max_latency_steps - 1;

        // Ensure we don't look back before start of simulation effectively (although ring buffer
        // handles it with init values) If dt < steps_back, we get values from initialization.

        int lookup_idx = (state_.dt - steps_back) % params_.max_latency_steps;
        if (lookup_idx < 0)
            lookup_idx += params_.max_latency_steps;

        observed_pressure = state_.pressure_history[lookup_idx];
    }

    float current_price_start = state_.price;

    // Calculate price change using history
    int prev_idx = (state_.dt - 1) % history_size;
    // Handle wrap-around for safety modulo arithmetic on potentially small negative if logic
    // changes
    if (prev_idx < 0)
        prev_idx += history_size;

    float previous_price = state_.price_history[prev_idx];
    float price_change = current_price_start - previous_price;

    updateAgentState(observed_pressure, price_change);
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
    safeFree(state_.d_target_inventory);
    safeFree(state_.d_cash);
    safeFree(state_.d_speed);
    safeFree(state_.d_local_density);
    safeFree(state_.d_greed);
    safeFree(state_.d_belief);
    safeFree(state_.d_risk_aversion);

    cudaFree(state_.d_rngStates);

    cudaFree(state_.d_speed_term_1);
    cudaFree(state_.d_speed_term_2);

    cudaFree(state_.d_cell_head);
    cudaFree(state_.d_agent_next);

    cudaFree(state_.d_boundaries_buffer);
    cudaFree(state_.d_pressure_buffer);

    // Free host history buffers
    delete[] state_.price_history;
    delete[] state_.pressure_history;
}

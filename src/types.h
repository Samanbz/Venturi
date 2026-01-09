#pragma once

#include <curand_kernel.h>

struct MarketState {
    int dt;          // t, current time step
    float price;     // S_t, current asset price
    float pressure;  // mu_t, current market pressure

    float* d_inventory = nullptr;         // Q_t^a, agent inventories
    float* d_cash = nullptr;              // X_t^a, agent cash balances
    float* d_execution_cost = nullptr;    // \delta_t^a, agent execution costs
    float* d_risk_aversion = nullptr;     // phi^a, agent risk aversion parameters
    float* d_local_density = nullptr;     // rho_t^a, agent local densities
    float* d_target_inventory = nullptr;  // Target inventory for each agent

    float* d_speed_term_1 = nullptr;
    float* d_speed_term_2 = nullptr;
    float* d_speed = nullptr;  // nu_t^a, agent trading speeds

    int* d_cell_head =
        nullptr;  // Spatial grid cell start indices (Used as d_cell_head for Linked List)
    int* d_agent_next =
        nullptr;  // Agent indices sorted by cell (Used as d_agent_next for Linked List)

    curandState* d_rngStates = nullptr;  // RNG states for each agent (persistent)

    // Scratch buffers for reductions
    float* d_boundaries_buffer = nullptr;  // [min_y, max_y, min_x, max_x, sum_c, sq_sum_c]
    float* d_pressure_buffer = nullptr;    // [sum_s1, sum_s2]
};

struct MarketParams {
    int num_agents;                 // N, number of trading agents
    int num_steps;                  // T, number of simulation steps
    float time_delta;               // simulation time step size
    float price_init;               // S_0, initial asset price
    float permanent_impact;         // alpha, permanent market impact factor
    float temporary_impact;         // kappa, temporary market impact factor
    float congestion_sensitivity;   // beta, congestion sensitivity factor
    float price_randomness_stddev;  // standard deviation of price brownian motion

    float sph_smoothing_radius;  // radius for SPH smoothing
    int hash_table_size;         // size of spatial hash table

    float mass_alpha;  // base mass for local density calculation
    float mass_beta;   // Inventory scaling factor

    float decay_rate;   // decay rate for inventory initialization
    float risk_mean;    // mean for risk aversion initialization
    float risk_stddev;  // standard deviation for risk aversion initialization

    float target_inventory_mean;    // mean for target inventory initialization
    float target_inventory_stddev;  // standard deviation for target inventory initialization
};

struct Boundaries {
    float minY, maxY;
    float minX, maxX;
    float minColor, maxColor;
};

#pragma once

#include <curand_kernel.h>

struct MarketState {
    int dt;          // t, current time step
    float price;     // S_t, current asset price
    float pressure;  // mu_t, current market pressure

    float* d_inventory;       // Q_t^a, agent inventories
    float* d_cash;            // X_t^a, agent cash balances
    float* d_execution_cost;  // \delta_t^a, agent execution costs
    float* d_risk_aversion;   // phi^a, agent risk aversion parameters

    float* d_speed_term_1;
    float* d_speed_term_2;
    float* d_speed;  // nu_t^a, agent trading speeds

    float* d_risk_aversion_sorted;
    float* d_inventory_sorted;       // Q_t^a, agent inventories
    float* d_cash_sorted;            // X_t^a, agent cash balances
    float* d_execution_cost_sorted;  // \delta_t^a, agent execution costs
    float* d_speed_sorted;           // nu_t^a, agent trading speeds
    float* d_local_density_sorted;  // rho_t^a, agent local densities, always sorted by spatial hash

    int* d_cell_start;   // Spatial grid cell start indices
    int* d_cell_end;     // Spatial grid cell end indices
    int* d_agent_hash;   // Agent's cell indices
    int* d_agent_index;  // Agent indices sorted by cell

    curandState* d_rngStates;  // RNG states for each agent (persistent)
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
};

using BoundaryPair = std::pair<std::pair<float, float>, std::pair<float, float>>;

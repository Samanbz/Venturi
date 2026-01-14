#pragma once

#include <curand_kernel.h>

/**
 * @brief Holds the current simulation state of the market and agents.
 *
 * This structure contains host and device pointers for all agent variables,
 * intermediate buffers for calculation, and historical data for plotting.
 * Most pointers here point to GPU memory (device) unless specified otherwise.
 */
struct MarketState {
    int dt;          ///< Current time step index (t)
    float price;     ///< Current asset price ($S_t$)
    float pressure;  ///< Current market pressure ($\mu_t$)

    float* d_inventory = nullptr;         ///< Device: Agent inventories ($Q_t^a$)
    float* d_cash = nullptr;              ///< Device: Agent cash balances ($X_t^a$)
    float* d_execution_cost = nullptr;    ///< Device: Agent execution costs ($\delta_t^a$)
    float* d_risk_aversion = nullptr;     ///< Device: Agent risk aversion parameters ($\phi^a$)
    float* d_local_density = nullptr;     ///< Device: Local agent density ($\rho_t^a$)
    float* d_target_inventory = nullptr;  ///< Device: Target inventory levels
    float* d_greed = nullptr;             ///< Device: Greed factor (price sensitivity)
    float* d_belief = nullptr;            ///< Device: Belief/Momentum factor

    float* d_speed_term_1 = nullptr;  ///< Device: Scratch buffer for speed calculation term 1
    float* d_speed_term_2 = nullptr;  ///< Device: Scratch buffer for speed calculation term 2
    float* d_speed = nullptr;         ///< Device: Agent trading speeds ($\nu_t^a$)

    int* d_cell_head = nullptr;   ///< Device: Spatial Hash Grid - Cell Head Indices
    int* d_agent_next = nullptr;  ///< Device: Spatial Hash Grid - Linked List Next Indices

    curandState* d_rngStates = nullptr;  ///< Device: Persistent RNG states per agent

    // Scratch buffers for reductions
    float* d_boundaries_buffer = nullptr;  ///< Device: Min/Max/Sum reduction buffer (6 floats)
    float* d_pressure_buffer = nullptr;    ///< Device: Pressure reduction buffer (2 floats)

    // Market State History (Host pointers)
    float* price_history = nullptr;     ///< Host: Ring buffer for past prices
    float* pressure_history = nullptr;  ///< Host: Ring buffer for past pressures
};

/**
 * @brief Configuration parameters for the market simulation.
 *
 * Defines the initial conditions, physical constants, and hyperparameters
 * for the Mean Field Game and Agent-Based Model.
 */
struct MarketParams {
    int num_agents;    ///< Number of trading agents ($N$)
    int num_steps;     ///< Total simulation steps ($T$)
    float time_delta;  ///< Time step size ($dt$)
    float price_init;  ///< Initial asset price ($S_0$)

    // Latency
    int max_latency_steps;        ///< Size of latency buffer (steps)
    float latency_mean;           ///< Mean latency (seconds)
    float latency_jitter_stddev;  ///< Latency variation (stddev)

    float permanent_impact;         ///< Permanent market impact factor ($\alpha$)
    float temporary_impact;         ///< Temporary market impact factor ($\kappa$)
    float congestion_sensitivity;   ///< Congestion sensitivity factor ($\beta$)
    float price_randomness_stddev;  ///< Price volatility ($\sigma$)

    float sph_smoothing_radius;  ///< Radius for SPH density kernel ($h$)
    int hash_table_size;         ///< Size of the spatial hash table

    float mass_alpha;  ///< Base mass for density calc
    float mass_beta;   ///< Mass scaling by inventory

    float decay_rate;   ///< Exponential decay rate for initial distribution
    float risk_mean;    ///< Mean Risk Aversion
    float risk_stddev;  ///< StdDev Risk Aversion

    float greed_mean;    ///< Mean Greed
    float greed_stddev;  ///< StdDev Greed
    float trend_decay;   ///< Decay factor for belief momentum

    float target_inventory_mean;    ///< Mean Target Inventory
    float target_inventory_stddev;  ///< StdDev Target Inventory

    float buyer_proportion;  ///< Proportion of agents initialized as buyers (0.0 to 1.0)

    float inertia;  ///< Speed smoothing factor (0.0 = no inertia, 1.0 = full freeze)
};

/**
 * @brief Min/Max boundaries for simulation variables.
 * Used for normalizing visualization colors and camera view.
 */
struct Boundaries {
    float minY, maxY;          ///< Y-Axis range (e.g., Inventory)
    float minX, maxX;          ///< X-Axis range (e.g., Cost)
    float minColor, maxColor;  ///< Color-Axis range (e.g., Speed)
};

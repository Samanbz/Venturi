#pragma once

#include "types.h"

// Forward declarations for kernel launchers

/**
 * @brief Initializes random number generator states for all agents.
 *
 * @param d_rngStates Device pointer to array of RNG states.
 * @param num_agents Number of agents (threads) to initialize.
 * @param seed Seed for the random number generator.
 */
extern void launchSetupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed);

/**
 * @brief Initializes a buffer with an exponential distribution.
 *
 * @param d_data Device pointer to the data buffer.
 * @param lambda Rate parameter for exponential distribution.
 * @param d_rngStates Device pointer to initialized RNG states.
 * @param num_agents Number of agents.
 */
extern void launchInitializeExponential(float* d_data,
                                        float lambda,
                                        curandState* d_rngStates,
                                        int num_agents);

/**
 * @brief Initializes a buffer with a uniform distribution [min, max].
 *
 * @param d_data Device pointer to the data buffer.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param d_rngStates Device pointer to RNG states.
 * @param num_agents Number of agents.
 */
extern void launchInitializeUniform(
    float* d_data, float min, float max, curandState* d_rngStates, int num_agents);

/**
 * @brief Initializes a buffer with a normal distribution.
 *
 * @param d_data Device pointer to the data buffer.
 * @param mean Mean of the normal distribution.
 * @param stddev Standard deviation.
 * @param d_rngStates Device pointer to RNG states.
 * @param num_agents Number of agents.
 */
extern void launchInitializeNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents);

/**
 * @brief Initializes a buffer with a log-normal distribution.
 *
 * @param d_data Device pointer to the data buffer.
 * @param mean Mean of the underlying normal distribution.
 * @param stddev Standard deviation of the underlying normal distribution.
 * @param d_rngStates Device pointer to RNG states.
 * @param num_agents Number of agents.
 */
extern void launchInitializeLogNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents);

/**
 * @brief Randomly flips the sign of values in the buffer (50% chance).
 * Used to create a mix of long/short positions.
 *
 * @param d_data Device pointer to the data buffer.
 * @param num_agents Number of agents.
 */
extern void launchFlipSigns(float* d_data, int num_agents);

/**
 * @brief Builds the Spatial Hash Grid on the GPU.
 *
 * Clears the hash table and re-inserts all agents based on their
 * current (Cost, Inventory) coordinates.
 *
 * @param d_inventory Device pointer to agent inventories (Y coord).
 * @param d_execution_cost Device pointer to agent costs (X coord).
 * @param d_cell_head Device pointer to hash table head indices.
 * @param d_agent_next Device pointer to linked list next indices.
 * @param params System parameters (cell size, hash table size).
 */
extern void launchBuildSpatialHash(const float* d_inventory,
                                   const float* d_execution_cost,
                                   int* d_cell_head,
                                   int* d_agent_next,
                                   MarketParams params);

/**
 * @brief Computes local density for each agent using SPH smoothing.
 *
 * Iterates through neighbors in the spatial hash grid to sum weighted mass.
 *
 * @param d_inventory Device pointer to agent inventories.
 * @param d_execution_cost Device pointer to agent costs.
 * @param d_cell_head Device pointer to spatial hash heads.
 * @param d_agent_next Device pointer to linked list nexts.
 * @param d_local_density Output device pointer for calculated densities.
 * @param params System parameters (radius, mass constants).
 */
extern void launchComputeLocalDensities(const float* d_inventory,
                                        const float* d_execution_cost,
                                        const int* d_cell_head,
                                        const int* d_agent_next,
                                        float* d_local_density,
                                        MarketParams params);

/**
 * @brief Computes intermediate speed terms based on agent state and density.
 *
 * @param d_risk_aversion Agent risk aversion.
 * @param d_local_density Agent local density.
 * @param d_inventory Agent inventory.
 * @param d_target_inventory Agent target inventory.
 * @param d_speed_term_1 Output scratch buffer 1.
 * @param d_speed_term_2 Output scratch buffer 2.
 * @param dt Current time step.
 * @param params Market parameters.
 */
extern void launchComputeSpeedTerms(const float* d_risk_aversion,
                                    const float* d_local_density,
                                    const float* d_inventory,
                                    const float* d_target_inventory,
                                    float* d_speed_term_1,
                                    float* d_speed_term_2,
                                    int dt,
                                    MarketParams params);

/**
 * @brief Updates full agent state (Inventory, Cash, Speed, Belief) via SDE integration.
 * Performs the Euler-Maruyama step.
 *
 * @param d_speed_term_1 Computed speed term 1.
 * @param d_speed_term_2 Computed speed term 2.
 * @param d_local_density Local density.
 * @param d_agent_indices (Optional) Permutation indices if sorted.
 * @param pressure Global market pressure.
 * @param d_greed Greed factor.
 * @param d_belief Belief/Momentum state (updated in-place).
 * @param price_change Change in price from last step.
 * @param d_speed Output: New trading speed.
 * @param d_inventory In/Out: Inventory level.
 * @param d_target_inventory Target inventory.
 * @param d_execution_cost In/Out: Execution cost.
 * @param d_cash In/Out: Cash balance.
 * @param price Current asset price.
 * @param params Market parameters.
 */
extern void launchUpdateAgentState(const float* d_speed_term_1,
                                   const float* d_speed_term_2,
                                   const float* d_local_density,
                                   const int* d_agent_indices,
                                   float pressure,
                                   const float* d_greed,
                                   float* d_belief,
                                   float price_change,
                                   float* d_speed,
                                   float* d_inventory,
                                   const float* d_target_inventory,
                                   float* d_execution_cost,
                                   float* d_cash,
                                   float price,
                                   MarketParams params);

/**
 * @brief Aggregates agent speed terms to compute total market pressure.
 *
 * @param d_speed_term_1 Speed term 1 buffer.
 * @param d_speed_term_2 Speed term 2 buffer.
 * @param d_pressure_buffer Intermediate reduction buffer.
 * @param pressure Output pointer for scalar result.
 * @param num_agents Number of agents.
 */
extern void launchComputePressure(const float* d_speed_term_1,
                                  const float* d_speed_term_2,
                                  float* d_pressure_buffer,
                                  float* pressure,
                                  int num_agents);

/**
 * @brief Computes the bounding box (Min/Max) for visualization variables.
 *
 * @param d_x Data mapped to X-axis.
 * @param d_y Data mapped to Y-axis.
 * @param d_c Data mapped to Color-axis.
 * @param d_buffer Scratch buffer for reduction (size 6).
 * @param num_agents Number of agents.
 * @return Boundaries struct with min/max values.
 */
extern Boundaries launchComputeBoundaries(
    const float* d_x, const float* d_y, const float* d_c, float* d_buffer, int num_agents);

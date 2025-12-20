#pragma once

#include <curand_kernel.h>

#include <random>

#include "types.h"

// Forward declarations for CUDA helper functions (defined in simulation.cu)
extern void copyParamsToDevice(const MarketParams& params);
extern void setupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed);
extern void launchInitializeExponential(float* d_data,
                                        float lambda,
                                        curandState* d_rngStates,
                                        int num_agents);
extern void launchInitializeUniform(
    float* d_data, float min, float max, curandState* d_rngStates, int num_agents);
extern void launchInitializeNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents);

extern void launchCalculateSpatialHash(const float* d_inventory,
                                       const float* d_execution_cost,
                                       int* d_agent_hash,
                                       int* d_agent_indices,
                                       int num_agents);
extern void launchFindCellBounds(const int* d_sorted_hashes,
                                 int* d_cell_start,
                                 int* d_cell_end,
                                 int num_agents);

extern void launchSortByKey(int* d_keys, int* d_values, int num_agents);
extern void launchReorderData(const int* sorted_indices,
                              const float* in_inventory,
                              const float* in_execution_cost,
                              const float* in_cash,
                              const float* in_speed,
                              float* out_inventory,
                              float* out_execution_cost,
                              float* out_cash,
                              float* out_speed,
                              int num_agents);
extern void launchComputeLocalDensities(const float* d_inventory,
                                        const float* d_execution_cost,
                                        const int* d_cell_start_idx,
                                        const int* d_cell_end_idxs,
                                        float* d_local_density,
                                        int num_agents);

extern void launchComputePressure(const float* d_speed_term_1,
                                  const float* d_speed_term_2,
                                  float* pressure,
                                  int num_agents);

extern void launchUpdateSpeedInventoryExecutionCost(const float* d_speed_term_1,
                                                    const float* d_speed_term_2,
                                                    const float* d_local_density,
                                                    const int* d_agent_indices,
                                                    float pressure,
                                                    float* d_speed,
                                                    float* d_inventory_sorted,
                                                    float* d_inventory_original,
                                                    float* d_execution_cost_sorted,
                                                    float* d_execution_cost_original,
                                                    int num_agents);

extern void launchComputeSpeedTerms(const float* d_risk_aversion,
                                    const float* d_local_density,
                                    const float* d_inventory,
                                    float* d_speed_term_1,
                                    float* d_speed_term_2,
                                    int dt,
                                    int num_agents);

class Simulation {
   public:
    Simulation(const MarketParams& params, float* vk_X = nullptr, float* vk_Y = nullptr);
    ~Simulation();

    void step();
    void run();

    BoundaryPair getBoundaries() const;

    friend class UpdateLogicFixture;

    //    private:
    void computeLocalDensities();
    void computePressure();
    void updateSpeedInventoryExecutionCost();
    void updatePrice();

    bool externalMemoryProvided;

    // Simulation parameters and state
    MarketParams params_;
    MarketState state_;

    // Random number generator for price updates
    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;
};
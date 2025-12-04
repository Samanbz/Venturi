#pragma once

#include <curand_kernel.h>

#include "types.h"

// Forward declarations for CUDA helper functions (defined in simulation.cu)
extern void copyParamsToDevice(const MarketParams& params);
extern void setupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed);
extern void launchInitializeInventories(float* d_inventory, curandState* d_rngStates,
                                        int num_agents);
extern void launchInitializeRiskAversions(float* d_risk_aversion, curandState* d_rngStates,
                                          int num_agents);

extern void launchCalculateSpatialHash(const float* d_inventory, const float* d_execution_cost,
                                       int* d_agent_hash, int* d_agent_indices, int num_agents);
extern void launchFindCellBounds(const int* d_sorted_hashes, int* d_cell_start, int* d_cell_end,
                                 int num_agents);
extern void launchComputeLocalDensities(const float* d_inventory, const float* d_execution_cost,
                                        const int* d_cell_start, const int* d_cell_end,
                                        float* d_local_density);

class Simulation {
   public:
    Simulation(const MarketParams& params);
    ~Simulation();

    void step();

   private:
    void computeLocalDensities();
    MarketParams params_;
    MarketState state_;
};
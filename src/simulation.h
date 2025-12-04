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

class Simulation {
   public:
    Simulation(const MarketParams& params);
    ~Simulation();

    void step();

   private:
    MarketParams params_;
    MarketState state_;
};
#pragma once

#include "types.h"

// Forward declarations for kernel launchers
extern void launchSetupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed);
extern void launchInitializeExponential(float* d_data,
                                        float lambda,
                                        curandState* d_rngStates,
                                        int num_agents);
extern void launchInitializeUniform(
    float* d_data, float min, float max, curandState* d_rngStates, int num_agents);
extern void launchInitializeNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents);
extern void launchInitializeLogNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents);

extern void launchBuildSpatialHash(const float* d_inventory,
                                   const float* d_execution_cost,
                                   int* d_cell_head,
                                   int* d_agent_next,
                                   MarketParams params);

extern void launchComputeLocalDensities(const float* d_inventory,
                                        const float* d_execution_cost,
                                        const int* d_cell_head,
                                        const int* d_agent_next,
                                        float* d_local_density,
                                        MarketParams params);

extern void launchComputeSpeedTerms(const float* d_risk_aversion,
                                    const float* d_local_density,
                                    const float* d_inventory,
                                    float* d_speed_term_1,
                                    float* d_speed_term_2,
                                    int dt,
                                    MarketParams params);

extern void launchUpdateAgentState(const float* d_speed_term_1,
                                   const float* d_speed_term_2,
                                   const float* d_local_density,
                                   const int* d_agent_indices,
                                   float pressure,
                                   float* d_speed,
                                   float* d_inventory,
                                   float* d_execution_cost,
                                   float* d_cash,
                                   float price,
                                   MarketParams params);

extern void launchComputePressure(const float* d_speed_term_1,
                                  const float* d_speed_term_2,
                                  float* d_pressure_buffer,
                                  float* pressure,
                                  int num_agents);

extern Boundaries launchComputeBoundaries(
    const float* d_x, const float* d_y, const float* d_c, float* d_buffer, int num_agents);

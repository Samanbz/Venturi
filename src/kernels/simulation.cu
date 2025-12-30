#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <ctime>

#include "common.cuh"
#include "types.h"

extern __constant__ MarketParams d_params;

__global__ void calculateSpatialHashKernel(const float* inventory,
                                           const float* execution_cost,
                                           int* agent_hash,
                                           int* agent_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    // Discretize to Integer Grid Coordinates (can be very large but not negative)
    int grid_x = (int) (execution_cost[idx] / d_params.sph_smoothing_radius);
    int grid_y = (int) (inventory[idx] / d_params.sph_smoothing_radius);

    // Compute unbounded spatial hash
    unsigned int h = ((grid_x * 73856093) ^ (grid_y * 19349663));

    // Wrap to table size
    agent_hash[idx] = h % d_params.hash_table_size;
    agent_indices[idx] = idx;
}

__global__ void findCellBoundsKernel(const int* sorted_hashes, int* cell_start, int* cell_end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    int hash = sorted_hashes[idx];

    // Handle Start of Cell
    if (idx == 0 || hash != sorted_hashes[idx - 1]) {
        cell_start[hash] = idx;
    }

    // Handle End of Cell
    if (idx == d_params.num_agents - 1 || hash != sorted_hashes[idx + 1]) {
        cell_end[hash] = idx + 1;  // Exclusive end
    }
}

__device__ inline float computeInteraction(
    float my_inv, float my_cost, float n_inv, float n_cost, float h2, float poly6) {
    float d_inv = my_inv - n_inv;
    float d_cost = my_cost - n_cost;
    float r2 = d_inv * d_inv + d_cost * d_cost;
    return (r2 < h2) ? (poly6 * powf(h2 - r2, 3)) : 0.0f;
}

__global__ void computeLocalDensitiesKernel(const float* __restrict__ inventory,
                                            const float* __restrict__ execution_cost,
                                            const int* __restrict__ cell_start_idx,
                                            const int* __restrict__ cell_end_idxs,
                                            const int* __restrict__ agent_indices,
                                            float* __restrict__ d_local_density) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    float my_inv = inventory[idx];
    float my_cost = execution_cost[idx];

    // Constants
    float h2 = d_params.sph_smoothing_radius * d_params.sph_smoothing_radius;
    float poly6 = 315.0f / (64.0f * 3.14159265f * powf(d_params.sph_smoothing_radius, 9));
    float density_acc = 0.0f;

    int my_grid_x = floor(my_cost / d_params.sph_smoothing_radius);
    int my_grid_y = floor(my_inv / d_params.sph_smoothing_radius);

    // Flattened Neighbor Loop
    for (int n = 0; n < 9; n++) {
        int neighbor_grid_x = my_grid_x + ((n % 3) - 1);
        int neighbor_grid_y = my_grid_y + ((n / 3) - 1);

        unsigned int h = ((neighbor_grid_x * 73856093) ^ (neighbor_grid_y * 19349663));
        unsigned int hash = h % d_params.hash_table_size;
        int start = cell_start_idx[hash];
        int end = cell_end_idxs[hash];

        for (int j = start; j < end; j++) {
            float n_inv = inventory[j];
            float n_cost = execution_cost[j];
            float neighbor_mass = d_params.mass_alpha + d_params.mass_beta * n_inv;
            density_acc +=
                neighbor_mass * computeInteraction(my_inv, my_cost, n_inv, n_cost, h2, poly6);
        }
    }

    // Scatter directly to original index
    int original_idx = agent_indices[idx];
    d_local_density[original_idx] = density_acc;
}

void launchCalculateSpatialHash(const float* d_inventory,
                                const float* d_execution_cost,
                                int* d_agent_hash,
                                int* d_agent_indices,
                                int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    calculateSpatialHashKernel<<<numBlocks, blockSize>>>(d_inventory, d_execution_cost,
                                                         d_agent_hash, d_agent_indices);
    // No sync here - letting caller control synchronization for better performance
}

void launchFindCellBounds(const int* d_sorted_hashes,
                          int* d_cell_start,
                          int* d_cell_end,
                          int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    findCellBoundsKernel<<<numBlocks, blockSize>>>(d_sorted_hashes, d_cell_start, d_cell_end);
    // No sync here - letting caller control synchronization for better performance
}

void launchSortByKey(int* d_keys, int* d_values, int num_agents) {
    thrust::device_ptr<int> t_keys(d_keys);
    thrust::device_ptr<int> t_values(d_values);
    thrust::sort_by_key(t_keys, t_keys + num_agents, t_values);
}

__global__ void reorderDataKernel(const int* __restrict__ sorted_indices,
                                  const float* __restrict__ in1,
                                  float* __restrict__ out1,
                                  const float* __restrict__ in2,
                                  float* __restrict__ out2,
                                  int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    int sorted_idx = sorted_indices[idx];
    out1[idx] = in1[sorted_idx];
    out2[idx] = in2[sorted_idx];
}

void launchReorderData(const int* sorted_indices,
                       int num_agents,
                       const float* in1,
                       float* out1,
                       const float* in2,
                       float* out2) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    reorderDataKernel<<<numBlocks, blockSize>>>(sorted_indices, in1, out1, in2, out2, num_agents);
}

void launchComputeLocalDensities(const float* d_inventory,
                                 const float* d_execution_cost,
                                 const int* d_cell_start,
                                 const int* d_cell_end,
                                 const int* d_agent_indices,
                                 float* d_local_density,
                                 int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    computeLocalDensitiesKernel<<<numBlocks, blockSize>>>(
        d_inventory, d_execution_cost, d_cell_start, d_cell_end, d_agent_indices, d_local_density);
    // No sync here - letting caller control synchronization for better performance
}

__global__ void computeSpeedTermsKernel(const float* __restrict__ risk_aversion,
                                        const float* __restrict__ local_density,
                                        const float* __restrict__ inventory,
                                        float* __restrict__ speed_term_1,
                                        float* __restrict__ speed_term_2,
                                        int dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    float personal_decay_rate = sqrtf(risk_aversion[idx] / d_params.temporary_impact);
    float local_temporary_impact =
        d_params.temporary_impact * (1 + d_params.congestion_sensitivity * local_density[idx]);
    speed_term_1[idx] = d_params.permanent_impact *
                        (1 - expf(-personal_decay_rate * (d_params.num_steps - dt))) /
                        (2 * local_temporary_impact * personal_decay_rate);
    speed_term_2[idx] =
        (2 * sqrtf(d_params.temporary_impact * risk_aversion[idx]) * inventory[idx]) /
        (2 * local_temporary_impact);
}

__global__ void updateAgentStateKernel(const float* __restrict__ speed_term_1,
                                       const float* __restrict__ speed_term_2,
                                       const float* __restrict__ local_density,
                                       const int* __restrict__ agent_indices,
                                       const float pressure,
                                       float* __restrict__ speed,
                                       float* __restrict__ inventory,
                                       float* __restrict__ execution_cost,
                                       float* __restrict__ cash,
                                       float price) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    float local_temporary_impact =
        d_params.temporary_impact * (1 + d_params.congestion_sensitivity * local_density[idx]);
    speed[idx] = pressure * speed_term_1[idx] - speed_term_2[idx];

    float new_inv = inventory[idx] + speed[idx] * d_params.time_delta;
    // Clamp new inventory to 0 if it goes negative
    new_inv = fmaxf(new_inv, 0.0f);
    inventory[idx] = new_inv;
    float new_execution_cost = -local_temporary_impact * speed[idx];
    execution_cost[idx] = new_execution_cost;

    cash[idx] +=
        -speed[idx] * (price + d_params.temporary_impact * speed[idx]) * d_params.time_delta;
}

void launchComputeSpeedTerms(const float* d_risk_aversion,
                             const float* d_local_density,
                             const float* d_inventory,
                             float* d_speed_term_1,
                             float* d_speed_term_2,
                             int dt,
                             int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    computeSpeedTermsKernel<<<numBlocks, blockSize>>>(d_risk_aversion, d_local_density, d_inventory,
                                                      d_speed_term_1, d_speed_term_2, dt);
}

void launchUpdateAgentState(const float* d_speed_term_1,
                            const float* d_speed_term_2,
                            const float* d_local_density,
                            const int* d_agent_indices,
                            float pressure,
                            float* d_speed,
                            float* d_inventory,
                            float* d_execution_cost,
                            float* d_cash,
                            float price,
                            int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    updateAgentStateKernel<<<numBlocks, blockSize>>>(
        d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices, pressure, d_speed,
        d_inventory, d_execution_cost, d_cash, price);
}

void launchComputePressure(const float* d_speed_term_1,
                           const float* d_speed_term_2,
                           float* pressure,  // CPU Pointer
                           int num_agents) {
    // Consume any previous error to prevent thrust::reduce from crashing
    cudaGetLastError();

    thrust::device_ptr<const float> t_1(d_speed_term_1);
    thrust::device_ptr<const float> t_2(d_speed_term_2);

    float s1 = thrust::reduce(thrust::device, t_1, t_1 + num_agents, 0.0f);
    float s2 = thrust::reduce(thrust::device, t_2, t_2 + num_agents, 0.0f);

    *pressure = -s2 / (1.0f - s1);
}

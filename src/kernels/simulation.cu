#include <curand_kernel.h>

#include <ctime>

#include "common.cuh"
#include "types.h"

// Helper for atomic min/max on float
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old =
            atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old =
            atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Linked List Spatial Hashing Kernels
__global__ void buildSpatialHashKernel(const float* inventory,
                                       const float* execution_cost,
                                       int* cell_head,
                                       int* agent_next,
                                       MarketParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_agents)
        return;

    // Discretize to Integer Grid Coordinates
    long long grid_x = floor(execution_cost[idx] / params.sph_smoothing_radius);
    long long grid_y = floor(inventory[idx] / params.sph_smoothing_radius);

    // Compute unbounded spatial hash
    unsigned int h = ((unsigned int) grid_x * 73856093) ^ ((unsigned int) grid_y * 19349663);
    // Use bitwise AND for modulo since hash_table_size is power of 2
    int hash = h & (params.hash_table_size - 1);

    int old_head = atomicExch(&cell_head[hash], idx);
    agent_next[idx] = old_head;
}

__device__ inline float computeInteraction(
    float my_inv, float my_cost, float n_inv, float n_cost, float h2, float poly6) {
    float d_inv = my_inv - n_inv;
    float d_cost = my_cost - n_cost;
    float r2 = d_inv * d_inv + d_cost * d_cost;

    if (r2 < h2) {
        float term = h2 - r2;
        return poly6 * term * term * term;
    }
    return 0.0f;
}

__global__ void computeLocalDensitiesKernel(const float* __restrict__ inventory,
                                            const float* __restrict__ execution_cost,
                                            const int* __restrict__ cell_head,
                                            const int* __restrict__ agent_next,
                                            float* __restrict__ d_local_density,
                                            MarketParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_agents)
        return;

    float my_inv = __ldg(&inventory[idx]);
    float my_cost = __ldg(&execution_cost[idx]);

    // Constants
    float h2 = params.sph_smoothing_radius * params.sph_smoothing_radius;
    // Precompute powf(radius, 9)
    float h9 = h2 * h2 * h2 * params.sph_smoothing_radius * params.sph_smoothing_radius *
               params.sph_smoothing_radius;
    float poly6 = 315.0f / (64.0f * 3.14159265f * h9);

    float density_acc = 0.0f;

    long long my_grid_x = floor(my_cost / params.sph_smoothing_radius);
    long long my_grid_y = floor(my_inv / params.sph_smoothing_radius);

    int hash_mask = params.hash_table_size - 1;

    // Nested Neighbor Loop to avoid division/modulo
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            long long neighbor_grid_x = my_grid_x + dx;
            long long neighbor_grid_y = my_grid_y + dy;

            unsigned int h = ((unsigned int) neighbor_grid_x * 73856093) ^
                             ((unsigned int) neighbor_grid_y * 19349663);
            int hash = h & hash_mask;

            // Traverse Linked List
            int curr = cell_head[hash];
            while (curr != -1) {
                float n_inv = __ldg(&inventory[curr]);
                float n_cost = __ldg(&execution_cost[curr]);
                float neighbor_mass = params.mass_alpha + params.mass_beta * n_inv;
                density_acc +=
                    neighbor_mass * computeInteraction(my_inv, my_cost, n_inv, n_cost, h2, poly6);

                curr = __ldg(&agent_next[curr]);
            }
        }
    }

    d_local_density[idx] = density_acc;
}

void launchBuildSpatialHash(const float* d_inventory,
                            const float* d_execution_cost,
                            int* d_cell_head,
                            int* d_agent_next,
                            MarketParams params) {
    int blockSize = 256;
    int numBlocks = (params.num_agents + blockSize - 1) / blockSize;
    buildSpatialHashKernel<<<numBlocks, blockSize>>>(d_inventory, d_execution_cost, d_cell_head,
                                                     d_agent_next, params);
}

void launchComputeLocalDensities(const float* d_inventory,
                                 const float* d_execution_cost,
                                 const int* d_cell_head,
                                 const int* d_agent_next,
                                 float* d_local_density,
                                 MarketParams params) {
    int blockSize = 256;
    int numBlocks = (params.num_agents + blockSize - 1) / blockSize;
    computeLocalDensitiesKernel<<<numBlocks, blockSize>>>(
        d_inventory, d_execution_cost, d_cell_head, d_agent_next, d_local_density, params);
}

__global__ void computeSpeedTermsKernel(const float* __restrict__ risk_aversion,
                                        const float* __restrict__ local_density,
                                        const float* __restrict__ inventory,
                                        float* __restrict__ speed_term_1,
                                        float* __restrict__ speed_term_2,
                                        int dt,
                                        MarketParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_agents)
        return;

    float personal_decay_rate = sqrtf(risk_aversion[idx] / params.temporary_impact);
    float local_temporary_impact =
        params.temporary_impact * (1 + params.congestion_sensitivity * local_density[idx]);
    speed_term_1[idx] = params.permanent_impact *
                        (1 - expf(-personal_decay_rate * (params.num_steps - dt))) /
                        (2 * local_temporary_impact * personal_decay_rate);
    speed_term_2[idx] = (2 * sqrtf(params.temporary_impact * risk_aversion[idx]) * inventory[idx]) /
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
                                       float price,
                                       MarketParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_agents)
        return;

    float local_temporary_impact =
        params.temporary_impact * (1 + params.congestion_sensitivity * local_density[idx]);
    speed[idx] = pressure * speed_term_1[idx] - speed_term_2[idx];

    float new_inv = inventory[idx] + speed[idx] * params.time_delta;
    // Clamp new inventory to 0 if it goes negative
    new_inv = fmaxf(new_inv, 0.0f);
    inventory[idx] = new_inv;
    float new_execution_cost = -local_temporary_impact * speed[idx];
    execution_cost[idx] = new_execution_cost;

    cash[idx] += -speed[idx] * (price + params.temporary_impact * speed[idx]) * params.time_delta;
}

void launchComputeSpeedTerms(const float* d_risk_aversion,
                             const float* d_local_density,
                             const float* d_inventory,
                             float* d_speed_term_1,
                             float* d_speed_term_2,
                             int dt,
                             MarketParams params) {
    int blockSize = 256;
    int numBlocks = (params.num_agents + blockSize - 1) / blockSize;

    computeSpeedTermsKernel<<<numBlocks, blockSize>>>(d_risk_aversion, d_local_density, d_inventory,
                                                      d_speed_term_1, d_speed_term_2, dt, params);
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
                            MarketParams params) {
    int blockSize = 256;
    int numBlocks = (params.num_agents + blockSize - 1) / blockSize;

    updateAgentStateKernel<<<numBlocks, blockSize>>>(
        d_speed_term_1, d_speed_term_2, d_local_density, d_agent_indices, pressure, d_speed,
        d_inventory, d_execution_cost, d_cash, price, params);
}

// Reduction Kernels
__global__ void reducePressureKernel(const float* s1, const float* s2, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float my_s1 = (i < n) ? s1[i] : 0.0f;
    float my_s2 = (i < n) ? s2[i] : 0.0f;

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Shuffle down is faster but shared memory is portable
        // Using shared memory for simplicity
        sdata[tid] = my_s1;
        sdata[tid + blockDim.x] = my_s2;
        __syncthreads();
        if (tid < s) {
            my_s1 += sdata[tid + s];
            my_s2 += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&output[0], my_s1);
        atomicAdd(&output[1], my_s2);
    }
}

// Simple global atomic reduction for now (performance is fine for 100k)
__global__ void reducePressureAtomicKernel(const float* s1, const float* s2, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&output[0], s1[i]);
        atomicAdd(&output[1], s2[i]);
    }
}

void launchComputePressure(const float* d_speed_term_1,
                           const float* d_speed_term_2,
                           float* d_pressure_buffer,
                           float* pressure,
                           int num_agents) {
    // Reset buffer
    cudaMemset(d_pressure_buffer, 0, 2 * sizeof(float));

    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    // Use atomic reduction for simplicity and correctness
    reducePressureAtomicKernel<<<numBlocks, blockSize>>>(d_speed_term_1, d_speed_term_2,
                                                         d_pressure_buffer, num_agents);

    // Copy back result
    float h_buffer[2];
    cudaMemcpy(h_buffer, d_pressure_buffer, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    *pressure = -h_buffer[1] / (1.0f - h_buffer[0]);
}

__global__ void reduceBoundariesAtomicKernel(
    const float* x, const float* y, const float* c, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val_x = x[i];
        float val_y = y[i];
        float val_c = c[i];

        atomicMinFloat(&output[0], val_y);
        atomicMaxFloat(&output[1], val_y);
        atomicMinFloat(&output[2], val_x);
        atomicMaxFloat(&output[3], val_x);
        atomicAdd(&output[4], val_c);
    }
}

__global__ void reduceBoundariesSqSumAtomicKernel(const float* c,
                                                  float* output,
                                                  float mean,
                                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = c[i] - mean;
        atomicAdd(&output[5], diff * diff);
    }
}

Boundaries launchComputeBoundaries(
    const float* d_x, const float* d_y, const float* d_c, float* d_buffer, int num_agents) {
    // Initialize buffer: [min_y, max_y, min_x, max_x, sum_c, sq_sum_c]
    float init_vals[6] = {1e30f, -1e30f, 1e30f, -1e30f, 0.0f, 0.0f};
    cudaMemcpy(d_buffer, init_vals, 6 * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    reduceBoundariesAtomicKernel<<<numBlocks, blockSize>>>(d_x, d_y, d_c, d_buffer, num_agents);

    float h_buffer[6];
    cudaMemcpy(h_buffer, d_buffer, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    float min_y = h_buffer[0];
    float max_y = h_buffer[1];
    float min_x = h_buffer[2];
    float max_x = h_buffer[3];
    float mean_c = h_buffer[4] / num_agents;

    // Second pass for stddev
    cudaMemset(&d_buffer[5], 0, sizeof(float));
    reduceBoundariesSqSumAtomicKernel<<<numBlocks, blockSize>>>(d_c, d_buffer, mean_c, num_agents);

    float sq_sum_c;
    cudaMemcpy(&sq_sum_c, &d_buffer[5], sizeof(float), cudaMemcpyDeviceToHost);

    float stddev_c = sqrtf(sq_sum_c / num_agents);
    float limit = fmaxf(fabsf(mean_c - 2.0f * stddev_c), fabsf(mean_c + 2.0f * stddev_c));
    if (limit < 0.001f)
        limit = 1.0f;

    return {min_y, max_y, min_x, max_x, -limit, limit};
}

#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <ctime>

#include "types.h"

// Constant memory for market parameters (accessible by all kernels)
__constant__ MarketParams d_params;

// Host function to copy params to constant memory
void copyParamsToDevice(const MarketParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(MarketParams));
}

// Kernel to initialize RNG states (run ONCE, not per iteration)
__global__ void setupRNGKernel(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    // Each thread initializes its own RNG state
    // This is expensive but only happens once during setup
    curand_init(seed, idx, 0, &state[idx]);
}

// Kernel to initialize inventories using pre-initialized RNG states
__global__ void initializeInventoriesKernel(float* inventory, curandState* globalState) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    // Load state from global memory to local registers
    curandState localState = globalState[idx];

    // Generate uniform random number in (0, 1]
    float u = curand_uniform(&localState);

    // Inverse transform: -ln(U) / rate gives exponential distribution
    inventory[idx] = -logf(u) / d_params.decay_rate;

    // Save state back for next use
    globalState[idx] = localState;
}

// Kernel to initialize risk aversions using pre-initialized RNG states
__global__ void initializeRiskAversionsKernel(float* risk_aversion, curandState* globalState) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    // Load state from global memory to local registers
    curandState localState = globalState[idx];

    // Generate normally distributed random number
    float rand_normal = curand_normal(&localState);
    risk_aversion[idx] = d_params.risk_mean + d_params.risk_stddev * rand_normal;

    // Save state back for next use
    globalState[idx] = localState;
}

__global__ void calculateSpatialHashKernel(const float* inventory, const float* execution_cost,
                                           int* agent_hash, int* agent_indices) {
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

__global__ void reorderDataKernel(const int* __restrict__ sorted_indices,
                                  // Input Arrays (Read-Only)
                                  const float* __restrict__ in_inventory,
                                  const float* __restrict__ in_cost,
                                  const float* __restrict__ in_cash,
                                  const float* __restrict__ in_speed,
                                  // Output Arrays (Write-Only)
                                  float* __restrict__ out_inventory, float* __restrict__ out_cost,
                                  float* __restrict__ out_cash, float* __restrict__ out_speed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    int old_idx = sorted_indices[idx];

    out_inventory[idx] = in_inventory[old_idx];
    out_cost[idx] = in_cost[old_idx];
    out_cash[idx] = in_cash[old_idx];
    out_speed[idx] = in_speed[old_idx];
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

__device__ inline float computeInteraction(float my_inv, float my_cost, float n_inv, float n_cost,
                                           float h2, float poly6) {
    float d_inv = my_inv - n_inv;
    float d_cost = my_cost - n_cost;
    float r2 = d_inv * d_inv + d_cost * d_cost;
    return (r2 < h2) ? (poly6 * powf(h2 - r2, 3)) : 0.0f;
}

__global__ void computeLocalDensitiesKernel(const float* __restrict__ inventory,
                                            const float* __restrict__ execution_cost,
                                            const int* __restrict__ cell_start_idx,
                                            const int* __restrict__ cell_end_idxs,
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

    d_local_density[idx] = density_acc;
}

void setupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    setupRNGKernel<<<numBlocks, blockSize>>>(d_rngStates, seed);
    cudaDeviceSynchronize();
}

void launchInitializeInventories(float* d_inventory, curandState* d_rngStates, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeInventoriesKernel<<<numBlocks, blockSize>>>(d_inventory, d_rngStates);
    // No sync here - letting caller control synchronization for better performance
}

void launchInitializeRiskAversions(float* d_risk_aversion, curandState* d_rngStates,
                                   int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeRiskAversionsKernel<<<numBlocks, blockSize>>>(d_risk_aversion, d_rngStates);
    // No sync here - letting caller control synchronization for better performance
}

void launchCalculateSpatialHash(const float* d_inventory, const float* d_execution_cost,
                                int* d_agent_hash, int* d_agent_indices, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    calculateSpatialHashKernel<<<numBlocks, blockSize>>>(d_inventory, d_execution_cost,
                                                         d_agent_hash, d_agent_indices);
    // No sync here - letting caller control synchronization for better performance
}

void launchFindCellBounds(const int* d_sorted_hashes, int* d_cell_start, int* d_cell_end,
                          int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    findCellBoundsKernel<<<numBlocks, blockSize>>>(d_sorted_hashes, d_cell_start, d_cell_end);
    // No sync here - letting caller control synchronization for better performance
}

void launchSortByKey(int* d_keys, int* d_values, int num_agents) {
    // Clear any previous errors
    cudaGetLastError();

    // Ensure CUDA device is properly set before Thrust operations
    // This prevents "invalid device ordinal" errors in benchmark contexts
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // If we can't get device, try to set it explicitly
        cudaSetDevice(0);
    }

    thrust::device_ptr<int> t_keys(d_keys);
    thrust::device_ptr<int> t_values(d_values);
    thrust::sort_by_key(t_keys, t_keys + num_agents, t_values);
}

void launchReorderData(const MarketState& state, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    reorderDataKernel<<<numBlocks, blockSize>>>(
        state.d_agent_index, state.d_inventory, state.d_execution_cost, state.d_cash, state.d_speed,
        state.d_inventory_sorted, state.d_execution_cost_sorted, state.d_cash_sorted,
        state.d_speed_sorted);
    // No sync here - letting caller control synchronization for better performance
}

void launchComputeLocalDensities(const float* d_inventory, const float* d_execution_cost,
                                 const int* d_cell_start, const int* d_cell_end,
                                 float* d_local_density, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    computeLocalDensitiesKernel<<<numBlocks, blockSize>>>(
        d_inventory, d_execution_cost, d_cell_start, d_cell_end, d_local_density);
    // No sync here - letting caller control synchronization for better performance
}
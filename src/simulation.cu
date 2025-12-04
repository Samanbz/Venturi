#include <curand_kernel.h>

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

    // 3. Wrap to table size
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

__global__ void computeLocalDensitiesKernel(const float* inventory, const float* execution_cost,
                                            const int* cell_start_idx, const int* cell_end_idxs,
                                            float* d_local_density) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_params.num_agents)
        return;

    int cell_idx = agent_cell_indices[idx];

    float local_density = 0.0f;
    for (int neighbor_idx = cell_start_idx[cell_idx]; neighbor_idx < cell_end_idxs[cell_idx];
         neighbor_idx++) {
        // L2 distance between execution cost and inventory
        float dist = sqrtf(execution_cost[idx] - execution_cost[neighbor_idx]);
        if (dist >= d_params.sph_smoothing_radius)
            continue;
        local_density +=
            inventory[neighbor_idx] *
            (315 / (64.0f * M_PI * powf(d_params.sph_smoothing_radius, 9))) *
            powf(d_params.sph_smoothing_radius * d_params.sph_smoothing_radius - dist * dist, 3);
    }
    d_local_density[idx] = local_density;
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

void launchComputeLocalDensities(const float* d_inventory, const float* d_execution_cost,
                                 const int* d_cell_start, const int* d_cell_end,
                                 float* d_local_density) {
    int blockSize = 256;
    int numBlocks = (d_params.num_agents + blockSize - 1) / blockSize;

    computeLocalDensitiesKernel<<<numBlocks, blockSize>>>(
        d_inventory, d_execution_cost, d_cell_start, d_cell_end, d_local_density);
    // No sync here - letting caller control synchronization for better performance
}
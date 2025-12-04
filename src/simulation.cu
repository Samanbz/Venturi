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

// Host wrapper functions

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
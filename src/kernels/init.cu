#include <curand_kernel.h>

#include <ctime>

#include "common.cuh"

// Kernel to initialize RNG states (run ONCE, not per iteration)
__global__ void setupRNGKernel(curandState* state, unsigned long long seed, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    // Each thread initializes its own RNG state
    // This is expensive but only happens once during setup
    curand_init(seed, idx, 0, &state[idx]);
}

void launchSetupRNG(curandState* d_rngStates, int num_agents, unsigned long long seed) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    setupRNGKernel<<<numBlocks, blockSize>>>(d_rngStates, seed, num_agents);
    cudaDeviceSynchronize();
}

// Kernel to initialize array with exponential distribution (lambda)
__global__ void initializeExponentialKernel(float* data,
                                            float lambda,
                                            curandState* globalState,
                                            int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    curandState localState = globalState[idx];
    float u = curand_uniform(&localState);
    // curand_uniform returns (0.0, 1.0]
    data[idx] = -logf(u) / lambda;
    globalState[idx] = localState;
}

void launchInitializeExponential(float* d_data,
                                 float lambda,
                                 curandState* d_rngStates,
                                 int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeExponentialKernel<<<numBlocks, blockSize>>>(d_data, lambda, d_rngStates, num_agents);
}

// Kernel to initialize array with uniform distribution [min, max]
__global__ void initializeUniformKernel(
    float* data, float min, float max, curandState* globalState, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    curandState localState = globalState[idx];
    float u = curand_uniform(&localState);
    data[idx] = min + u * (max - min);
    globalState[idx] = localState;
}

void launchInitializeUniform(
    float* d_data, float min, float max, curandState* d_rngStates, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeUniformKernel<<<numBlocks, blockSize>>>(d_data, min, max, d_rngStates, num_agents);
}

// Kernel to initialize array with normal distribution (mean, stddev)
__global__ void initializeNormalKernel(
    float* data, float mean, float stddev, curandState* globalState, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    curandState localState = globalState[idx];
    float n = curand_normal(&localState);
    data[idx] = mean + n * stddev;
    globalState[idx] = localState;
}

void launchInitializeNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeNormalKernel<<<numBlocks, blockSize>>>(d_data, mean, stddev, d_rngStates, num_agents);
}

// Kernel to initialize array with log normal distribution (mean, stddev)
__global__ void initializeLogNormalKernel(
    float* data, float mean, float stddev, curandState* globalState, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    curandState localState = globalState[idx];
    float ln = curand_log_normal(&localState, mean, stddev);
    data[idx] = ln;
    globalState[idx] = localState;
}

void launchInitializeLogNormal(
    float* d_data, float mean, float stddev, curandState* d_rngStates, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    initializeLogNormalKernel<<<numBlocks, blockSize>>>(d_data, mean, stddev, d_rngStates,
                                                        num_agents);
}

// Kernel to flip signs for the second half of agents
__global__ void flipSignsKernel(float* data, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;

    // Second half are buyers (negative inventory)
    if (idx >= num_agents / 2) {
        data[idx] = -data[idx];
    }
}

void launchFlipSigns(float* d_data, int num_agents) {
    int blockSize = 256;
    int numBlocks = (num_agents + blockSize - 1) / blockSize;

    flipSignsKernel<<<numBlocks, blockSize>>>(d_data, num_agents);
}

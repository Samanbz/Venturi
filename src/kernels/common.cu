#include <cuda_runtime.h>

#include "common.cuh"

__constant__ MarketParams d_params;

void copyParamsToDevice(const MarketParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(MarketParams));
}
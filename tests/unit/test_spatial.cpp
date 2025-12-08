#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "../common/test_common.h"

// Test fixture for spatial hashing and density computation
class SpatialFixture : public BaseTestFixture {
   protected:
    void SetUp() override {
        BaseTestFixture::SetUp();

        // Override default test parameters
        params.num_agents = 1000;

        // Copy params to device
        copyParamsToDevice(params);

        // Allocate device memory
        cudaMalloc(&d_inventory, params.num_agents * sizeof(float));
        cudaMalloc(&d_execution_cost, params.num_agents * sizeof(float));
        cudaMalloc(&d_agent_hash, params.num_agents * sizeof(int));
        cudaMalloc(&d_agent_indices, params.num_agents * sizeof(int));
        cudaMalloc(&d_cell_start, params.hash_table_size * sizeof(int));
        cudaMalloc(&d_cell_end, params.hash_table_size * sizeof(int));
        cudaMalloc(&d_local_density, params.num_agents * sizeof(float));

        // Allocate host memory
        h_inventory.resize(params.num_agents);
        h_execution_cost.resize(params.num_agents);
        h_agent_hash.resize(params.num_agents);
        h_agent_indices.resize(params.num_agents);
        h_cell_start.resize(params.hash_table_size);
        h_cell_end.resize(params.hash_table_size);
        h_local_density.resize(params.num_agents);
    }

    void TearDown() override {
        cudaFree(d_inventory);
        cudaFree(d_execution_cost);
        cudaFree(d_agent_hash);
        cudaFree(d_agent_indices);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        cudaFree(d_local_density);
    }

    void copyHashesToHost() {
        cudaMemcpy(h_agent_hash.data(), d_agent_hash, params.num_agents * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agent_indices.data(), d_agent_indices, params.num_agents * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    void copyCellBoundsToHost() {
        cudaMemcpy(h_cell_start.data(), d_cell_start, params.hash_table_size * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cell_end.data(), d_cell_end, params.hash_table_size * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    void copyDensityToHost() {
        cudaMemcpy(h_local_density.data(), d_local_density, params.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    float* d_inventory = nullptr;
    float* d_execution_cost = nullptr;
    int* d_agent_hash = nullptr;
    int* d_agent_indices = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;
    float* d_local_density = nullptr;

    std::vector<float> h_inventory;
    std::vector<float> h_execution_cost;
    std::vector<int> h_agent_hash;
    std::vector<int> h_agent_indices;
    std::vector<int> h_cell_start;
    std::vector<int> h_cell_end;
    std::vector<float> h_local_density;
};

// Tests for calculateSpatialHash kernel

TEST_F(SpatialFixture, SpatialHashProducesValidHashes) {
    // Initialize with uniform distribution
    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);
    for (int i = 0; i < params.num_agents; ++i) {
        inventory[i] = static_cast<float>(i % 100);
        execution_cost[i] = static_cast<float>(i % 100);
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    cudaDeviceSynchronize();

    copyHashesToHost();

    // All hashes should be within valid range
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_GE(h_agent_hash[i], 0) << "Hash at index " << i << " should be non-negative";
        EXPECT_LT(h_agent_hash[i], params.hash_table_size)
            << "Hash at index " << i << " should be less than hash_table_size";
    }
}

TEST_F(SpatialFixture, SpatialHashIndicesAreInitialized) {
    // Initialize with simple values
    std::vector<float> inventory(params.num_agents, 1.0f);
    std::vector<float> execution_cost(params.num_agents, 1.0f);

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    cudaDeviceSynchronize();

    copyHashesToHost();

    // Agent indices should be initialized to 0, 1, 2, ..., num_agents-1
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_EQ(h_agent_indices[i], i)
            << "Agent index at position " << i << " should equal " << i;
    }
}

TEST_F(SpatialFixture, SpatialHashDeterministic) {
    // Same input should produce same output
    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);
    for (int i = 0; i < params.num_agents; ++i) {
        inventory[i] = static_cast<float>(i % 50) * 0.5f;
        execution_cost[i] = static_cast<float>(i % 50) * 0.3f;
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    cudaDeviceSynchronize();
    copyHashesToHost();
    std::vector<int> first_run = h_agent_hash;

    // Run again
    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    cudaDeviceSynchronize();
    copyHashesToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_EQ(first_run[i], h_agent_hash[i]) << "Hash should be deterministic at index " << i;
    }
}

TEST_F(SpatialFixture, NearbyPointsShareSimilarHashes) {
    // Points close together should have higher probability of sharing hash
    params.num_agents = 100;
    copyParamsToDevice(params);

    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);

    // Create clusters of nearby points
    for (int i = 0; i < params.num_agents; ++i) {
        int cluster = i / 10;
        inventory[i] = cluster * 5.0f + (i % 10) * 0.1f;
        execution_cost[i] = cluster * 5.0f + (i % 10) * 0.1f;
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    cudaDeviceSynchronize();
    copyHashesToHost();

    // Count unique hashes - should be much less than num_agents since points are clustered
    std::set<int> unique_hashes(h_agent_hash.begin(), h_agent_hash.end());
    EXPECT_LT(unique_hashes.size(), params.num_agents / 2)
        << "Clustered points should share many hash buckets";
}

// Tests for findCellBounds kernel

TEST_F(SpatialFixture, CellBoundsHandleEmptyCells) {
    // Initialize data
    std::vector<float> inventory(params.num_agents, 1.0f);
    std::vector<float> execution_cost(params.num_agents, 1.0f);

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    // Reset cell bounds
    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    // Calculate hashes and sort
    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    cudaDeviceSynchronize();

    copyCellBoundsToHost();

    // Empty cells should have start = -1, end = -1
    int occupied_cells = 0;
    for (int i = 0; i < params.hash_table_size; ++i) {
        if (h_cell_start[i] != -1) {
            occupied_cells++;
            EXPECT_NE(h_cell_end[i], -1) << "If cell has start, it must have end at cell " << i;
            EXPECT_GT(h_cell_end[i], h_cell_start[i])
                << "Cell end must be greater than start at cell " << i;
        }
    }

    EXPECT_GT(occupied_cells, 0) << "At least some cells should be occupied";
}

TEST_F(SpatialFixture, CellBoundsCoverAllAgents) {
    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);
    for (int i = 0; i < params.num_agents; ++i) {
        inventory[i] = static_cast<float>(i % 100);
        execution_cost[i] = static_cast<float>(i % 100);
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    cudaDeviceSynchronize();

    copyCellBoundsToHost();

    // Count total agents covered by cell bounds
    int total_agents = 0;
    for (int i = 0; i < params.hash_table_size; ++i) {
        if (h_cell_start[i] != -1) {
            total_agents += (h_cell_end[i] - h_cell_start[i]);
        }
    }

    EXPECT_EQ(total_agents, params.num_agents) << "All agents should be covered by cell bounds";
}

// Tests for computeLocalDensities kernel

TEST_F(SpatialFixture, LocalDensityAllPositive) {
    // Initialize with random-ish data
    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);
    for (int i = 0; i < params.num_agents; ++i) {
        inventory[i] = static_cast<float>((i * 17) % 100) * 0.1f;
        execution_cost[i] = static_cast<float>((i * 23) % 100) * 0.1f;
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    launchComputeLocalDensities(d_inventory, d_execution_cost, d_cell_start, d_cell_end,
                                d_local_density, params.num_agents);
    cudaDeviceSynchronize();

    copyDensityToHost();

    // All densities should be non-negative and finite
    for (int i = 0; i < params.num_agents; ++i) {
        EXPECT_GE(h_local_density[i], 0.0f)
            << "Local density should be non-negative at index " << i;
        EXPECT_FALSE(std::isnan(h_local_density[i]))
            << "Local density should not be NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_local_density[i]))
            << "Local density should not be Inf at index " << i;
    }
}

TEST_F(SpatialFixture, LocalDensityNonZeroForIdenticalPoints) {
    // Simpler test: when all points are at the same location, density should be non-zero
    params.num_agents = 100;
    params.sph_smoothing_radius = 1.0f;
    copyParamsToDevice(params);

    std::vector<float> inventory(params.num_agents, 10.0f);       // All at same position
    std::vector<float> execution_cost(params.num_agents, 10.0f);  // All at same position

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    launchComputeLocalDensities(d_inventory, d_execution_cost, d_cell_start, d_cell_end,
                                d_local_density, params.num_agents);
    cudaDeviceSynchronize();

    copyDensityToHost();

    // When all points are at the same location, density should be computed
    // (even if it's the kernel weight times the number of overlapping points)
    float total_density = 0.0f;
    for (int i = 0; i < params.num_agents; ++i) {
        total_density += h_local_density[i];
    }
    float avg_density = total_density / params.num_agents;

    // With 100 overlapping points, some density should be computed
    EXPECT_GT(avg_density, 0.0f) << "Density should be positive for overlapping points, got avg="
                                 << avg_density;
}

// Edge case tests

TEST_F(SpatialFixture, SmallNumberOfAgents) {
    params.num_agents = 10;
    copyParamsToDevice(params);

    std::vector<float> inventory(params.num_agents, 5.0f);
    std::vector<float> execution_cost(params.num_agents, 5.0f);

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    // Should not crash
    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    launchComputeLocalDensities(d_inventory, d_execution_cost, d_cell_start, d_cell_end,
                                d_local_density, params.num_agents);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

TEST_F(SpatialFixture, LargeNumberOfAgents) {
    params.num_agents = 100000;
    params.hash_table_size = 8192;
    copyParamsToDevice(params);

    // Reallocate for larger size
    TearDown();
    SetUp();

    std::vector<float> inventory(params.num_agents);
    std::vector<float> execution_cost(params.num_agents);
    for (int i = 0; i < params.num_agents; ++i) {
        inventory[i] = static_cast<float>(i % 1000) * 0.1f;
        execution_cost[i] = static_cast<float>(i % 1000) * 0.1f;
    }

    cudaMemcpy(d_inventory, inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_start, -1, params.hash_table_size * sizeof(int));
    cudaMemset(d_cell_end, -1, params.hash_table_size * sizeof(int));

    // Should handle large scale
    launchCalculateSpatialHash(d_inventory, d_execution_cost, d_agent_hash, d_agent_indices,
                               params.num_agents);
    launchSortByKey(d_agent_hash, d_agent_indices, params.num_agents);
    launchFindCellBounds(d_agent_hash, d_cell_start, d_cell_end, params.num_agents);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

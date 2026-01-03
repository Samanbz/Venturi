#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "../common/test_common.h"
#include "kernels/launchers.h"

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
        cudaMalloc(&d_cell_head, params.hash_table_size * sizeof(int));
        cudaMalloc(&d_agent_next, params.num_agents * sizeof(int));
        cudaMalloc(&d_local_density, params.num_agents * sizeof(float));

        // Allocate host memory
        h_inventory.resize(params.num_agents);
        h_execution_cost.resize(params.num_agents);
        h_cell_head.resize(params.hash_table_size);
        h_agent_next.resize(params.num_agents);
        h_local_density.resize(params.num_agents);
    }

    void TearDown() override {
        cudaFree(d_inventory);
        cudaFree(d_execution_cost);
        cudaFree(d_cell_head);
        cudaFree(d_agent_next);
        cudaFree(d_local_density);
    }

    void copyHashDataToHost() {
        cudaMemcpy(h_cell_head.data(), d_cell_head, params.hash_table_size * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agent_next.data(), d_agent_next, params.num_agents * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    void copyDensityToHost() {
        cudaMemcpy(h_local_density.data(), d_local_density, params.num_agents * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    float* d_inventory = nullptr;
    float* d_execution_cost = nullptr;
    int* d_cell_head = nullptr;
    int* d_agent_next = nullptr;
    float* d_local_density = nullptr;

    std::vector<float> h_inventory;
    std::vector<float> h_execution_cost;
    std::vector<int> h_cell_head;
    std::vector<int> h_agent_next;
    std::vector<float> h_local_density;
};

TEST_F(SpatialFixture, SpatialHashProducesValidLinkedLists) {
    // Initialize with uniform distribution
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = static_cast<float>(i % 100);
        h_execution_cost[i] = static_cast<float>(i % 100);
    }

    cudaMemcpy(d_inventory, h_inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, h_execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    // Reset heads
    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));

    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);
    cudaDeviceSynchronize();

    copyHashDataToHost();

    // Verify all agents are in the list exactly once
    std::vector<bool> agent_found(params.num_agents, false);
    int agents_count = 0;

    for (int i = 0; i < params.hash_table_size; ++i) {
        int curr = h_cell_head[i];
        while (curr != -1) {
            ASSERT_GE(curr, 0);
            ASSERT_LT(curr, params.num_agents);
            ASSERT_FALSE(agent_found[curr])
                << "Agent " << curr << " found twice (cycle or duplicate)";
            agent_found[curr] = true;
            agents_count++;
            curr = h_agent_next[curr];

            // Prevent infinite loop in test if cycle exists
            ASSERT_LE(agents_count, params.num_agents + 1);
        }
    }

    ASSERT_EQ(agents_count, params.num_agents);
}

TEST_F(SpatialFixture, SpatialHashDeterministic) {
    // Initialize
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = static_cast<float>(i * 1.5f);
        h_execution_cost[i] = static_cast<float>(i * 0.5f);
    }
    cudaMemcpy(d_inventory, h_inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, h_execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    // Run 1
    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));
    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);
    cudaDeviceSynchronize();

    std::vector<int> head1(params.hash_table_size);
    std::vector<int> next1(params.num_agents);
    cudaMemcpy(head1.data(), d_cell_head, params.hash_table_size * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(next1.data(), d_agent_next, params.num_agents * sizeof(int), cudaMemcpyDeviceToHost);

    // Run 2
    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));
    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);
    cudaDeviceSynchronize();

    std::vector<int> head2(params.hash_table_size);
    std::vector<int> next2(params.num_agents);
    cudaMemcpy(head2.data(), d_cell_head, params.hash_table_size * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(next2.data(), d_agent_next, params.num_agents * sizeof(int), cudaMemcpyDeviceToHost);

    // Note: Atomic operations order is NOT deterministic for the linked list order within a cell.
    // However, the set of agents in each cell should be deterministic.
    // So we can't compare head/next arrays directly if there are collisions.
    // But if we ensure no collisions (sparse), we can. Or we verify content of cells.

    // Let's verify that for each cell, the set of agents is the same.
    for (int i = 0; i < params.hash_table_size; ++i) {
        std::set<int> agents1;
        int curr = head1[i];
        while (curr != -1) {
            agents1.insert(curr);
            curr = next1[curr];
        }

        std::set<int> agents2;
        curr = head2[i];
        while (curr != -1) {
            agents2.insert(curr);
            curr = next2[curr];
        }

        ASSERT_EQ(agents1, agents2) << "Cell " << i << " content mismatch";
    }
}

TEST_F(SpatialFixture, NearbyPointsShareSameCell) {
    params.num_agents = 2;
    copyParamsToDevice(params);

    // Two points very close to each other
    h_inventory[0] = 10.0f;
    h_execution_cost[0] = 10.0f;

    h_inventory[1] = 10.0f + params.sph_smoothing_radius * 0.1f;
    h_execution_cost[1] = 10.0f + params.sph_smoothing_radius * 0.1f;

    cudaMemcpy(d_inventory, h_inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, h_execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));
    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);
    cudaDeviceSynchronize();

    copyHashDataToHost();

    // Find the cell containing agent 0
    int cell_idx = -1;
    for (int i = 0; i < params.hash_table_size; ++i) {
        int curr = h_cell_head[i];
        while (curr != -1) {
            if (curr == 0) {
                cell_idx = i;
                break;
            }
            curr = h_agent_next[curr];
        }
        if (cell_idx != -1)
            break;
    }

    ASSERT_NE(cell_idx, -1) << "Agent 0 not found in any cell";

    // Check if agent 1 is in the same cell
    bool found_1 = false;
    int curr = h_cell_head[cell_idx];
    while (curr != -1) {
        if (curr == 1)
            found_1 = true;
        curr = h_agent_next[curr];
    }

    ASSERT_TRUE(found_1) << "Agent 1 should be in the same cell as Agent 0";
}

TEST_F(SpatialFixture, LocalDensityAllPositive) {
    // Random distribution
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = static_cast<float>(i % 50);
        h_execution_cost[i] = static_cast<float>((i * 7) % 50);
    }
    cudaMemcpy(d_inventory, h_inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, h_execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));
    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);

    launchComputeLocalDensities(d_inventory, d_execution_cost, d_cell_head, d_agent_next,
                                d_local_density, params);
    cudaDeviceSynchronize();

    copyDensityToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        ASSERT_GE(h_local_density[i], 0.0f) << "Density negative at index " << i;
    }
}

TEST_F(SpatialFixture, LocalDensityNonZeroForIdenticalPoints) {
    params.num_agents = 10;
    copyParamsToDevice(params);

    // All points at same location
    for (int i = 0; i < params.num_agents; ++i) {
        h_inventory[i] = 50.0f;
        h_execution_cost[i] = 50.0f;
    }
    cudaMemcpy(d_inventory, h_inventory.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_execution_cost, h_execution_cost.data(), params.num_agents * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_cell_head, -1, params.hash_table_size * sizeof(int));
    launchBuildSpatialHash(d_inventory, d_execution_cost, d_cell_head, d_agent_next, params);

    launchComputeLocalDensities(d_inventory, d_execution_cost, d_cell_head, d_agent_next,
                                d_local_density, params);
    cudaDeviceSynchronize();

    copyDensityToHost();

    for (int i = 0; i < params.num_agents; ++i) {
        ASSERT_GT(h_local_density[i], 0.0f) << "Density should be positive";
    }
}

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../src/simulation.h"
#include "../../src/types.h"

// Function to check if a simulation with a specific number of agents maintains the target FPS
bool checkPerformance(int num_agents, int max_steps, float max_frame_time_ms) {
    MarketParams params;
    params.num_agents = num_agents;
    params.num_steps = max_steps;
    params.time_delta = 1.0f / 60.0f;
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.1f;
    params.permanent_impact = 1e-5f;
    params.temporary_impact = 0.01f;
    params.sph_smoothing_radius = 1.0f;
    params.congestion_sensitivity = 0.05f;

    // Adjust hash table size based on agent count
    int power = 1;
    while ((1 << power) < params.num_agents) {
        power++;
    }
    params.hash_table_size = (1 << (power + 1));

    params.decay_rate = 0.0001f;
    params.mass_alpha = 0.4f;
    params.mass_beta = 0.1f;
    params.risk_mean = 0.01f;
    params.risk_stddev = 0.1f;

    Simulation* sim = nullptr;
    try {
        sim = new Simulation(params);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize simulation with " << num_agents
                  << " agents: " << e.what() << std::endl;
        return false;
    }

    bool success = true;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        sim->step(false, false);
    }
    cudaDeviceSynchronize();

    float max_time = 0.0f;
    int max_time_step = -1;

    for (int step = 0; step < max_steps; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        sim->step(false, false);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;

        if (duration.count() > max_time) {
            max_time = duration.count();
            max_time_step = step;
        }

        if (duration.count() > max_frame_time_ms) {
            std::cout << "FAIL (Step " << step << ": " << duration.count() << "ms) " << std::flush;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "PASS (Max: " << max_time << "ms at step " << max_time_step << ") "
                  << std::flush;
    } else {
        // Already printed FAIL
    }
    std::cout << std::endl;

    delete sim;
    return success;
}

int findMaxAgents(int min_agents, int max_agents, int steps, float max_frame_time_ms) {
    int low = min_agents;
    int high = max_agents;
    int result = 0;

    std::cout << "Searching for max agents for " << (1000.0f / max_frame_time_ms)
              << " FPS (limit: " << max_frame_time_ms << "ms)..." << std::endl;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        // Round to nearest 1000 for cleaner numbers
        mid = (mid / 1000) * 1000;
        if (mid < low)
            mid = low;  // Avoid getting stuck if low is small

        std::cout << "  Testing " << mid << " agents... " << std::flush;
        if (checkPerformance(mid, steps, max_frame_time_ms)) {
            result = mid;
            low = mid + 1000;  // Step up by at least 1000
        } else {
            high = mid - 1000;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    int steps = 1000;  // Default check duration
    if (argc > 1) {
        steps = std::atoi(argv[1]);
    }

    std::cout << "Determining max agents for " << steps << " steps." << std::endl;

    // Search range
    int min_search = 1000;
    int max_search = 500000;  // Upper limit

    // 60 FPS = 16.66 ms
    int max_agents_60fps = findMaxAgents(min_search, max_search, steps, 16.6f);
    std::cout << "Max agents for 60 FPS: " << max_agents_60fps << std::endl;

    // 30 FPS = 33.33 ms
    // Start search from max_agents_60fps since it must be >=
    int max_agents_30fps =
        findMaxAgents(std::max(min_search, max_agents_60fps), max_search, steps, 33.3f);
    std::cout << "Max agents for 30 FPS: " << max_agents_30fps << std::endl;

    return 0;
}

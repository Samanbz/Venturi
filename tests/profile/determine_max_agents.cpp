#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../src/config.h"
#include "../../src/simulation.h"
#include "../../src/types.h"

// Function to check if a simulation with a specific number of agents maintains the target FPS
bool checkPerformance(SimConfig config, int check_duration_frames) {
    // Ensure derived params are set for this specific agent count/FPS
    config.finalize();

    float max_frame_time_ms = 1000.0f / config.targetFPS;
    int steps_per_frame = config.stepsPerFrame;

    Simulation* sim = nullptr;
    try {
        sim = new Simulation(config.marketParams);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize simulation with " << config.marketParams.num_agents
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
    int max_time_frame = -1;

    for (int frame = 0; frame < check_duration_frames; ++frame) {
        auto start = std::chrono::high_resolution_clock::now();

        // Run the number of physics steps required for one frame
        for (int s = 0; s < steps_per_frame; ++s) {
            sim->step(false, false);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;

        if (duration.count() > max_time) {
            max_time = duration.count();
            max_time_frame = frame;
        }

        if (duration.count() > max_frame_time_ms) {
            std::cout << "FAIL (Frame " << frame << ": " << duration.count() << "ms > "
                      << max_frame_time_ms << "ms) " << std::flush;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "PASS (Max: " << max_time << "ms at frame " << max_time_frame << ") "
                  << std::flush;
    } else {
        // Already printed FAIL
    }
    std::cout << std::endl;

    delete sim;
    return success;
}

int findMaxAgents(SimConfig baseConfig, int min_agents, int max_agents, int check_duration_frames) {
    int low = min_agents;
    int high = max_agents;
    int result = 0;

    std::cout << "Searching for max agents for " << baseConfig.targetFPS << " FPS..." << std::endl;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        // Round to nearest 1000 for cleaner numbers
        mid = (mid / 1000) * 1000;
        if (mid < low)
            mid = low;  // Avoid getting stuck

        // Configure for this attempt
        SimConfig testConfig = baseConfig;
        testConfig.marketParams.num_agents = mid;

        std::cout << "  Testing " << mid << " agents... " << std::flush;
        if (checkPerformance(testConfig, check_duration_frames)) {
            result = mid;
            low = mid + 1000;
        } else {
            high = mid - 1000;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    // Parse Config (handling standard flags like --fps, --speedup)
    SimConfig baseConfig = parseArgs(argc, argv);

    // Determine testing duration (frames)
    // We repurpose baseConfig.numFrames if it was set, default to 1000 for quick
    // test
    int test_frames = baseConfig.numFrames;
    if (test_frames > 5000)
        test_frames = 1000;  // Cap default if it seems like a full run config

    // Search range
    int min_search = 1000;
    int max_search = 1000000;  // 1 Million upper limit

    std::cout << "Determining max agents." << std::endl;
    std::cout << "Target FPS: " << baseConfig.targetFPS << std::endl;
    std::cout << "Test Duration: " << test_frames << " frames" << std::endl;

    int max_agents = findMaxAgents(baseConfig, min_search, max_search, test_frames);
    std::cout << "Max agents for " << baseConfig.targetFPS << " FPS: " << max_agents << std::endl;

    return 0;
}

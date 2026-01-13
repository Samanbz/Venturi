#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/config.h"
#include "../../src/simulation.h"
#include "../../src/types.h"

int main(int argc, char* argv[]) {
    // 1. Use Unified Config Parser
    SimConfig config = parseArgs(argc, argv);

    // 2. Handle Profile-Specific Arguments manually if needed, or just defaults
    std::string outputFile = "profile_results.csv";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        }
    }

    // Ensure we have a valid time delta even if we didn't specify FPS
    if (config.marketParams.time_delta <= 0.0f) {
        config.finalize();  // Recalculate if needed
    }

    std::cout << "Running Profile with:" << std::endl;
    std::cout << "  Agents: " << config.marketParams.num_agents << std::endl;
    std::cout << "  Steps: " << config.marketParams.num_steps << std::endl;
    std::cout << "  Time Delta: " << config.marketParams.time_delta << std::endl;
    std::cout << "  Output: " << outputFile << std::endl;
    std::cout << "  Hash Table Size: " << config.marketParams.hash_table_size << std::endl;

    // Initialize Simulation
    // We don't need Vulkan pointers for headless profiling
    Simulation sim(config.marketParams);

    std::ofstream csv(outputFile);
    csv << "step,duration_ms,fps_instant\n";

    std::cout << "Starting simulation..." << std::endl;

    using namespace std::chrono;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        sim.step(false, false);  // No render sync
    }
    cudaDeviceSynchronize();

    auto totalStart = high_resolution_clock::now();

    for (int step = 0; step < config.marketParams.num_steps; ++step) {
        auto start = high_resolution_clock::now();

        sim.step(false, false);   // No render sync
        cudaDeviceSynchronize();  // Ensure GPU is done for accurate timing

        auto end = high_resolution_clock::now();
        double duration_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double fps = 1000.0 / (duration_ms > 0 ? duration_ms : 0.001);

        csv << step << "," << duration_ms << "," << fps << "\n";

        if (step % 100 == 0) {
            std::cout << "Step " << step << ": " << duration_ms << " ms (" << fps << " FPS)"
                      << std::endl;
        }
    }

    auto totalEnd = high_resolution_clock::now();
    double totalDuration = duration_cast<seconds>(totalEnd - totalStart).count();
    std::cout << "Finished in " << totalDuration << " seconds." << std::endl;

    csv.close();
    return 0;
}

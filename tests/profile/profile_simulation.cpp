#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/simulation.h"
#include "../../src/types.h"

// Simple command line argument parser
struct ProfileParams {
    int num_agents = 100000;
    int num_steps = 1000;
    std::string output_file = "profile_results.csv";
    int distribution_type = 0;  // 0: Uniform, 1: Gaussian
    float risk_mean = 0.001f;
    float risk_stddev = 0.6f;
};

ProfileParams parseArgs(int argc, char* argv[]) {
    ProfileParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num_agents" && i + 1 < argc) {
            params.num_agents = std::stoi(argv[++i]);
        } else if (arg == "--num_steps" && i + 1 < argc) {
            params.num_steps = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            params.output_file = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            params.distribution_type = std::stoi(argv[++i]);
        } else if (arg == "--risk_mean" && i + 1 < argc) {
            params.risk_mean = std::stof(argv[++i]);
        } else if (arg == "--risk_stddev" && i + 1 < argc) {
            params.risk_stddev = std::stof(argv[++i]);
        }
    }
    return params;
}

int main(int argc, char* argv[]) {
    ProfileParams profileParams = parseArgs(argc, argv);

    std::cout << "Running Profile with:" << std::endl;
    std::cout << "  Agents: " << profileParams.num_agents << std::endl;
    std::cout << "  Steps: " << profileParams.num_steps << std::endl;
    std::cout << "  Output: " << profileParams.output_file << std::endl;

    MarketParams params;
    params.num_agents = profileParams.num_agents;
    params.risk_mean = profileParams.risk_mean;
    params.risk_stddev = profileParams.risk_stddev;

    // Adjust hash table size based on agent count to be fair
    // Using a power of 2 roughly 2x the agent count is a good heuristic for spatial hashing
    int power = 1;
    while ((1 << power) < params.num_agents) {
        power++;
    }
    params.hash_table_size = (1 << (power + 1));
    std::cout << "  Hash Table Size: " << params.hash_table_size << std::endl;

    // Initialize Simulation
    // We don't need Vulkan pointers for headless profiling
    Simulation sim(params);

    std::ofstream csv(profileParams.output_file);
    csv << "step,duration_ms,fps_instant\n";

    std::cout << "Starting simulation..." << std::endl;

    using namespace std::chrono;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        sim.step(false, false);  // No render sync
    }
    cudaDeviceSynchronize();

    auto totalStart = high_resolution_clock::now();

    for (int step = 0; step < profileParams.num_steps; ++step) {
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

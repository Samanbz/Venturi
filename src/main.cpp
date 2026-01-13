#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "canvas/canvas.h"
#include "canvas/offline_canvas.h"
#include "canvas/realtime_canvas.h"
#include "simulation.h"

static void printBoundaries(const Boundaries& boundaries) {
    std::cout << "Y Boundaries: [" << boundaries.minY << ", " << boundaries.maxY << "]\n";
    std::cout << "X Boundaries: [" << boundaries.minX << ", " << boundaries.maxX << "]\n";
    std::cout << "Color Boundaries: [" << boundaries.minColor << ", " << boundaries.maxColor
              << "]\n";
}

int main(int argc, char** argv) {
    bool offlineMode = false;
    std::string outputDir = "output";
    int numFrames = 5000;   // Default
    int numAgents = 50000;  // Default
    int width = 0;
    int height = 0;
    float latencyMean = 2.0f;
    float latencyJitter = 1.0f;
    int maxLatencySteps = 1024;  // Default buffer size

    int targetFPS = 60;
    float speedUp = 1.0f;
    float preferredDelta = 0.01f;  // 10ms stability target
    float historyDuration = 30.0f;
    float smoothingAlpha = 0.5f;  // 1.0 = No smoothing

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--offline") == 0) {
            offlineMode = true;
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            numFrames = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--agents") == 0 && i + 1 < argc) {
            numAgents = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--latency-mean") == 0 && i + 1 < argc) {
            latencyMean = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--latency-jitter") == 0 && i + 1 < argc) {
            latencyJitter = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            targetFPS = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--speedup") == 0 && i + 1 < argc) {
            speedUp = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--history") == 0 && i + 1 < argc) {
            historyDuration = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--smooth") == 0 && i + 1 < argc) {
            smoothingAlpha = std::stof(argv[++i]);
        }
    }

    if (targetFPS <= 0) {
        targetFPS = 60;
    }

    // Calculate steps and timedelta
    float simTimePerFrame = speedUp / static_cast<float>(targetFPS);
    int stepsPerFrame = std::max(1, static_cast<int>(std::round(simTimePerFrame / preferredDelta)));
    float actualDelta = simTimePerFrame / stepsPerFrame;

    std::cout << "Configuration:\n";
    std::cout << "  Target FPS: " << targetFPS << "\n";
    std::cout << "  Speed Up: " << speedUp << "x\n";
    std::cout << "  Simulation Time Per Frame: " << simTimePerFrame << "s\n";
    std::cout << "  Internal Time Delta: " << actualDelta << "s (" << (1.0f / actualDelta)
              << " Hz)\n";
    std::cout << "  Steps Per Frame: " << stepsPerFrame << "\n";

    if (width == 0)
        width = 1920;
    if (height == 0)
        height = 1080;

    MarketParams params{};
    params.num_agents = numAgents;
    params.num_steps = numFrames * stepsPerFrame;

    // Latency
    params.latency_mean = latencyMean;
    params.latency_jitter_stddev = latencyJitter;
    params.max_latency_steps = maxLatencySteps;

    params.time_delta = actualDelta;
    params.price_init = 100.0f;
    params.price_randomness_stddev = 1.5f;
    params.permanent_impact = 1e-6f;
    params.temporary_impact = 0.1f;
    params.sph_smoothing_radius = 1.0f;
    params.congestion_sensitivity = 0.06f;
    int power = 1;
    while ((1 << power) < params.num_agents) {
        power++;
    }
    params.hash_table_size = (1 << (power + 1));
    params.decay_rate = 3e-4f;
    params.mass_alpha = 0.001f;
    params.mass_beta = 0.3f;
    params.risk_mean = 1.0f;
    params.risk_stddev = 0.9f;
    params.greed_mean = 2.0f;
    params.greed_stddev = 20.0f;
    params.trend_decay = 0.5f;
    params.target_inventory_mean = 0.0f;
    params.target_inventory_stddev = 1e2f;

    std::unique_ptr<Canvas> canvas;

    if (offlineMode) {
        std::cout << "Starting offline rendering..." << std::endl;
        std::cout << "Agents: " << numAgents << std::endl;
        std::cout << "Frames: " << numFrames << std::endl;
        std::cout << "Output Directory: " << outputDir << std::endl;
        std::filesystem::create_directories(outputDir);

        canvas =
            std::make_unique<OfflineCanvas>(params.num_agents, static_cast<uint32_t>(width),
                                            static_cast<uint32_t>(height), outputDir, numFrames);
    } else {
        auto rtCanvas = std::make_unique<RealTimeCanvas>(
            params.num_agents, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        rtCanvas->setHistoryDuration(historyDuration);
        rtCanvas->setSmoothingAlpha(smoothingAlpha);
        canvas = std::move(rtCanvas);
    }

    canvas->setTargetFPS(targetFPS);
    canvas->setStepsPerFrame(stepsPerFrame);

    auto [X_devicePtr, Y_devicePtr, Color_devicePtr] = canvas->getCudaDevicePointers();
    auto [fdWait, fdSignal] = canvas->exportSemaphores();

    Simulation sim{params,
                   X_devicePtr,
                   Y_devicePtr,
                   Color_devicePtr,
                   PlotVar::ExecutionCost,
                   PlotVar::Inventory,
                   PlotVar::Speed};
    sim.importSemaphores(fdWait, fdSignal);

    // Initial step to stabilize and compute initial boundaries
    sim.step();

    // Initial boundary setup
    Boundaries boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas->setBoundaries(boundaries, 10.0f, 10.0f, true);

    // Run the loop
    canvas->run(sim);

    if (offlineMode) {
        std::cout << "\nOffline rendering complete." << std::endl;
    }

    return 0;
}

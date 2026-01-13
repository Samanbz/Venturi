#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "canvas/canvas.h"
#include "canvas/offline_canvas.h"
#include "canvas/realtime_canvas.h"
#include "config.h"
#include "simulation.h"

static void printBoundaries(const Boundaries& boundaries) {
    std::cout << "Y Boundaries: [" << boundaries.minY << ", " << boundaries.maxY << "]\n";
    std::cout << "X Boundaries: [" << boundaries.minX << ", " << boundaries.maxX << "]\n";
    std::cout << "Color Boundaries: [" << boundaries.minColor << ", " << boundaries.maxColor
              << "]\n";
}

int main(int argc, char** argv) {
    // Parse Configuration
    SimConfig config = parseArgs(argc, argv);

    std::cout << "Configuration:\n";
    std::cout << "  Target FPS: " << config.targetFPS << "\n";
    std::cout << "  Speed Up: " << config.speedUp << "x\n";
    std::cout << "  Steps Per Frame: " << config.stepsPerFrame << "\n";
    std::cout << "  Time Delta: " << config.marketParams.time_delta << "s\n";
    std::cout << "  Agents: " << config.marketParams.num_agents << "\n";

    // Setup Canvas
    std::unique_ptr<Canvas> canvas;

    if (config.offlineMode) {
        std::cout << "Starting offline rendering..." << std::endl;
        std::cout << "Frames: " << config.numFrames << std::endl;
        std::cout << "Output Directory: " << config.outputDir << std::endl;
        std::filesystem::create_directories(config.outputDir);

        canvas = std::make_unique<OfflineCanvas>(
            config.marketParams.num_agents, static_cast<uint32_t>(config.width),
            static_cast<uint32_t>(config.height), config.outputDir, config.numFrames);
    } else {
        auto rtCanvas = std::make_unique<RealTimeCanvas>(config.marketParams.num_agents,
                                                         static_cast<uint32_t>(config.width),
                                                         static_cast<uint32_t>(config.height));
        rtCanvas->setHistoryDuration(config.historyDuration);
        rtCanvas->setSmoothingAlpha(config.smoothingAlpha);
        canvas = std::move(rtCanvas);
    }

    canvas->setTargetFPS(config.targetFPS);
    canvas->setStepsPerFrame(config.stepsPerFrame);
    canvas->setZoomSchedule(config.zoomStart, config.zoomEnd, config.zoomDuration);
    canvas->setGridConfig(config.showGrid, config.gridSpacing);

    // Setup Simulation
    auto [X_devicePtr, Y_devicePtr, Color_devicePtr] = canvas->getCudaDevicePointers();

    auto [fdWait, fdSignal] = canvas->exportSemaphores();

    Simulation sim{config.marketParams,    X_devicePtr,        Y_devicePtr,   Color_devicePtr,
                   PlotVar::ExecutionCost, PlotVar::Inventory, PlotVar::Speed};
    sim.importSemaphores(fdWait, fdSignal);

    // Initial step to stabilize and compute initial boundaries
    sim.step();

    // Initial boundary setup
    Boundaries boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas->setBoundaries(boundaries, config.zoomStart, config.zoomStart, true);

    // Run Loop
    canvas->run(sim);

    if (config.offlineMode) {
        std::cout << "\nOffline rendering complete." << std::endl;
    }

    return 0;
}

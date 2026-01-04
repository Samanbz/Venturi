
#include <iostream>

#include "canvas.h"
#include "simulation.h"

static void printBoundaries(const Boundaries& boundaries) {
    std::cout << "Y Boundaries: [" << boundaries.minY << ", " << boundaries.maxY << "]\n";
    std::cout << "X Boundaries: [" << boundaries.minX << ", " << boundaries.maxX << "]\n";
    std::cout << "Color Boundaries: [" << boundaries.minColor << ", " << boundaries.maxColor
              << "]\n";
}

int main() {
    MarketParams params{};
    params.num_agents = 10000;
    params.num_steps = 5000;

    params.time_delta = 1.0f / 30.0f;
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.8f;  // Increased for visual noise

    // Weaken the permanent feedback loop to ensure Sum(C1) < 1.0
    params.permanent_impact = 1e-6f;
    // Strengthen the cost of trading to act as a "brake" on speed
    params.temporary_impact = 0.01f;

    params.sph_smoothing_radius = 1.0f;
    params.congestion_sensitivity = 0.01f;

    // Dynamically size the hash table to ensure low collision rates
    // Target: ~0.5 agents per bucket or less
    int power = 1;
    while ((1 << power) < params.num_agents) {
        power++;
    }
    params.hash_table_size = (1 << (power + 1));

    params.decay_rate = 0.00006f;

    // Agent Parameters
    params.mass_alpha = 0.4f;
    params.mass_beta = 0.1f;
    params.risk_mean = 0.001f;  // Lower risk aversion usually looks better visually
    params.risk_stddev = 2.0f;

    Canvas canvas{params.num_agents};

    auto [X_devicePtr, Y_devicePtr, Color_devicePtr] = canvas.getCudaDevicePointers();
    auto [fdWait, fdSignal] = canvas.exportSemaphores();

    Simulation sim{params,
                   X_devicePtr,
                   Y_devicePtr,
                   Color_devicePtr,
                   PlotVar::ExecutionCost,
                   PlotVar::Inventory,
                   PlotVar::Speed};
    sim.importSemaphores(fdWait, fdSignal);

    sim.step();

    Boundaries boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas.setBoundaries(boundaries, 0.1f, 0.1f, true);

    // Run at 60 FPS, with 1 simulation step per frame
    canvas.mainLoop(sim, 30, 1);
    return 0;
}
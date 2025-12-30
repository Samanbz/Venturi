
#include <iostream>

#include "canvas.h"
#include "simulation.h"

static void printBoundaries(const BoundaryPair& boundaries) {
    std::cout << "Y Boundaries: [" << boundaries.first.first << ", " << boundaries.first.second
              << "]\n";
    std::cout << "X Boundaries: [" << boundaries.second.first << ", " << boundaries.second.second
              << "]\n";
}

int main() {
    MarketParams params{};
    params.num_agents = 100000;
    params.num_steps = 10000;

    params.time_delta = 1.0f / 60.0f;
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.8f;  // Increased for visual noise

    // Weaken the permanent feedback loop to ensure Sum(C1) < 1.0
    params.permanent_impact = 1e12f;
    // Strengthen the cost of trading to act as a "brake" on speed
    params.temporary_impact = 0.01f;

    params.sph_smoothing_radius = 1.5f;
    params.congestion_sensitivity = 0.05f;

    params.hash_table_size = exp2(12);

    params.decay_rate = 0.0005f;

    // Agent Parameters
    params.mass_alpha = 0.4f;
    params.mass_beta = 0.1f;
    params.risk_mean = 0.001f;  // Lower risk aversion usually looks better visually
    params.risk_stddev = 0.6f;

    Canvas canvas{params.num_agents};

    auto [X_devicePtr, Y_devicePtr] = canvas.getCudaDevicePointers();
    auto [fdWait, fdSignal] = canvas.exportSemaphores();

    Simulation sim{params, X_devicePtr, Y_devicePtr, PlotVar::ExecutionCost, PlotVar::Inventory};
    sim.importSemaphores(fdWait, fdSignal);

    sim.step();

    BoundaryPair boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas.setBoundaries(boundaries, 0.1f, 0.1f, true);

    canvas.mainLoop(sim);
    return 0;
}

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
    params.num_agents = 200000;
    params.num_steps = 100000;

    params.time_delta = 1.0f / 10.0f;  // Standard 30FPS
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.9f;  // Increased for visual noise

    // Weaken the permanent feedback loop to ensure Sum(C1) < 1.0
    params.permanent_impact = 1e6f;
    // Strengthen the cost of trading to act as a "brake" on speed
    params.temporary_impact = 0.2f;

    params.sph_smoothing_radius = 1.0f;
    params.congestion_sensitivity = 0.01f;  // Make traffic actually painful

    params.hash_table_size = exp2(12);

    params.decay_rate = 0.00001f;

    // Agent Parameters
    params.mass_alpha = 0.4f;
    params.mass_beta = 0.1f;
    params.risk_mean = 0.001f;  // Lower risk aversion usually looks better visually
    params.risk_stddev = 0.99f;

    Canvas canvas{params.num_agents};

    auto [X_devicePtr, Y_devicePtr] = canvas.getCudaDevicePointers();
    auto [fdWait, fdSignal] = canvas.exportSemaphores();

    Simulation sim{params, X_devicePtr, Y_devicePtr};
    sim.importSemaphores(fdWait, fdSignal);

    BoundaryPair boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas.setBoundaries(boundaries);

    canvas.mainLoop(sim);
    return 0;
}
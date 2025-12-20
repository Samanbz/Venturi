
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
    params.num_agents = 1000;
    params.num_steps = 100000;

    params.time_delta = 1.0f / 60.0f;  // Standard 60FPS
    params.price_init = 100.0f;
    params.price_randomness_stddev = 0.6f;  // Increased for visual noise

    // Weaken the permanent feedback loop to ensure Sum(C1) < 1.0
    params.permanent_impact = 1e-5f;
    // Strengthen the cost of trading to act as a "brake" on speed
    params.temporary_impact = 0.8f;

    params.sph_smoothing_radius = 5.0f;
    params.congestion_sensitivity = 0.5f;  // Make traffic actually painful

    params.hash_table_size = exp2(10);

    params.decay_rate = 0.001f;

    // Agent Parameters
    params.mass_alpha = 1.0f;
    params.mass_beta = 0.1f;
    params.risk_mean = 0.5f;  // Lower risk aversion usually looks better visually
    params.risk_stddev = 0.05f;

    Canvas canvas{params.num_agents};

    auto [X_devicePtr, Y_devicePtr] = canvas.getCudaDevicePointers();

    Simulation sim{params, X_devicePtr, Y_devicePtr};
    sim.step();

    BoundaryPair boundaries = sim.getBoundaries();
    printBoundaries(boundaries);
    canvas.setBoundaries(boundaries);

    while (!glfwWindowShouldClose(canvas.window_)) {
        glfwPollEvents();

        // 1. Run Physics
        sim.step();

        // 2. WAIT for Physics to finish writing to memory
        cudaDeviceSynchronize();

        // 3. Draw
        canvas.drawFrame();

        // 4. WAIT for Draw to finish reading memory
        vkDeviceWaitIdle(canvas.device_);
    }
    return 0;
}
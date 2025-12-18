#include "simulation.h"

int main() {
    MarketParams params;
    params.num_agents = 10000;
    params.num_steps = 1000;
    params.time_delta = 0.01f;
    params.price_init = 100.0f;
    params.permanent_impact = 0.1f;
    params.temporary_impact = 0.01f;
    params.congestion_sensitivity = 0.5f;
    params.price_randomness_stddev = 0.5f;
    params.sph_smoothing_radius = 1.0f;
    params.hash_table_size = 1024;
    params.mass_alpha = 1.0f;
    params.mass_beta = 0.1f;
    params.decay_rate = 0.05f;
    params.risk_mean = 0.5f;
    params.risk_stddev = 0.1f;

    Simulation sim{params};

    sim.initWindow();
    sim.initVulkan();
    sim.mainLoop();
    sim.cleanup();

    return 0;
}
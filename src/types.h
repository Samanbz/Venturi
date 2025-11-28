
struct MarketState {
    int dt;            // t, current time step
    float d_price;     // S_t, current asset price
    float d_pressure;  // mu_t, current market pressure

    float* d_inventory;  // Q_t^a, agent inventories
    float* d_cash;       // X_t^a, agent cash balances
    float* d_speed;      // nu_t^a, agent trading speeds
    float* d_density;    // rho_t^a, agent local densities

    float* d_risk_aversion;  // phi^a, agent risk aversion parameters
};

struct MarketParams {
    int num_agents;                // N, number of trading agents
    int num_steps;                 // T, number of simulation steps
    float price_init;              // S_0, initial asset price
    float permanent_impact;        // alpha, permanent market impact factor
    float temporary_impact;        // kappa, temporary market impact factor
    float congestion_sensitivity;  // beta, congestion sensitivity factor

    float inventory_mean;    // Mean of initial inventory distribution
    float inventory_stddev;  // Stddev of initial inventory distribution
};
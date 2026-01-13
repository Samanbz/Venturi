#pragma once
#include <random>

#include "types.h"

enum class PlotVar {
    Inventory,
    ExecutionCost,
    Cash,
    Speed,
    RiskAversion,
    LocalDensity,
    Greed,
    Belief
};

/**
 * @brief Main simulation class managing market agents and physics updates.
 *
 * The Simulation class maintains the state of all GPU-based market agents and
 * handles the time-stepping of the Mean Field Game physics (HJB/FPK equations).
 * It supports interoperability with Vulkan for visualization by accepting external
 * device pointers for specific visualization variables.
 */
class Simulation {
    friend class UpdateLogicFixture;
    friend class DensityEvolutionFixture;
    friend class SpeedPressureFixture;
    friend class LocalDensityFixture;
    friend class SpatialHashingFixture;

   public:
    /**
     * @brief Constructs the simulation and allocates GPU resources.
     *
     * @param params Market configuration parameters (agent count, SDE constants).
     * @param vk_X Optional pointer to Vulkan Vertex Buffer for X-axis position.
     * @param vk_Y Optional pointer to Vulkan Vertex Buffer for Y-axis position.
     * @param vk_Color Optional pointer to Vulkan Vertex Buffer for color data.
     * @param xVar Which agent variable to map to the X-axis (default: ExecutionCost).
     * @param yVar Which agent variable to map to the Y-axis (default: Inventory).
     * @param colorVar Which agent variable to map to the Color channel (default: Speed).
     *
     * If Vulkan pointers are provided, the simulation writes directly to them (Zero-Copy).
     * Otherwise, internal CUDA memory is allocated for these fields.
     */
    Simulation(const MarketParams& params,
               float* vk_X = nullptr,
               float* vk_Y = nullptr,
               float* vk_Color = nullptr,
               PlotVar xVar = PlotVar::ExecutionCost,
               PlotVar yVar = PlotVar::Inventory,
               PlotVar colorVar = PlotVar::Speed);

    /**
     * @brief Destructor. Frees all allocated CUDA memory and handles.
     */
    ~Simulation();

    /**
     * @brief Advances the simulation by one time step.
     *
     * Performs spatial hashing, density estimation, and SDE integration.
     *
     * @param waitForRender If true, waits for the 'cudaWaitSemaphore' before writing.
     * @param signalRender If true, signals the 'cudaSignalSemaphore' after writing.
     */
    void step(bool waitForRender = true, bool signalRender = true);

    /**
     * @brief Continuous run loop (blocking). Mainly used for console/headless mode.
     */
    void run();

    /**
     * @brief Calculates global boundaries (min/max) of the currently plotted variables.
     *
     * Useful for auto-scaling the camera view in the renderer.
     *
     * @return Boundaries struct containing min/max for X, Y, and Color.
     */
    Boundaries getBoundaries() const;

    /**
     * @brief Imports Vulkan semaphores for synchronizing compute with graphics.
     *
     * @param fdWait File descriptor for the semaphore to wait on (Render Finished).
     * @param fdSignal File descriptor for the semaphore to signal (Compute Finished).
     */
    void importSemaphores(int fdWait, int fdSignal);

    const MarketState& getState() const { return state_; }
    const MarketParams& getParams() const { return params_; }

   private:
    /**
     * @brief Launches the kernel to compute local agent density using Spatial Hashing.
     */
    void computeLocalDensities();

    /**
     * @brief Launches the kernel to aggregate global market pressure.
     */
    void computePressure();

    /**
     * @brief Launches the kernel to update agent states (Inventory, Cash, etc.) via SDEs.
     *
     * @param observed_pressure Global market pressure observed in the previous step.
     * @param price_change Change in asset price in the previous step.
     */
    void updateAgentState(float observed_pressure, float price_change);

    /**
     * @brief Updates the global asset price based on total order imbalance.
     */
    void updatePrice();

    bool externalMemoryProvided;
    bool owns_memory_ = true;  // Don't forget this from previous steps!

    MarketParams params_;
    MarketState state_;

    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;

    cudaExternalSemaphore_t cudaWaitSemaphore = nullptr;
    cudaExternalSemaphore_t cudaSignalSemaphore = nullptr;

    float* d_plot_x = nullptr;
    float* d_plot_y = nullptr;
    float* d_plot_color = nullptr;

    PlotVar colorVar_;
};
#pragma once
#include <random>

#include "types.h"

enum class PlotVar { Inventory, ExecutionCost, Cash, Speed, RiskAversion, LocalDensity };

class Simulation {
   public:
    Simulation(const MarketParams& params,
               float* vk_X = nullptr,
               float* vk_Y = nullptr,
               float* vk_Color = nullptr,
               PlotVar xVar = PlotVar::ExecutionCost,
               PlotVar yVar = PlotVar::Inventory,
               PlotVar colorVar = PlotVar::Speed);
    ~Simulation();

    void step(bool waitForRender = true, bool signalRender = true);
    void run();

    Boundaries getBoundaries() const;

    void importSemaphores(int fdWait, int fdSignal);

    // Friend classes usually go at the end or top
    friend class UpdateLogicFixture;

    //    private:
    void computeLocalDensities();
    void computePressure();
    void updateAgentState(float observed_pressure);
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
};
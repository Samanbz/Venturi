#pragma once
#include <random>

#include "types.h"

class Simulation {
   public:
    Simulation(const MarketParams& params, float* vk_X = nullptr, float* vk_Y = nullptr);
    ~Simulation();

    void step();
    void run();

    BoundaryPair getBoundaries() const;

    void importSemaphores(int fdWait, int fdSignal);

    // Friend classes usually go at the end or top
    friend class UpdateLogicFixture;

    //    private:
    void computeLocalDensities();
    void computePressure();
    void updateAgentState();
    void updatePrice();

    bool externalMemoryProvided;
    bool owns_memory_ = true;  // Don't forget this from previous steps!

    MarketParams params_;
    MarketState state_;

    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;

    cudaExternalSemaphore_t cudaWaitSemaphore = nullptr;
    cudaExternalSemaphore_t cudaSignalSemaphore = nullptr;
};
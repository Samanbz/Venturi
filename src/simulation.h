#pragma once

#include "types.h"

class Simulation {
   public:
    Simulation(const MarketParams& params);
    ~Simulation();

    void step();
    MarketState getState() const;

   private:
    void initializeInventories(const float mean, const float stddev);
    void computeLocalDensities();
    void computeInventories();

    MarketParams params_;
    MarketState state_;
};
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "types.h"

/**
 * @brief Unified configuration for Simulation, Rendering, and Initialization.
 */
struct SimConfig {
    // Application Config
    bool offlineMode = false;
    std::string outputDir = "output";
    int numFrames = 5000;
    int width = 1920;
    int height = 1080;
    int targetFPS = 60;
    float speedUp = 1.0f;
    float preferredDelta = 0.01f;  // 10ms stability target
    float historyDuration = 30.0f;
    float smoothingAlpha = 0.5f;

    // Zoom Schedule
    float zoomStart = 10.0f;
    float zoomEnd = 50.0f;
    int zoomDuration = 2000;

    // Grid Config
    bool showGrid = true;
    float gridSpacing = 100.0f;

    // Timing (Derived)
    int stepsPerFrame = 1;

    // Market Parameters
    MarketParams marketParams;

    SimConfig() {
        // Defaults matched to main.cpp original values
        marketParams.num_agents = 50000;
        marketParams.num_steps = 0;      // Calculated in finalize()
        marketParams.time_delta = 0.0f;  // Calculated in finalize()

        marketParams.latency_mean = 2.0f;
        marketParams.latency_jitter_stddev = 100.0f;
        marketParams.max_latency_steps = 1024;

        marketParams.price_init = 100.0f;
        marketParams.price_randomness_stddev = 1.5f;
        marketParams.permanent_impact = 1e-5f;
        marketParams.temporary_impact = 100.0f;
        marketParams.sph_smoothing_radius = 100.0f;
        marketParams.congestion_sensitivity = 0.9f;

        marketParams.hash_table_size = 0;  // Calculated in finalize()

        marketParams.decay_rate = 1e-4f;
        marketParams.mass_alpha = 0.01f;
        marketParams.mass_beta = 0.1f;
        marketParams.risk_mean = 1.0f;
        marketParams.risk_stddev = 1.1f;
        marketParams.greed_mean = 5.0f;
        marketParams.greed_stddev = 10.0f;
        marketParams.trend_decay = 0.9f;
        marketParams.target_inventory_mean = 0.0f;
        marketParams.target_inventory_stddev = 1e2f;

        marketParams.buyer_proportion = 0.5f;
    }

    /**
     * @brief Computes derived parameters (time_delta, stepsPerFrame, hashatable)
     * based on the current configuration. call this after parsing args.
     */
    void finalize() {
        if (targetFPS <= 0)
            targetFPS = 60;

        float simTimePerFrame = speedUp / static_cast<float>(targetFPS);
        stepsPerFrame = std::max(1, static_cast<int>(std::round(simTimePerFrame / preferredDelta)));
        marketParams.time_delta = simTimePerFrame / stepsPerFrame;

        // If num_steps wasn't explicitly set (e.g. for profiling), calculate from frames
        if (marketParams.num_steps == 0) {
            marketParams.num_steps = numFrames * stepsPerFrame;
        }

        // Calculate Spatial Hash Table Size (Power of 2 > Num Agents)
        if (marketParams.hash_table_size == 0) {
            int power = 1;
            while ((1 << power) < marketParams.num_agents) {
                power++;
            }
            marketParams.hash_table_size = (1 << (power + 1));
        }
    }
};

/**
 * @brief Parses command line arguments into SimConfig.
 */
static SimConfig parseArgs(int argc, char** argv) {
    SimConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // --- Application & Rendering ---
        if (arg == "--offline") {
            config.offlineMode = true;
        } else if (arg == "--out-dir" && i + 1 < argc) {
            config.outputDir = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            config.numFrames = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            config.targetFPS = std::stoi(argv[++i]);
        } else if (arg == "--speedup" && i + 1 < argc) {
            config.speedUp = std::stof(argv[++i]);
        } else if (arg == "--history" && i + 1 < argc) {
            config.historyDuration = std::stof(argv[++i]);
        } else if (arg == "--smooth" && i + 1 < argc) {
            config.smoothingAlpha = std::stof(argv[++i]);
        }

        // --- Zoom Schedule ---
        else if (arg == "--zoom-start" && i + 1 < argc) {
            config.zoomStart = std::stof(argv[++i]);
        } else if (arg == "--zoom-end" && i + 1 < argc) {
            config.zoomEnd = std::stof(argv[++i]);
        } else if (arg == "--zoom-duration" && i + 1 < argc) {
            config.zoomDuration = std::stoi(argv[++i]);
        }

        // --- Grid Config ---
        else if (arg == "--no-grid") {
            config.showGrid = false;
        } else if (arg == "--grid-spacing" && i + 1 < argc) {
            config.gridSpacing = std::stof(argv[++i]);
        }

        // --- Market Parameters ---
        else if (arg == "--agents" && i + 1 < argc) {
            config.marketParams.num_agents = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            config.marketParams.num_steps = std::stoi(argv[++i]);
        } else if (arg == "--latency-mean" && i + 1 < argc) {
            config.marketParams.latency_mean = std::stof(argv[++i]);
        } else if (arg == "--latency-jitter" && i + 1 < argc) {
            config.marketParams.latency_jitter_stddev = std::stof(argv[++i]);
        } else if (arg == "--price-init" && i + 1 < argc) {
            config.marketParams.price_init = std::stof(argv[++i]);
        } else if (arg == "--price-vol" && i + 1 < argc) {
            config.marketParams.price_randomness_stddev = std::stof(argv[++i]);
        } else if (arg == "--impact-perm" && i + 1 < argc) {
            config.marketParams.permanent_impact = std::stof(argv[++i]);
        } else if (arg == "--impact-temp" && i + 1 < argc) {
            config.marketParams.temporary_impact = std::stof(argv[++i]);
        } else if (arg == "--congestion" && i + 1 < argc) {
            config.marketParams.congestion_sensitivity = std::stof(argv[++i]);
        } else if (arg == "--decay" && i + 1 < argc) {
            config.marketParams.decay_rate = std::stof(argv[++i]);
        } else if (arg == "--risk-mean" && i + 1 < argc) {
            config.marketParams.risk_mean = std::stof(argv[++i]);
        } else if (arg == "--risk-std" && i + 1 < argc) {
            config.marketParams.risk_stddev = std::stof(argv[++i]);
        } else if (arg == "--greed-mean" && i + 1 < argc) {
            config.marketParams.greed_mean = std::stof(argv[++i]);
        } else if (arg == "--greed-std" && i + 1 < argc) {
            config.marketParams.greed_stddev = std::stof(argv[++i]);
        } else if (arg == "--trend-decay" && i + 1 < argc) {
            config.marketParams.trend_decay = std::stof(argv[++i]);
        } else if (arg == "--buyers" && i + 1 < argc) {
            config.marketParams.buyer_proportion = std::stof(argv[++i]);
        }
    }

    config.finalize();
    return config;
}

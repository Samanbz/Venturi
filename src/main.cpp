#include "canvas.h"

int main() {
    Canvas sim{};

    sim.initWindow();
    sim.initVulkan();
    sim.mainLoop();
    sim.cleanup();

    return 0;
}
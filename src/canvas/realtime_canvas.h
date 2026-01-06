#pragma once

#include "canvas.h"

/**
 * @brief Canvas implementation for Real-Time visualization using GLFW and Vulkan SwapChain.
 */
class RealTimeCanvas : public Canvas {
   public:
    /**
     * @brief Constructs a new Real Time Canvas object.
     *
     * @param numVertices Number of particles/vertices to render.
     * @param width Window width.
     * @param height Window height.
     */
    RealTimeCanvas(size_t numVertices, uint32_t width = 1920, uint32_t height = 1080);
    ~RealTimeCanvas() override;

    /**
     * @brief Starts the simulation loop.
     */
    void run(Simulation& sim) override;

   protected:
    void initWindow() override;
    std::vector<const char*> getRequiredExtensions() override;
    void createSurface() override;
    void initSwapchainResources() override;
    void createFramebuffers() override;
    void drawFrame(Simulation& sim, bool& running) override;

   private:
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();
    void createSwapChain();

    GLFWwindow* window_;
    VkSwapchainKHR swapChain_;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkExtent2D swapChainExtent;

    VkSemaphore imageAvailableSemaphore;
    VkQueue presentQueue_;
};
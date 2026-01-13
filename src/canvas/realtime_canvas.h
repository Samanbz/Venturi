#pragma once

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "canvas.h"
#include "imgui.h"
#include "implot.h"

struct ScrollingBuffer {
    std::vector<float> Data;
    std::vector<float> Time;
    float Span = 10.0f;
    float Alpha = 1.0f;
    float LastValue = 0.0f;
    bool Initialized = false;

    void AddPoint(float x, float y) {
        float val = y;
        if (!Initialized) {
            LastValue = y;
            Initialized = true;
        } else {
            val = Alpha * y + (1.0f - Alpha) * LastValue;
            LastValue = val;
        }

        Time.push_back(x);
        Data.push_back(val);

        if (x < Span)
            return;

        // Prune old data to keep window valid
        float threshold = x - Span;
        if (!Time.empty() && Time.front() < threshold) {
            size_t count = 0;
            // Linear search is fast enough for small N
            while (count < Time.size() && Time[count] < threshold) {
                count++;
            }
            if (count > 0) {
                Time.erase(Time.begin(), Time.begin() + count);
                Data.erase(Data.begin(), Data.begin() + count);
            }
        }
    }
};

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

    void setHistoryDuration(float duration) {
        priceHistory_.Span = duration;
        pressureHistory_.Span = duration;
    }

    void setSmoothingAlpha(float alpha) {
        priceHistory_.Alpha = alpha;
        pressureHistory_.Alpha = alpha;
    }

   protected:
    void initWindow() override;
    std::vector<const char*> getRequiredExtensions() override;
    void createSurface() override;
    void initSwapchainResources() override;
    void createFramebuffers() override;
    void drawFrame(Simulation& sim, bool& running) override;
    void drawUI(VkCommandBuffer cmd) override;

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

    // UI Members
    VkDescriptorPool imguiPool_;
    ScrollingBuffer priceHistory_;
    ScrollingBuffer pressureHistory_;

    void initImGui();
    void renderUI(Simulation& sim);

    VkSemaphore imageAvailableSemaphore;
    VkQueue presentQueue_;
};
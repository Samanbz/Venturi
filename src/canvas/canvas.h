#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vulkan/vulkan.h>
#define GFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "simulation.h"
#include "types.h"

/**
 * @brief Push constants used in shaders for coordinate transformation and coloration.
 */
struct PushConstants {
    glm::mat4 p;        ///< Projection matrix (Ortho)
    float minC;         ///< Minimum color value range
    float maxC;         ///< Maximum color value range
    float w;            ///< Point weight/size or interpolation factor
    float pad;          ///< Alignment pad
    glm::vec2 scale;    ///< Reprojection scale
    glm::vec2 offset;   ///< Reprojection offset
    float trailWeight;  ///< Trail Weight (fade rate)
    float contrast;     ///< Gamma correction
};

/**
 * @brief Storage for Queue Families indices found on the Physical Device.
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value(); }
    bool isCompleteWithPresent() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

/**
 * @brief Details about SwapChain support (Capabilities, Formats, Present Modes).
 */
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

/**
 * @brief Abstract Base Class for the generic Venturi Rendering Canvas to manage Vulkan state.
 *
 * This class handles all common Vulkan initialization, including:
 * - Instance & Device creation
 * - Command Pools & Buffers
 * - Synchronization Primitive creation
 * - Offscreen Trail/Heatmap rendering resources
 * - Graphics Pipelines (Main, Fade, Copy)
 *
 * Subclasses (RealTimeCanvas, OfflineCanvas) must implement:
 * - initWindow() (optional)
 * - getRequiredExtensions()
 * - createSurface() (optional)
 * - initSwapchainResources() (for SwapChain vs Offline Image)
 * - createFramebuffers()
 * - drawFrame() (Loop logic)
 */
class Canvas {
   public:
    Canvas(size_t numVertices, uint32_t width, uint32_t height)
        : numVertices(numVertices), width_(width), height_(height) {}

    virtual ~Canvas();

    // Prevent copying
    Canvas(const Canvas&) = delete;
    Canvas& operator=(const Canvas&) = delete;

    /**
     * @brief Maps internal Vulkan memory to CUDA pointers for interop.
     */
    std::tuple<float*, float*, float*> getCudaDevicePointers();

    /**
     * @brief Exports Vulkan semaphores for external synchronization (CUDA).
     *
     * @return Pair of file descriptors {fdWait, fdSignal}.
     *         - fdWait: Semaphore to wait on (Vulkan finished reading).
     *         - fdSignal: Semaphore to signal (Vulkan finished writing, though rarely used in this
     * direction).
     */
    std::pair<int, int> exportSemaphores();

    /**
     * @brief Updates the camera/simulation boundaries for rendering mapping.
     *
     * @param boundaries Struct containing Min/Max values for X, Y, and Color.
     * @param zoomX Zoom factor for X (0.0 to 1.0). -1 to keep current.
     * @param zoomY Zoom factor for Y (0.0 to 1.0). -1 to keep current.
     * @param immediate If true, snaps to new boundaries instantly (no smoothing).
     */
    void setBoundaries(Boundaries boundaries,
                       float zoomX = -1.0f,
                       float zoomY = -1.0f,
                       bool immediate = false);

    /**
     * @brief Sets the number of simulation steps to perform per rendered frame.
     * @param steps Number of physics steps.
     */
    void setStepsPerFrame(int steps) { stepsPerFrame_ = steps; }

    /**
     * @brief Sets the target framerate cap.
     * @param fps Target FPS (e.g., 60).
     */
    void setTargetFPS(int fps) { targetFPS_ = fps; }

    /**
     * @brief Sets dynamic zoom schedule.
     *
     * @param start Initial zoom factor.
     * @param end Final zoom factor.
     * @param duration Duration in frames.
     */
    void setZoomSchedule(float start, float end, int duration);

    /**
     * @brief Configures the grid.
     *
     * @param enabled Whether to draw the grid.
     * @param spacing Pixel spacing target for grid lines.
     */
    void setGridConfig(bool enabled, float spacing) {
        gridEnabled_ = enabled;
        gridSpacing_ = spacing;
    }

    /**
     * @brief Main run loop abstract method.
     *
     * @param sim Reference to the Simulation object.
     */
    virtual void run(Simulation& sim) = 0;

   protected:
    int stepsPerFrame_ = 2;  // Default speed
    int targetFPS_ = 60;     // Default FPS
    float zoomX_ = 0.9f;
    float zoomY_ = 0.9f;

    // Grid settings
    bool gridEnabled_ = true;
    float gridSpacing_ = 100.0f;

    // Zoom Schedule State
    float zoomScheduleStart_ = 1.0f;
    float zoomScheduleEnd_ = 1.0f;
    int zoomScheduleDuration_ = 0;
    int currentFrameCount_ = 0;

    void updateZoomSchedule();

    void initVulkan();
    void createVulkanInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createVertexBuffers();
    void createRenderPass();
    void createGraphicsPipeline();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();

    // Mode-specific initialization hooks

    /**
     * @brief Optional window initialization for RealTime canvas.
     */
    virtual void initWindow() {}  // Only RealTime

    /**
     * @brief Returns list of Vulkan instance extensions required by the surface/window.
     */
    virtual std::vector<const char*> getRequiredExtensions() = 0;

    /**
     * @brief Optional surface creation for SwapChain based canvases.
     */
    virtual void createSurface() {}  // Only RealTime

    /**
     * @brief Initializes SwapChain or Offscreen images.
     */
    virtual void initSwapchainResources() = 0;  // Swapchain vs Offline Image

    /**
     * @brief Creates Framebuffers wrapping the target images.
     */
    virtual void createFramebuffers() = 0;

    /**
     * @brief Records and submits command buffers for a single frame.
     */
    virtual void drawFrame(Simulation& sim, bool& running) = 0;

    /**
     * @brief Render ImGui UI elements in the render pass (Optional hook).
     */
    virtual void drawUI(VkCommandBuffer cmd) {}

    // Common Resources
    void createOffscreenResources();
    void createOffscreenRenderPass();
    void createFadePipeline();
    void createGridPipeline();
    void createCopyPipeline();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSet();
    void createTextureSampler();

    // Helpers
    VkShaderModule createShaderModule(const std::vector<char>& code);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    int getMemoryFd(VkDeviceMemory memory);
    int getSemaphoreFd(VkSemaphore semaphore);
    void recordCommandBuffer(VkCommandBuffer commandBuffer,
                             VkFramebuffer framebuffer,
                             VkExtent2D extent,
                             bool endCommandBuffer = true);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    // Members
    uint32_t width_;
    uint32_t height_;
    size_t numVertices;

    VkInstance instance_;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_;
    VkQueue graphicsQueue_;
    // Present Queue is effectively same as Graphics Queue in this simplified single-queue setup, or
    // handled by RealTime subclass
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    // Vertex Buffers
    VkBuffer xBuffer, yBuffer, colorBuffer;
    VkDeviceMemory xBufferMemory, yBufferMemory, colorBufferMemory;

    // Rendering Pipeline Common
    VkRenderPass renderPass;  // The final "Output" renderpass (Swapchain or Offline Image)
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkPipeline offscreenGraphicsPipeline;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    // Synchronization
    VkSemaphore renderFinishedSemaphore;  // Vulkan finished rendering (signal to CUDA or Present)
    VkSemaphore cudaFinishedSemaphore;    // CUDA finished simulation (wait on this)
    VkSemaphore vulkanFinishedSemaphore;  // For export
    VkFence inFlightFence;

    // Offscreen Trail Effect Resources (Common)
    VkImage offscreenImages[2];
    VkDeviceMemory offscreenImageMemories[2];
    VkImageView offscreenImageViews[2];
    VkRenderPass offscreenRenderPass;
    VkFramebuffer offscreenFramebuffers[2];

    VkPipeline fadePipeline;
    VkPipelineLayout fadePipelineLayout;

    VkPipeline gridPipeline;
    VkPipelineLayout gridPipelineLayout;

    VkPipeline copyPipeline;
    VkPipelineLayout copyPipelineLayout;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSets[2];  // Set 0 reads [1], Set 1 reads [0]
    VkSampler textureSampler;

    // Camera Boundaries
    float minX = 0.0f, maxX = 1.0f;
    float minY = 0.0f, maxY = 1.0f;
    float minColor = 0.0f, maxColor = 1.0f;

    // Previous frame boundaries for reprojection
    float prevMinX = 0.0f, prevMaxX = 1.0f;
    float prevMinY = 0.0f, prevMaxY = 1.0f;
    uint32_t currentTrailIndex = 0;

    float targetMinX = 0.0f, targetMaxX = 1.0f;
    float targetMinY = 0.0f, targetMaxY = 1.0f;
    float targetMinColor = 0.0f, targetMaxColor = 1.0f;

    // Format decided by subclass
    VkFormat swapChainImageFormat;

    // Flag to help logic separation
    bool isOffline_ = false;

    const std::vector<const char*> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
};

#pragma once

// Force GLM to use radians
#define GLM_FORCE_RADIANS
// Force GLM to use Vulkan's Depth Range (0.0 to 1.0) instead of OpenGL's (-1.0 to 1.0)
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vulkan/vulkan.h>
#define GFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <optional>
#include <vector>

#include "simulation.h"
#include "types.h"
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class Canvas {
   public:
    Canvas(size_t numVertices) : numVertices(numVertices) {
        initWindow();
        initVulkan();
    }

    ~Canvas() { cleanup(); }

    /**
     * @brief Get CUDA device pointers mapped to the Vulkan vertex buffers
     *
     * @return std::pair<float*, float*> The CUDA device pointers for the X and Y vertex buffers
     */
    std::pair<float*, float*> getCudaDevicePointers();

    /**
     * @brief Export the semaphores for synchronization with CUDA
     *
     * @return std::pair<int, int>
     */
    std::pair<int, int> exportSemaphores();

    /**
     * @brief Set the axis boundaries for rendering
     *
     * @param boundaries The boundary pairs for the Y and X axes
     * @param padX Padding to apply to the X axis boundaries (in percent of range)
     * @param padY Padding to apply to the Y axis boundaries (in percent of range)
     * @param immediate Whether to set the boundaries immediately or interpolate to them
     */
    void setBoundaries(BoundaryPair boundaries,
                       float padX = 0.1f,
                       float padY = 0.1f,
                       bool immediate = false);

    //    private:
    /**
     * @brief Initialize the GLFW window
     */
    void initWindow();

    /**
     * @brief Create a Vulkan Instance object
     */
    void createVulkanInstance();

    /**
     * @brief Check if the required device extensions are supported
     *
     * @param device The physical device to check
     * @return true if all required extensions are supported
     * @return false otherwise
     */
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    /**
     * @brief Find the queue families supported by the given device
     *
     * @param device The physical device to query
     * @return QueueFamilyIndices The indices of the supported queue families
     */
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    /**
     * @brief Query the swap chain support details for a given physical device
     *
     * @param device The physical device to query
     * @return SwapChainSupportDetails The details of the swap chain support
     */
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    /**
     * @brief Check if a physical device is suitable for the application's needs based on available
     * extensions, swap chain support, and queue families
     *
     * @param device The physical device to check
     * @return true if the device is suitable
     * @return false otherwise
     */
    bool isDeviceSuitable(VkPhysicalDevice device);

    /**
     * @brief Pick a suitable physical device (GPU) for the application
     */
    void pickPhysicalDevice();

    /**
     * @brief Create a Logical Device object
     */
    void createLogicalDevice();

    /**
     * @brief Find a suitable memory type based on the type filter and required properties
     *
     * @param typeFilter The type filter to match against
     * @param properties The required memory properties
     * @return uint32_t The index of the suitable memory type
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    /**
     * @brief Create Vertex Buffer objects
     */
    void createVertexBuffers();

    /**
     * @brief Get the file descriptor of the vertex buffer.
     *
     * @returns int The file descriptor of the vertex buffer memory.
     */
    int getMemoryFd(VkDeviceMemory memory);

    /**
     * @brief Choose the best surface format for the swap chain
     *
     * @param availableFormats The list of available surface formats
     * @return VkSurfaceFormatKHR The chosen surface format
     */
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats);

    /**
     * @brief Choose the best present mode for the swap chain
     *
     * @param availablePresentModes The list of available present modes
     * @return VkPresentModeKHR The chosen present mode
     */
    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes);

    /**
     * @brief Choose the swap extent (resolution) for the swap chain
     *
     * @param capabilities The surface capabilities
     * @return VkExtent2D The chosen swap extent
     */
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    /**
     * @brief Create a Surface object
     */
    void createSurface();

    /**
     * @brief Create a Swap Chain object
     */
    void createSwapChain();

    /**
     * @brief Create Image Views for the swap chain images
     */
    void createImageViews();

    /**
     * @brief Create a Render Pass object
     */
    void createRenderPass();

    /**
     * @brief Create a Shader Module object
     *
     * @param code The SPIR-V bytecode of the shader
     * @return VkShaderModule The created shader module
     */
    VkShaderModule createShaderModule(const std::vector<char>& code);

    /**
     * @brief Create a Graphics Pipeline object
     */
    void createGraphicsPipeline();

    /**
     * @brief Create a Framebuffers object
     */
    void createFramebuffers();

    /**
     * @brief Create a Command Pool object
     */
    void createCommandPool();

    /**
     * @brief Create a Command Buffer object
     */
    void createCommandBuffer();

    /**
     * @brief Record commands into the command buffer
     *
     * @param commandBuffer The command buffer to record into
     * @param imageIndex The index of the swap chain image to render to
     */
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    /**
     * @brief Create a semaphore that can be exported for inter-process or inter-API synchronization
     *
     * @return VkSemaphore The created exportable semaphore
     */
    VkSemaphore createExportableSemaphore();

    /**
     * @brief Get the file descriptor of a given semaphore
     *
     * @param VkSemaphore semaphore The semaphore to get the FD for
     * @return int The file descriptor of the semaphore
     */
    int getSemaphoreFd(VkSemaphore semaphore);

    /**
     * @brief Create synchronization objects
     */
    void createSyncObjects();

    /**
     * @brief Initialize Vulkan components
     */
    void initVulkan();

    /**
     * @brief The main rendering loop
     */
    void mainLoop(Simulation& sim);

    /**
     * @brief Draw a single frame
     */
    void drawFrame();

    /**
     * @brief Cleanup Vulkan resources
     */
    void cleanup();

    const uint32_t WIDTH = 1920;
    const uint32_t HEIGHT = 1080;

    const std::vector<const char*> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    float minX = 0.0f;
    float maxX = 1.0f;
    float minY = 0.0f;
    float maxY = 1.0f;

    float targetMinX = 0.0f;
    float targetMaxX = 1.0f;
    float targetMinY = 0.0f;
    float targetMaxY = 1.0f;

    GLFWwindow* window_;
    VkInstance instance_;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_;
    VkSurfaceKHR surface_;

    VkQueue graphicsQueue_;
    VkQueue presentQueue_;

    size_t numVertices;
    VkBuffer xBuffer;
    VkBuffer yBuffer;
    VkDeviceMemory xBufferMemory;
    VkDeviceMemory yBufferMemory;

    VkSwapchainKHR swapChain_;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkRenderPass renderPass;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkSemaphore cudaFinishedSemaphore;
    VkSemaphore vulkanFinishedSemaphore;

    VkFence inFlightFence;
};
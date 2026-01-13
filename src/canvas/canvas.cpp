#include "canvas.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "utils.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::seconds;

void Canvas::setZoomSchedule(float start, float end, int duration) {
    zoomScheduleStart_ = start;
    zoomScheduleEnd_ = end;
    zoomScheduleDuration_ = duration;
    currentFrameCount_ = 0;
}

void Canvas::updateZoomSchedule() {
    if (zoomScheduleDuration_ <= 0)
        return;

    if (currentFrameCount_ <= zoomScheduleDuration_) {
        float t =
            static_cast<float>(currentFrameCount_) / static_cast<float>(zoomScheduleDuration_);
        // Linear interpolation of zoom factor
        float currentZoom = zoomScheduleStart_ + (zoomScheduleEnd_ - zoomScheduleStart_) * t;

        zoomX_ = currentZoom;
        zoomY_ = currentZoom;
    }
    currentFrameCount_++;
}

Canvas::~Canvas() {
    vkDeviceWaitIdle(device_);

    vkDestroyBuffer(device_, xBuffer, nullptr);
    vkDestroyBuffer(device_, yBuffer, nullptr);
    vkDestroyBuffer(device_, colorBuffer, nullptr);
    vkFreeMemory(device_, xBufferMemory, nullptr);
    vkFreeMemory(device_, yBufferMemory, nullptr);
    vkFreeMemory(device_, colorBufferMemory, nullptr);

    vkDestroySemaphore(device_, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device_, cudaFinishedSemaphore, nullptr);
    vkDestroySemaphore(device_, vulkanFinishedSemaphore, nullptr);
    vkDestroyFence(device_, inFlightFence, nullptr);

    vkDestroyCommandPool(device_, commandPool, nullptr);

    vkDestroyPipeline(device_, fadePipeline, nullptr);
    vkDestroyPipelineLayout(device_, fadePipelineLayout, nullptr);
    vkDestroyPipeline(device_, copyPipeline, nullptr);
    vkDestroyPipelineLayout(device_, copyPipelineLayout, nullptr);

    vkDestroyRenderPass(device_, offscreenRenderPass, nullptr);
    for (int i = 0; i < 2; i++) {
        vkDestroyFramebuffer(device_, offscreenFramebuffers[i], nullptr);
        vkDestroyImageView(device_, offscreenImageViews[i], nullptr);
        vkDestroyImage(device_, offscreenImages[i], nullptr);
        vkFreeMemory(device_, offscreenImageMemories[i], nullptr);
    }
    vkDestroySampler(device_, textureSampler, nullptr);

    vkDestroyDescriptorPool(device_, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);

    vkDestroyPipeline(device_, offscreenGraphicsPipeline, nullptr);
    vkDestroyPipeline(device_, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
    vkDestroyRenderPass(device_, renderPass, nullptr);

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }

    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

void Canvas::initVulkan() {
    createVulkanInstance();
    // For RealTime, we need surface before picking physical device
    if (!isOffline_) {
        createSurface();
    }

    pickPhysicalDevice();
    createLogicalDevice();

    createVertexBuffers();

    // Choose format before creating render pass
    if (isOffline_) {
        swapChainImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
    } else {
        SwapChainSupportDetails support = querySwapChainSupport(physicalDevice_);
        // Simple selection logic: prefer B8G8R8A8 SRGB
        swapChainImageFormat = support.formats[0].format;  // Default
        for (const auto& fmt : support.formats) {
            if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
                fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                swapChainImageFormat = fmt.format;
                break;
            }
        }
    }

    // Create Render Passes
    createRenderPass();
    createOffscreenRenderPass();

    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFadePipeline();
    createCopyPipeline();

    createCommandPool();

    createOffscreenResources();
    createTextureSampler();
    createDescriptorPool();
    createDescriptorSet();

    initSwapchainResources();
    createFramebuffers();

    createCommandBuffer();
    createSyncObjects();
}

void Canvas::createVulkanInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Venturi Simulation";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = 0;

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }
}

void Canvas::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (const auto& device : devices) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool complete = false;
        if (isOffline_)
            complete = indices.isComplete();
        else
            complete = indices.isCompleteWithPresent();

        // Simplified extension check for brevity (assuming support if found basically, or rely on
        // validation) Re-adding extension check logic:
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                             availableExtensions.data());
        std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(),
                                                 DEVICE_EXTENSIONS.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        bool extensionsSupported = requiredExtensions.empty();

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            if (isOffline_) {
                swapChainAdequate = true;
            } else {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate =
                    !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }
        }

        if (complete && extensionsSupported && swapChainAdequate) {
            physicalDevice_ = device;
            break;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to find a suitable GPU");
}

void Canvas::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value()};
    if (indices.presentFamily.has_value()) {
        uniqueQueueFamilies.insert(indices.presentFamily.value());
    }

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.wideLines = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    createInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
}

QueueFamilyIndices Canvas::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        if (!isOffline_) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
        }

        if (isOffline_) {
            if (indices.graphicsFamily.has_value())
                break;
        } else {
            if (indices.isCompleteWithPresent())
                break;
        }
        i++;
    }
    return indices;
}

VkShaderModule Canvas::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return shaderModule;
}

uint32_t Canvas::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

int Canvas::getMemoryFd(VkDeviceMemory memory) {
    auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    func(device_, &fdInfo, &fd);
    return fd;
}

int Canvas::getSemaphoreFd(VkSemaphore semaphore) {
    auto func = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR");
    VkSemaphoreGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fdInfo.semaphore = semaphore;
    fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd;
    func(device_, &fdInfo, &fd);
    return fd;
}

std::tuple<float*, float*, float*> Canvas::getCudaDevicePointers() {
    VkDeviceSize bufferSize = numVertices * sizeof(float);
    int fdX = getMemoryFd(xBufferMemory);
    int fdY = getMemoryFd(yBufferMemory);
    int fdColor = getMemoryFd(colorBufferMemory);
    float* ptrX = mapFDToCudaPointer(fdX, bufferSize);
    float* ptrY = mapFDToCudaPointer(fdY, bufferSize);
    float* ptrColor = mapFDToCudaPointer(fdColor, bufferSize);
    return {ptrX, ptrY, ptrColor};
}

std::pair<int, int> Canvas::exportSemaphores() {
    int fdVulkanFinished = getSemaphoreFd(vulkanFinishedSemaphore);
    int fdCudaFinished = getSemaphoreFd(cudaFinishedSemaphore);
    return {fdVulkanFinished, fdCudaFinished};
}

void Canvas::setBoundaries(Boundaries boundaries, float zoomX, float zoomY, bool immediate) {
    if (zoomX > 0.0f) {
        zoomX_ = zoomX;
    }
    if (zoomY > 0.0f) {
        zoomY_ = zoomY;
    }

    float rangeX = boundaries.maxX - boundaries.minX;
    float rangeY = boundaries.maxY - boundaries.minY;

    // Handle Zoom (Maintain Center at 0,0)
    float centerX = 0.0f;
    float centerY = 0.0f;

    float newRangeX = rangeX / zoomX_;
    float newRangeY = rangeY / zoomY_;

    targetMinY = centerY - newRangeY * 0.5f;
    targetMaxY = centerY + newRangeY * 0.5f;
    targetMinX = centerX - newRangeX * 0.5f;
    targetMaxX = centerX + newRangeX * 0.5f;

    targetMinColor = boundaries.minColor;
    targetMaxColor = boundaries.maxColor;

    if (immediate) {
        minY = targetMinY;
        maxY = targetMaxY;
        minX = targetMinX;
        maxX = targetMaxX;
        minColor = targetMinColor;
        maxColor = targetMaxColor;
    }
}

// Implementing condensed versions
void Canvas::createVertexBuffers() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = numVertices * sizeof(float);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkMemoryRequirements memReq;
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    allocInfo.pNext = &exportInfo;

    auto createBuf = [&](VkBuffer& buf, VkDeviceMemory& mem) {
        vkCreateBuffer(device_, &bufferInfo, nullptr, &buf);
        vkGetBufferMemoryRequirements(device_, buf, &memReq);
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device_, &allocInfo, nullptr, &mem);
        vkBindBufferMemory(device_, buf, mem, 0);
    };

    createBuf(xBuffer, xBufferMemory);
    createBuf(yBuffer, yBufferMemory);
    createBuf(colorBuffer, colorBufferMemory);
}

void Canvas::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = isOffline_ ? VK_FORMAT_R8G8B8A8_SRGB : swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout =
        isOffline_ ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void Canvas::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_);
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool);
}

void Canvas::createCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);
}

void Canvas::createSyncObjects() {
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &exportInfo;

    vkCreateSemaphore(device_, &semInfo, nullptr, &cudaFinishedSemaphore);
    vkCreateSemaphore(device_, &semInfo, nullptr, &vulkanFinishedSemaphore);  // Exported

    semInfo.pNext = nullptr;
    vkCreateSemaphore(device_, &semInfo, nullptr, &renderFinishedSemaphore);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFence);
}

void Canvas::recordCommandBuffer(VkCommandBuffer commandBuffer,
                                 VkFramebuffer framebuffer,
                                 VkExtent2D extent,
                                 bool endCommandBuffer) {
    size_t bufferSize = numVertices;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    uint32_t currIndex = currentTrailIndex;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.image = offscreenImages[currIndex];
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &barrier);

    // Pass 1: Offscreen
    VkRenderPassBeginInfo offscreenInfo{};
    offscreenInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    offscreenInfo.renderPass = offscreenRenderPass;
    offscreenInfo.framebuffer = offscreenFramebuffers[currIndex];
    offscreenInfo.renderArea.extent = extent;
    vkCmdBeginRenderPass(commandBuffer, &offscreenInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {0.0f, 0.0f, (float) extent.width, (float) extent.height, 0.0f, 1.0f};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, extent};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Fade
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fadePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fadePipelineLayout, 0,
                            1, &descriptorSets[currIndex], 0, nullptr);

    float prevW = prevMaxX - prevMinX;
    float prevH = prevMaxY - prevMinY;
    if (std::abs(prevW) < 1e-5f)
        prevW = 1.0f;
    if (std::abs(prevH) < 1e-5f)
        prevH = 1.0f;

    PushConstants fadePush{};
    fadePush.scale = {(maxX - minX) / prevW, (maxY - minY) / prevH};
    fadePush.offset = {(minX - prevMinX) / prevW, (prevMaxY - maxY) / prevH};
    fadePush.trailWeight = 0.07f;  // This was fadeRate

    vkCmdPushConstants(commandBuffer, fadePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &fadePush);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    // Points (trail)
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreenGraphicsPipeline);
    PushConstants push;
    push.p = glm::ortho(minX, maxX, maxY, minY, -1.0f, 1.0f);
    push.minC = minColor;
    push.maxC = maxColor;
    push.w = 0.05f;
    push.trailWeight = 0.05f;  // Alpha for trail accumulation

    // Adaptive constrast based on symmetricity
    bool symmetric = (minColor < 0.0f);
    push.contrast = symmetric ? 1.3f : 1.0f;  // >1.0 makes a smooth bell-curve blend to white

    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstants), &push);

    VkBuffer vBufs[] = {xBuffer, yBuffer, colorBuffer};
    VkDeviceSize offsets[] = {0, 0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 3, vBufs, offsets);
    vkCmdDraw(commandBuffer, static_cast<uint32_t>(bufferSize), 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    // Transition back to Read Only
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);

    // Pass 2: Final
    VkRenderPassBeginInfo finalInfo{};
    finalInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    finalInfo.renderPass = renderPass;
    finalInfo.framebuffer = framebuffer;
    finalInfo.renderArea.extent = extent;
    VkClearValue clear = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    finalInfo.clearValueCount = 1;
    finalInfo.pClearValues = &clear;

    vkCmdBeginRenderPass(commandBuffer, &finalInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipeline);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    // Bind the set that reads the image we JUST wrote to (currIndex)
    // The set that reads 'currIndex' is 'currIndex+1' (mod 2).
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout, 0,
                            1, &descriptorSets[(currIndex + 1) % 2], 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    // Points (Live)
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    push.w = 1.0f;
    push.trailWeight = 0.9f;  // Make live points fully opaque
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstants), &push);
    vkCmdBindVertexBuffers(commandBuffer, 0, 3, vBufs, offsets);
    vkCmdDraw(commandBuffer, static_cast<uint32_t>(bufferSize), 1, 0, 0);

    drawUI(commandBuffer);

    vkCmdEndRenderPass(commandBuffer);

    if (endCommandBuffer) {
        vkEndCommandBuffer(commandBuffer);
    }

    prevMinX = minX;
    prevMaxX = maxX;
    prevMinY = minY;
    prevMaxY = maxY;
    currentTrailIndex = (currentTrailIndex + 1) % 2;
}

void Canvas::createOffscreenRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // Transitioned at end of pass

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    vkCreateRenderPass(device_, &renderPassInfo, nullptr, &offscreenRenderPass);
}

void Canvas::createOffscreenResources() {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo allocCmdInfo{};
    allocCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandPool = commandPool;
    allocCmdInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &allocCmdInfo, &cb);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &beginInfo);

    for (int i = 0; i < 2; i++) {
        vkCreateImage(device_, &imageInfo, nullptr, &offscreenImages[i]);

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device_, offscreenImages[i], &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkAllocateMemory(device_, &allocInfo, nullptr, &offscreenImageMemories[i]);
        vkBindImageMemory(device_, offscreenImages[i], offscreenImageMemories[i], 0);

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image = offscreenImages[i];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkClearColorValue clearColor = {0.0f, 0.0f, 0.0f, 0.0f};
        VkImageSubresourceRange subRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, offscreenImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &subRange);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = offscreenImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device_, &viewInfo, nullptr, &offscreenImageViews[i]);

        VkImageView attachments[] = {offscreenImageViews[i]};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = offscreenRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = attachments;
        fbInfo.width = width_;
        fbInfo.height = height_;
        fbInfo.layers = 1;
        vkCreateFramebuffer(device_, &fbInfo, nullptr, &offscreenFramebuffers[i]);
    }

    vkEndCommandBuffer(cb);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cb;

    vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue_);
    vkFreeCommandBuffers(device_, commandPool, 1, &cb);
}

void Canvas::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler);
}

void Canvas::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &samplerLayoutBinding;

    vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout);
}

void Canvas::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 2;
    vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool);
}

void Canvas::createDescriptorSet() {
    std::vector<VkDescriptorSetLayout> layouts(2, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts.data();

    // Allocate 2 sets
    if (vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (int i = 0; i < 2; i++) {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // i=0 reads from 1, i=1 reads from 0
        imageInfo.imageView = offscreenImageViews[(i + 1) % 2];
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device_, 1, &descriptorWrite, 0, nullptr);
    }
}

void Canvas::createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");
    VkShaderModule vertModule = createShaderModule(vertShaderCode);
    VkShaderModule fragModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStageInfo, fragStageInfo};

    VkVertexInputBindingDescription bindings[] = {{0, 4, VK_VERTEX_INPUT_RATE_VERTEX},
                                                  {1, 4, VK_VERTEX_INPUT_RATE_VERTEX},
                                                  {2, 4, VK_VERTEX_INPUT_RATE_VERTEX}};
    VkVertexInputAttributeDescription attribs[] = {{0, 0, VK_FORMAT_R32_SFLOAT, 0},
                                                   {1, 1, VK_FORMAT_R32_SFLOAT, 0},
                                                   {2, 2, VK_FORMAT_R32_SFLOAT, 0}};

    VkPipelineVertexInputStateCreateInfo vertInput{};
    vertInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertInput.vertexBindingDescriptionCount = 3;
    vertInput.pVertexBindingDescriptions = bindings;
    vertInput.vertexAttributeDescriptionCount = 3;
    vertInput.pVertexAttributeDescriptions = attribs;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAsm.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.lineWidth = isOffline_ ? 2.0f : 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.blendEnable = VK_TRUE;
    blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
    blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAtt.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.logicOpEnable = VK_FALSE;
    colorBlend.logicOp = VK_LOGIC_OP_COPY;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynState.pDynamicStates = dynStates.data();

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;

    vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertInput;
    pipeInfo.pInputAssemblyState = &inputAsm;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisample;
    pipeInfo.pColorBlendState = &colorBlend;
    pipeInfo.pDynamicState = &dynState;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = offscreenRenderPass;
    pipeInfo.subpass = 0;
    pipeInfo.basePipelineHandle = VK_NULL_HANDLE;

    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                              &offscreenGraphicsPipeline);

    pipeInfo.renderPass = renderPass;
    blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &graphicsPipeline);

    vkDestroyShaderModule(device_, vertModule, nullptr);
    vkDestroyShaderModule(device_, fragModule, nullptr);
}

void Canvas::createCopyPipeline() {
    auto vertCode = readFile("shaders/quad.spv");
    auto fragCode = readFile("shaders/copy.spv");
    VkShaderModule v = createShaderModule(vertCode);
    VkShaderModule f = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = v;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = f;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStageInfo, fragStageInfo};

    VkPipelineVertexInputStateCreateInfo emptyInput{};
    emptyInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    emptyInput.vertexBindingDescriptionCount = 0;
    emptyInput.pVertexBindingDescriptions = nullptr;
    emptyInput.vertexAttributeDescriptionCount = 0;
    emptyInput.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAsm.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.sampleShadingEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    ba.alphaBlendOp = VK_BLEND_OP_ADD;
    ba.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.logicOpEnable = VK_FALSE;
    cb.logicOp = VK_LOGIC_OP_COPY;
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;

    std::vector<VkDynamicState> ds = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(ds.size());
    dyn.pDynamicStates = ds.data();

    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &descriptorSetLayout;

    vkCreatePipelineLayout(device_, &pl, nullptr, &copyPipelineLayout);

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &emptyInput;
    pi.pInputAssemblyState = &inputAsm;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = copyPipelineLayout;
    pi.renderPass = renderPass;
    pi.subpass = 0;
    pi.basePipelineHandle = VK_NULL_HANDLE;

    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &copyPipeline);

    vkDestroyShaderModule(device_, v, nullptr);
    vkDestroyShaderModule(device_, f, nullptr);
}

void Canvas::createFadePipeline() {
    auto vertCode = readFile("shaders/fade.vert.spv");
    auto fragCode = readFile("shaders/fade.frag.spv");
    VkShaderModule v = createShaderModule(vertCode);
    VkShaderModule f = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = v;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = f;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStageInfo, fragStageInfo};

    VkPipelineVertexInputStateCreateInfo emptyInput{};
    emptyInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    emptyInput.vertexBindingDescriptionCount = 0;
    emptyInput.pVertexBindingDescriptions = nullptr;
    emptyInput.vertexAttributeDescriptionCount = 0;
    emptyInput.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAsm.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.sampleShadingEnable = VK_FALSE;

    // Correct Blending: Pure Decay.
    // DstRGB = DstRGB * (1 - SrcAlpha)
    // Src color (RGB) is ignored (Multiplied by ZERO).
    // Src Alpha determines how much of Dst is kept.
    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable = VK_FALSE;
    ba.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.logicOpEnable = VK_FALSE;
    cb.logicOp = VK_LOGIC_OP_COPY;
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;

    std::vector<VkDynamicState> ds = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(ds.size());
    dyn.pDynamicStates = ds.data();

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset = 0;
    pc.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &descriptorSetLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pc;

    vkCreatePipelineLayout(device_, &pl, nullptr, &fadePipelineLayout);

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &emptyInput;
    pi.pInputAssemblyState = &inputAsm;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = fadePipelineLayout;
    pi.renderPass = offscreenRenderPass;
    pi.subpass = 0;
    pi.basePipelineHandle = VK_NULL_HANDLE;

    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &fadePipeline);

    vkDestroyShaderModule(device_, v, nullptr);
    vkDestroyShaderModule(device_, f, nullptr);
}

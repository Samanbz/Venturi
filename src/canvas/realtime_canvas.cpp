#include "realtime_canvas.h"

#include <algorithm>
#include <chrono>
#include <thread>

using namespace std::chrono;

RealTimeCanvas::RealTimeCanvas(size_t numVertices, uint32_t width, uint32_t height)
    : Canvas(numVertices, width, height) {
    isOffline_ = false;
    initWindow();
    initVulkan();
    // Present semaphore
    VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
    vkCreateSemaphore(device_, &info, nullptr, &imageAvailableSemaphore);

    // Setup queues - needed for present
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
    vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
}

RealTimeCanvas::~RealTimeCanvas() {
    vkDestroySemaphore(device_, imageAvailableSemaphore, nullptr);
    for (auto fb : swapChainFramebuffers)
        vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto iv : swapChainImageViews)
        vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapChain_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void RealTimeCanvas::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width_, height_, "Venturi Simulation", nullptr, nullptr);
}

std::vector<const char*> RealTimeCanvas::getRequiredExtensions() {
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    return std::vector<const char*>(exts, exts + count);
}

void RealTimeCanvas::createSurface() {
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");
}

SwapChainSupportDetails Canvas::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount,
                                             details.formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount,
                                                  details.presentModes.data());
    }
    return details;
}
VkSurfaceFormatKHR RealTimeCanvas::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& fmt : availableFormats)
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return fmt;
    return availableFormats[0];
}

VkPresentModeKHR RealTimeCanvas::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& mode : availablePresentModes)
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            return mode;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D RealTimeCanvas::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    VkExtent2D actual = {(uint32_t) w, (uint32_t) h};
    actual.width = std::clamp(actual.width, capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width);
    actual.height = std::clamp(actual.height, capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return actual;
}

void RealTimeCanvas::initSwapchainResources() {
    createSwapChain();
    createImageViews();
}

void RealTimeCanvas::createSwapChain() {
    SwapChainSupportDetails support = querySwapChainSupport(physicalDevice_);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
    VkExtent2D extent = chooseSwapExtent(support.capabilities);
    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
    uint32_t qInd[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = qInd;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void RealTimeCanvas::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device_, &createInfo, nullptr, &swapChainImageViews[i]);
    }
}

void RealTimeCanvas::createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {swapChainImageViews[i]};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;
        vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
    }
}

void RealTimeCanvas::drawFrame(Simulation& sim, bool& running) {
    vkWaitForFences(device_, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &inFlightFence);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(device_, swapChain_, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE,
                          &imageIndex);

    vkResetCommandBuffer(commandBuffer, 0);
    recordCommandBuffer(commandBuffer, swapChainFramebuffers[imageIndex], swapChainExtent);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore, cudaFinishedSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore, vulkanFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFence);

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
    VkSwapchainKHR swapChains[] = {swapChain_};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    vkQueuePresentKHR(presentQueue_, &presentInfo);

    if (glfwWindowShouldClose(window_))
        running = false;
}

void RealTimeCanvas::run(Simulation& sim) {
    const int targetFPS = 60;
    const microseconds frameDuration(1000000 / targetFPS);

    // Note: Main calls sim.step() once before calling run(), so the pipeline is primed.
    bool running = true;
    while (running) {
        auto frameStart = high_resolution_clock::now();
        glfwPollEvents();

        drawFrame(sim, running);  // waits for cudaFinished

        for (int i = 0; i < stepsPerFrame_; ++i) {
            bool waitForRender = (i == 0);
            bool signalRender = (i == stepsPerFrame_ - 1);
            sim.step(waitForRender, signalRender);
        }

        // Update boundaries
        float alpha = 0.005f;
        minX += (targetMinX - minX) * alpha;
        maxX += (targetMaxX - maxX) * alpha;
        minY += (targetMinY - minY) * alpha;
        maxY += (targetMaxY - maxY) * alpha;
        minColor += (targetMinColor - minColor) * alpha;
        maxColor += (targetMaxColor - maxColor) * alpha;

        setBoundaries(sim.getBoundaries());

        auto frameEnd = high_resolution_clock::now();
        auto elapsed = duration_cast<microseconds>(frameEnd - frameStart);
        if (elapsed < frameDuration)
            std::this_thread::sleep_for(frameDuration - elapsed);
    }
}
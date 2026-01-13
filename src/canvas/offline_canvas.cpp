#include "offline_canvas.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../utils/simple_plotter.h"

OfflineCanvas::OfflineCanvas(size_t numVertices,
                             uint32_t width,
                             uint32_t height,
                             const std::string& outputDir,
                             int numFrames)
    : Canvas(numVertices, width, height), outputDir_(outputDir), numFrames_(numFrames) {
    isOffline_ = true;
    initVulkan();
    // Start writer thread
    stopWriter_ = false;
    writerThread_ = std::thread(&OfflineCanvas::writerLoop, this);
}

OfflineCanvas::~OfflineCanvas() {
    // Shutdown writer thread
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        stopWriter_ = true;
    }
    queueCv_.notify_one();
    if (writerThread_.joinable()) {
        writerThread_.join();
    }

    if (mappedData_) {
        vkUnmapMemory(device_, downloadBufferMemory_);
    }

    vkDestroyFramebuffer(device_, offlineFramebuffer_, nullptr);
    vkDestroyImageView(device_, offlineImageView_, nullptr);
    vkDestroyImage(device_, offlineImage_, nullptr);
    vkFreeMemory(device_, offlineImageMemory_, nullptr);
    vkDestroyBuffer(device_, downloadBuffer_, nullptr);
    vkFreeMemory(device_, downloadBufferMemory_, nullptr);
}

std::vector<const char*> OfflineCanvas::getRequiredExtensions() {
    return {};
}

void OfflineCanvas::initSwapchainResources() {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateImage(device_, &imageInfo, nullptr, &offlineImage_);

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device_, offlineImage_, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &allocInfo, nullptr, &offlineImageMemory_);
    vkBindImageMemory(device_, offlineImage_, offlineImageMemory_, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = offlineImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device_, &viewInfo, nullptr, &offlineImageView_);

    // Download buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = width_ * height_ * 4;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bufferInfo, nullptr, &downloadBuffer_);

    vkGetBufferMemoryRequirements(device_, downloadBuffer_, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memReq.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device_, &allocInfo, nullptr, &downloadBufferMemory_);
    vkBindBufferMemory(device_, downloadBuffer_, downloadBufferMemory_, 0);

    // Persistent Map
    vkMapMemory(device_, downloadBufferMemory_, 0, width_ * height_ * 4, 0, &mappedData_);
}

void OfflineCanvas::createFramebuffers() {
    VkImageView attachments[] = {offlineImageView_};
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = width_;
    framebufferInfo.height = height_;
    framebufferInfo.layers = 1;
    vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &offlineFramebuffer_);
}

void OfflineCanvas::drawFrame(Simulation& sim, bool& running) {
    vkWaitForFences(device_, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &inFlightFence);
    vkResetCommandBuffer(commandBuffer, 0);

    // Record main rendering (do not end buffer)
    recordCommandBuffer(commandBuffer, offlineFramebuffer_, {width_, height_}, false);

    // Image Transition for Copy
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image = offlineImage_;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyImageToBuffer(commandBuffer, offlineImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           downloadBuffer_, 1, &region);

    vkEndCommandBuffer(commandBuffer);

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = {cudaFinishedSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};

    // Only wait for CUDA if it's not the first frame (first frame has no preceding simulation step)
    if (currentFrame_ > 0) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
    } else {
        submitInfo.waitSemaphoreCount = 0;
    }

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VkSemaphore signalSemaphores[] = {vulkanFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFence);
}

void OfflineCanvas::saveFrame(int frameNum) {
    vkWaitForFences(device_, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    // mappedData_ is already valid

    // Get buffer from Recycler
    std::vector<unsigned char> pixels;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!recycleQueue_.empty()) {
            pixels = std::move(recycleQueue_.front());
            recycleQueue_.pop();
        }
    }

    if (pixels.empty()) {
        pixels.resize(width_ * height_ * 4);
    }

    // Direct Memcpy
    std::memcpy(pixels.data(), mappedData_, width_ * height_ * 4);

    // Draw Plots
    if (!priceHistory_.empty()) {
        int plotW = width_ / 4;
        int plotH = height_ / 6;
        int margin = 20;

        // Price (Top) - Cyan
        float minP = *std::min_element(priceHistory_.begin(), priceHistory_.end());
        float maxP = *std::max_element(priceHistory_.begin(), priceHistory_.end());
        if (std::abs(maxP - minP) < 0.1f) {
            maxP += 0.1f;
            minP -= 0.1f;
        }  // Prevent div by zero

        SimplePlotter::PlotLine(pixels, width_, height_, priceHistory_, minP, maxP, margin, margin,
                                plotW, plotH, "Price", 0, 255, 255);

        // Pressure (Below) - Magenta
        float minPr = *std::min_element(pressureHistory_.begin(), pressureHistory_.end());
        float maxPr = *std::max_element(pressureHistory_.begin(), pressureHistory_.end());
        if (std::abs(maxPr - minPr) < 0.1f) {
            maxPr += 0.1f;
            minPr -= 0.1f;
        }

        SimplePlotter::PlotLine(pixels, width_, height_, pressureHistory_, minPr, maxPr, margin,
                                margin + plotH + 10, plotW, plotH, "Pressure", 255, 0, 255);
    }

    // Push to queue
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        writeQueue_.push({frameNum, std::move(pixels)});
    }
    queueCv_.notify_one();
}

void OfflineCanvas::writerLoop() {
    while (true) {
        FrameData frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] { return !writeQueue_.empty() || stopWriter_; });

            if (writeQueue_.empty() && stopWriter_) {
                return;
            }
            frame = std::move(writeQueue_.front());
            writeQueue_.pop();
        }

        // Write to disk
        std::stringstream ss;
        ss << outputDir_ << "/frame_" << std::setfill('0') << std::setw(5) << frame.frameNum
           << ".ppm";
        std::ofstream file(ss.str(), std::ios::binary);
        file << "P6\n" << width_ << " " << height_ << "\n255\n";

        std::vector<unsigned char> row(width_ * 3);
        const unsigned char* src = frame.pixels.data();

        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                size_t i = (y * width_ + x) * 4;
                row[x * 3] = src[i];
                row[x * 3 + 1] = src[i + 1];
                row[x * 3 + 2] = src[i + 2];
            }
            file.write(reinterpret_cast<const char*>(row.data()), width_ * 3);
        }

        // Recycle the buffer
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            // Don't keep too many in memory if writer is fast
            if (recycleQueue_.size() < 10) {
                recycleQueue_.push(std::move(frame.pixels));
            }
        }
    }
}

void OfflineCanvas::run(Simulation& sim) {
    auto startTime = std::chrono::steady_clock::now();
    auto lastUpdate = startTime;
    double smoothedFPS = 0.0;
    int lastUpdateFrame = 0;

    int framesToRender = numFrames_;

    std::cout << "Starting offline rendering..." << std::endl;
    std::cout << "Total Frames: " << framesToRender << std::endl;
    std::cout << "Output Directory: " << outputDir_ << std::endl;

    // Initial boundary setup
    setBoundaries(sim.getBoundaries(), -1.0f, -1.0f, true);

    for (int i = 0; i < numFrames_; ++i) {
        currentFrame_ = i;

        // Smooth camera movement
        // Alpha adjusted to match RealTime (approx 0.01 at 60Hz ~= 0.02 at 30Hz)
        float alpha = 0.001f;
        minX += (targetMinX - minX) * alpha;
        maxX += (targetMaxX - maxX) * alpha;
        minY += (targetMinY - minY) * alpha;
        maxY += (targetMaxY - maxY) * alpha;
        minColor += (targetMinColor - minColor) * alpha;
        maxColor += (targetMaxColor - maxColor) * alpha;

        bool r = true;
        drawFrame(sim, r);

        // Collect stats
        priceHistory_.push_back(sim.state_.price);
        pressureHistory_.push_back(sim.state_.pressure);

        saveFrame(i);  // This waits for fence

        for (int k = 0; k < stepsPerFrame_; ++k) {
            bool waitForRender = (k == 0);
            bool signalRender = (k == stepsPerFrame_ - 1);
            sim.step(false, signalRender);
        }

        // Update targets for next frame
        setBoundaries(sim.getBoundaries());  // Non-immediate

        // Progress Bar & Time Estimation
        auto now = std::chrono::steady_clock::now();
        double timeSinceLastUpdate = std::chrono::duration<double>(now - lastUpdate).count();

        // Update every 0.25 seconds or at the end
        if (timeSinceLastUpdate > 0.25 || i == numFrames_ - 1) {
            float progress = (float) (i + 1) / framesToRender;
            int barWidth = 50;
            int pos = barWidth * progress;

            double totalElapsed = std::chrono::duration<double>(now - startTime).count();

            // Calculate instant FPS for this block
            double blockFPS = (i + 1 - lastUpdateFrame) / timeSinceLastUpdate;

            // Exponential Smoothing
            if (lastUpdateFrame == 0) {
                smoothedFPS = blockFPS;
            } else {
                smoothedFPS = 0.3 * blockFPS + 0.7 * smoothedFPS;
            }

            float remaining = (framesToRender - (i + 1)) / std::max(0.1, smoothedFPS);

            std::cout << "\r[";
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos)
                    std::cout << "=";
                else if (j == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "% "
                      << "Elapsed: " << int(totalElapsed) << "s "
                      << "ETA: " << int(remaining) << "s "
                      << "FPS: " << std::fixed << std::setprecision(1) << smoothedFPS << "   "
                      << std::flush;

            lastUpdate = now;
            lastUpdateFrame = i + 1;
        }
    }
    std::cout << std::endl << "Rendering complete!" << std::endl;
}

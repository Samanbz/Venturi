#include "offline_canvas.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

OfflineCanvas::OfflineCanvas(size_t numVertices,
                             uint32_t width,
                             uint32_t height,
                             const std::string& outputDir,
                             int numFrames)
    : Canvas(numVertices, width, height), outputDir_(outputDir), numFrames_(numFrames) {
    isOffline_ = true;
    initVulkan();
}

OfflineCanvas::~OfflineCanvas() {
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

    recordCommandBuffer(commandBuffer, offlineFramebuffer_, {width_, height_});

    // Append Copy command
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.image = offscreenImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &barrier);
    // Offscreen Pass
    VkRenderPassBeginInfo offscreenInfo{};
    offscreenInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    offscreenInfo.renderPass = offscreenRenderPass;
    offscreenInfo.framebuffer = offscreenFramebuffer;
    offscreenInfo.renderArea.extent = {width_, height_};
    vkCmdBeginRenderPass(commandBuffer, &offscreenInfo, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport = {0.0f, 0.0f, (float) width_, (float) height_, 0.0f, 1.0f};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {width_, height_}};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fadePipeline);
    float fadeRate = 0.05f;
    vkCmdPushConstants(commandBuffer, fadePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(float), &fadeRate);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreenGraphicsPipeline);
    PushConstants push;
    push.p = glm::ortho(minX, maxX, maxY, minY, -1.0f, 1.0f);
    push.minC = minColor;
    push.maxC = maxColor;
    push.w = 0.5f;
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstants), &push);
    VkBuffer vBufs[] = {xBuffer, yBuffer, colorBuffer};
    VkDeviceSize offsets[] = {0, 0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 3, vBufs, offsets);
    vkCmdDraw(commandBuffer, static_cast<uint32_t>(numVertices), 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // Final Pass
    VkRenderPassBeginInfo finalInfo{};
    finalInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    finalInfo.renderPass = renderPass;
    finalInfo.framebuffer = offlineFramebuffer_;
    finalInfo.renderArea.extent = {width_, height_};
    VkClearValue clear = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    finalInfo.clearValueCount = 1;
    finalInfo.pClearValues = &clear;
    vkCmdBeginRenderPass(commandBuffer, &finalInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipeline);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout, 0,
                            1, &descriptorSet, 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    push.w = 1.0f;
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstants), &push);
    vkCmdBindVertexBuffers(commandBuffer, 0, 3, vBufs, offsets);
    vkCmdDraw(commandBuffer, static_cast<uint32_t>(numVertices), 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // Extra Copy Step
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image = offlineImage_;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
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
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VkSemaphore signalSemaphores[] = {vulkanFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFence);
}

void OfflineCanvas::saveFrame(int frameNum) {
    vkWaitForFences(device_, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    void* data;
    vkMapMemory(device_, downloadBufferMemory_, 0, width_ * height_ * 4, 0, &data);

    std::stringstream ss;
    ss << outputDir_ << "/frame_" << std::setfill('0') << std::setw(5) << frameNum << ".ppm";
    std::ofstream file(ss.str(), std::ios::binary);
    file << "P6\n" << width_ << " " << height_ << "\n255\n";
    unsigned char* pixels = static_cast<unsigned char*>(data);
    std::vector<unsigned char> row(width_ * 3);
    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            size_t i = (y * width_ + x) * 4;
            row[x * 3] = pixels[i];
            row[x * 3 + 1] = pixels[i + 1];
            row[x * 3 + 2] = pixels[i + 2];
        }
        file.write(reinterpret_cast<const char*>(row.data()), width_ * 3);
    }
    vkUnmapMemory(device_, downloadBufferMemory_);
}

void OfflineCanvas::run(Simulation& sim) {
    auto startTime = std::chrono::steady_clock::now();
    int framesToRender = numFrames_;

    std::cout << "Starting offline rendering..." << std::endl;
    std::cout << "Total Frames: " << framesToRender << std::endl;
    std::cout << "Output Directory: " << outputDir_ << std::endl;

    // 2 steps per frame default
    for (int i = 0; i < numFrames_; ++i) {
        bool r = true;
        drawFrame(sim, r);
        saveFrame(i);  // This waits for fence

        sim.step(true, true);

        // Progress Bar & Time Estimation
        if (i % 10 == 0 || i == numFrames_ - 1) {
            float progress = (float) (i + 1) / framesToRender;
            int barWidth = 50;
            int pos = barWidth * progress;

            auto now = std::chrono::steady_clock::now();
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            float framesPerSec = (i + 1) / (float) std::max(1L, elapsed);
            float remaining = (framesToRender - (i + 1)) / framesPerSec;

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
                      << "Elapsed: " << elapsed << "s "
                      << "ETA: " << int(remaining) << "s    " << std::flush;
        }

        setBoundaries(sim.getBoundaries(), 0.1f, 0.1f, true);  // Immediate update for consistency
    }
    std::cout << std::endl << "Rendering complete!" << std::endl;
}

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "canvas.h"

/**
 * @brief Canvas implementation for Offline rendering (headless).
 */
class OfflineCanvas : public Canvas {
   public:
    /**
     * @brief Constructs a new Offline Canvas.
     *
     * @param numVertices Number of particles.
     * @param width Image width.
     * @param height Image height.
     * @param outputDir Directory to save frames.
     * @param numFrames Total frames to simulate and render.
     */
    OfflineCanvas(size_t numVertices,
                  uint32_t width,
                  uint32_t height,
                  const std::string& outputDir,
                  int numFrames);
    ~OfflineCanvas() override;

    void run(Simulation& sim) override;

   protected:
    std::vector<const char*> getRequiredExtensions() override;
    void initSwapchainResources() override;
    void createFramebuffers() override;
    void drawFrame(Simulation& sim, bool& running) override;

   private:
    void saveFrame(int frameNum);

    // Async Writer
    void writerLoop();

    std::string outputDir_;
    int numFrames_;
    int currentFrame_ = 0;

    VkImage offlineImage_;
    VkDeviceMemory offlineImageMemory_;
    VkImageView offlineImageView_;
    VkFramebuffer offlineFramebuffer_;

    VkBuffer downloadBuffer_;
    VkDeviceMemory downloadBufferMemory_;
    void* mappedData_ = nullptr;  // Persistent mapping

    // Threading components
    struct FrameData {
        int frameNum;
        std::vector<unsigned char> pixels;
    };

    std::thread writerThread_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::queue<FrameData> writeQueue_;
    std::queue<std::vector<unsigned char>> recycleQueue_;  // Optimization: Buffer Pooling
    std::atomic<bool> stopWriter_{false};
};
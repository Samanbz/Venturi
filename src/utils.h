#include <cuda.h>
#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>
#include <vector>

/**
 * @brief Reads a raw binary file into a byte vector.
 *
 * @param filename Absolute or relative path to the file.
 * @return std::vector<char> containing file contents.
 * @throws std::runtime_error if file cannot be opened.
 */
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

/**
 * @brief Maps a file descriptor (from Vulkan external memory) to a CUDA device pointer.
 *
 * @param fd File descriptor representing the external memory.
 * @param size Size of the memory block in bytes.
 * @return float* Mapped CUDA device pointer.
 */
float* mapFDToCudaPointer(int fd, size_t size) {
    // Define the import properties
    cudaExternalMemoryHandleDesc desc{};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;

    // Import the external memory block
    cudaExternalMemory_t externalMem;
    cudaError_t err = cudaImportExternalMemory(&externalMem, &desc);
    if (err != cudaSuccess) {
        // cudaImportExternalMemory takes ownership of the FD,
        // so we don't need to close(fd) manually if successful.
        throw std::runtime_error("Failed to import Vulkan memory to CUDA");
    }

    // Map it to a pointer
    float* devicePtr = nullptr;
    cudaExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    err = cudaExternalMemoryGetMappedBuffer((void**) &devicePtr, externalMem, &bufferDesc);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to map CUDA pointer");
    }

    return devicePtr;
}
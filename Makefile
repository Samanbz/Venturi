# Venturi Makefile
# Wraps CMake for convenient building and testing

BUILD_DIR := build
CMAKE := cmake
CTEST := ctest

.PHONY: all build test clean rebuild configure run test-verbose benchmark help

# Default target
all: build

# Configure CMake (run once or after CMakeLists.txt changes)
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc ..

# Build the project
build: configure
	@$(CMAKE) --build $(BUILD_DIR) -j$(shell nproc)

# Run the main executable
run: build
	@./$(BUILD_DIR)/venturi

# Run all tests
test: build
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure

# Run tests with verbose output
test-verbose: build
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure --verbose

# Run benchmarks
benchmark: build
	@if [ -f $(BUILD_DIR)/venturi_benchmarks ]; then \
		./$(BUILD_DIR)/venturi_benchmarks; \
	else \
		echo "Benchmarks not built. Make sure Google Benchmark is installed."; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf output/
	@echo "Cleaned build and output directories."

# Full rebuild
rebuild: clean build

# Help
help:
	@echo "Venturi Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the project (default)"
	@echo "  build        - Build the project"
	@echo "  run          - Build and run the main executable"
	@echo "  test         - Build and run all tests"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  clean        - Remove build directory"
	@echo "  rebuild      - Clean and rebuild"
	@echo "  configure    - Run CMake configuration"
	@echo "  help         - Show this help message"

ffmpeg:
	ffmpeg -framerate 30 -i output/frame_%05d.ppm -c:v libx264 -pix_fmt yuv420p -preset veryslow -crf 0 -tune animation output.mp4